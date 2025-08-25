from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from core.utils import get_groq_key
import os
from dotenv import load_dotenv
import time
import litellm
from datetime import datetime
import signal
import sys
import json
import threading
import asyncio
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import queue

# FastAPI imports
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

load_dotenv()

# Pydantic models for request/response validation
class ThoughtModel(BaseModel):
    timestamp: str
    agent: str
    type: str
    message: str
    metadata: Dict[str, Any] = {}


class AgentStateModel(BaseModel):
    last_activity: str
    status: str
    message: str


class SystemStatusModel(BaseModel):
    running: bool
    paused: bool = False
    cycle_count: int
    last_cycle_time: Optional[str] = None
    next_cycle_time: Optional[str] = None
    interval_minutes: float = 2.0


class ControlRequest(BaseModel):
    pass


class IntervalRequest(BaseModel):
    minutes: float


class ControlResponse(BaseModel):
    status: str
    message: str
    running: Optional[bool] = None
    paused: Optional[bool] = None
    interval_minutes: Optional[float] = None


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


class AgentThoughtsAPI:
    """FastAPI-based API to expose agent thoughts via HTTP and WebSocket"""

    def __init__(self, port: int = 5000):
        self.app = FastAPI(
            title="Market Making Agent Thoughts API",
            description="Real-time API for monitoring market making agents",
            version="1.0.0",
        )
        self.port = port
        self.thoughts_log: List[ThoughtModel] = []
        self.agent_states: Dict[str, AgentStateModel] = {}
        self.system_controller = None
        self.websocket_manager = WebSocketManager()

        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.setup_routes()

    def set_system_controller(self, controller):
        """Set reference to the main system for control operations"""
        self.system_controller = controller

    def setup_routes(self):
        """Setup all API routes"""

        @self.app.get("/")
        async def root():
            return {
                "message": "Market Making Agent Thoughts API",
                "version": "1.0.0",
                "websocket_url": f"ws://localhost:{self.port}/ws/thoughts",
                "docs_url": f"http://localhost:{self.port}/docs",
            }

        # ===== THOUGHTS API =====
        @self.app.get("/api/thoughts", response_model=Dict[str, Any])
        async def get_thoughts():
            """Get recent agent thoughts via HTTP"""
            return {
                "thoughts": [thought.dict() for thought in self.thoughts_log[-50:]],
                "agent_states": {k: v.dict() for k, v in self.agent_states.items()},
                "total_thoughts": len(self.thoughts_log),
            }

        @self.app.get("/api/agents", response_model=Dict[str, AgentStateModel])
        async def get_agents():
            """Get current agent states"""
            return {k: v for k, v in self.agent_states.items()}

        # ===== SYSTEM CONTROL API =====
        @self.app.get("/api/control/status", response_model=SystemStatusModel)
        async def get_system_status():
            """Get current system status"""
            if not self.system_controller:
                raise HTTPException(
                    status_code=500, detail="System controller not available"
                )

            return SystemStatusModel(
                running=self.system_controller.running,
                paused=getattr(self.system_controller, "paused", False),
                cycle_count=self.system_controller.cycle_count,
                last_cycle_time=getattr(
                    self.system_controller, "last_cycle_time", None
                ),
                next_cycle_time=getattr(
                    self.system_controller, "next_cycle_time", None
                ),
                interval_minutes=getattr(
                    self.system_controller, "interval_minutes", 2.0
                ),
            )

        @self.app.post("/api/control/start", response_model=ControlResponse)
        async def start_system():
            """Start/resume the market making system"""
            if not self.system_controller:
                raise HTTPException(
                    status_code=500, detail="System controller not available"
                )

            try:
                self.system_controller.start_execution()
                await self.log_thought("System", "control", "System started via API")
                return ControlResponse(
                    status="success",
                    message="Market making system started",
                    running=True,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/control/stop", response_model=ControlResponse)
        async def stop_system():
            """Stop the market making system"""
            if not self.system_controller:
                raise HTTPException(
                    status_code=500, detail="System controller not available"
                )

            try:
                self.system_controller.stop_execution()
                await self.log_thought("System", "control", "System stopped via API")
                return ControlResponse(
                    status="success",
                    message="Market making system stopped",
                    running=False,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/control/pause", response_model=ControlResponse)
        async def pause_system():
            """Pause the market making system"""
            if not self.system_controller:
                raise HTTPException(
                    status_code=500, detail="System controller not available"
                )

            try:
                self.system_controller.pause_execution()
                await self.log_thought("System", "control", "System paused via API")
                return ControlResponse(
                    status="success", message="Market making system paused", paused=True
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/control/resume", response_model=ControlResponse)
        async def resume_system():
            """Resume the paused market making system"""
            if not self.system_controller:
                raise HTTPException(
                    status_code=500, detail="System controller not available"
                )

            try:
                self.system_controller.resume_execution()
                await self.log_thought("System", "control", "System resumed via API")
                return ControlResponse(
                    status="success",
                    message="Market making system resumed",
                    paused=False,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/control/trigger-cycle", response_model=ControlResponse)
        async def trigger_immediate_cycle():
            """Trigger an immediate cycle execution"""
            if not self.system_controller:
                raise HTTPException(
                    status_code=500, detail="System controller not available"
                )

            try:
                self.system_controller.trigger_immediate_cycle()
                await self.log_thought(
                    "System", "control", "Immediate cycle triggered via API"
                )
                return ControlResponse(
                    status="success", message="Immediate cycle triggered"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/control/set-interval", response_model=ControlResponse)
        async def set_cycle_interval(request: IntervalRequest):
            """Set the cycle interval in minutes"""
            if not self.system_controller:
                raise HTTPException(
                    status_code=500, detail="System controller not available"
                )

            try:
                if request.minutes < 0.1:  # Minimum 6 seconds
                    raise HTTPException(
                        status_code=400,
                        detail="Minimum interval is 0.1 minutes (6 seconds)",
                    )

                self.system_controller.set_interval(request.minutes)
                await self.log_thought(
                    "System",
                    "control",
                    f"Cycle interval set to {request.minutes} minutes via API",
                )
                return ControlResponse(
                    status="success",
                    message=f"Cycle interval set to {request.minutes} minutes",
                    interval_minutes=request.minutes,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ===== WEBSOCKET ENDPOINT =====
        @self.app.websocket("/ws/thoughts")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time agent thoughts"""
            await self.websocket_manager.connect(websocket)
            print(f"[API] Client connected to thoughts stream")

            try:
                # Send connection confirmation
                await websocket.send_json(
                    {
                        "type": "connected",
                        "message": "Connected to agent thoughts stream",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # Keep connection alive
                while True:
                    await websocket.receive_text()  # Wait for any message from client
            except WebSocketDisconnect:
                print(f"[API] Client disconnected from thoughts stream")
                self.websocket_manager.disconnect(websocket)
            except Exception as e:
                print(f"[API] WebSocket error: {e}")
                self.websocket_manager.disconnect(websocket)

    async def log_thought(
        self,
        agent_name: str,
        thought_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log agent thought and broadcast via WebSocket"""
        timestamp = datetime.now().isoformat()

        thought = ThoughtModel(
            timestamp=timestamp,
            agent=agent_name,
            type=thought_type,  # 'thinking', 'action', 'result', 'error', 'control', 'cycle_start', etc.
            message=message,
            metadata=metadata or {},
        )

        # Add to log (keep last 100)
        self.thoughts_log.append(thought)
        if len(self.thoughts_log) > 100:
            self.thoughts_log = self.thoughts_log[-100:]

        # Update agent state
        self.agent_states[agent_name] = AgentStateModel(
            last_activity=timestamp, status=thought_type, message=message
        )

        # Broadcast via WebSocket
        await self.websocket_manager.broadcast(
            {"type": "agent_thought", "data": thought.dict()}
        )

        # Print to console
        print(f"[{agent_name}] {thought_type.upper()}: {message}")

    def log_thought_sync(
        self,
        agent_name: str,
        thought_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Synchronous wrapper for log_thought (for backward compatibility)"""
        # Create new event loop in thread if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If loop is running, create a task
            asyncio.create_task(
                self.log_thought(agent_name, thought_type, message, metadata)
            )
        else:
            # If loop is not running, run it
            loop.run_until_complete(
                self.log_thought(agent_name, thought_type, message, metadata)
            )

    def start_server(self):
        """Start the FastAPI server in background thread"""

        def run_server():
            uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="info")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        print(f"[API] Agent thoughts API started on http://localhost:{self.port}")
        print(f"[API] WebSocket: ws://localhost:{self.port}/ws/thoughts")
        print(f"[API] API docs: http://localhost:{self.port}/docs")
        print(f"[API] HTTP endpoints: http://localhost:{self.port}/api/thoughts")


class RetryingLLM(LLM):
    def __init__(self, retries=5, backoff=2, api=None, agent_name="Unknown", **kwargs):
        super().__init__(**kwargs)
        self.retries = retries
        self.backoff = backoff
        self.api = api
        self.agent_name = agent_name

    def _safe_tokens(self, messages):
        """Truncate overly long prompts before sending to Groq."""
        for m in messages:
            if m.get("content"):
                tokens = len(m["content"].split())
                if tokens > self.max_prompt_tokens:
                    m["content"] = " ".join(
                        m["content"].split()[: self.max_prompt_tokens]
                    )
        return messages

    def call(self, *args, **kwargs):
        if self.api:
            self.api.log_thought_sync(
                self.agent_name, "thinking", "Processing LLM request..."
            )

        for i in range(self.retries):
            try:
                result = super().call(*args, **kwargs)
                if self.api:
                    self.api.log_thought_sync(
                        self.agent_name,
                        "result",
                        f"LLM call successful (attempt {i+1})",
                    )
                return result
            except litellm.RateLimitError as e:
                wait = self.backoff**i
                if self.api:
                    self.api.log_thought_sync(
                        self.agent_name,
                        "error",
                        f"Rate limit hit, retrying in {wait}s ({i+1}/{self.retries})",
                    )
                time.sleep(wait)

        error_msg = "LLM call failed after all retries"
        if self.api:
            self.api.log_thought_sync(self.agent_name, "error", error_msg)
        raise Exception(error_msg)


# Custom Agent class to capture thoughts
class MonitoredAgent(Agent):
    def __init__(self, api=None, **kwargs):
        super().__init__(**kwargs)
        self.api = api
        self.agent_name = kwargs.get("role", "Unknown Agent")

    def execute_task(self, task):
        if self.api:
            self.api.log_thought_sync(
                self.agent_name, "action", f"Starting task: {task.description[:100]}..."
            )

        try:
            result = super().execute_task(task)
            if self.api:
                self.api.log_thought_sync(
                    self.agent_name, "result", f"Task completed successfully"
                )
            return result
        except Exception as e:
            if self.api:
                self.api.log_thought_sync(
                    self.agent_name, "error", f"Task failed: {str(e)}"
                )
            raise


class MarketMakingSystem:
    def __init__(self, api_port=5000):
        self.running = True
        self.paused = False
        self.cycle_count = 0
        self.interval_minutes = 2.0
        self.last_cycle_time = None
        self.next_cycle_time = None
        self.api = AgentThoughtsAPI(port=api_port)
        self.api.set_system_controller(self)
        self.setup_signal_handlers()
        self.initialize_system()

    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)

    def shutdown_handler(self, signum, frame):
        """Handle graceful shutdown"""
        print(
            f"\n[{datetime.now()}] Shutdown signal received. Stopping market making system..."
        )
        self.running = False

    # Control methods for API
    def start_execution(self):
        """Start the execution system"""
        self.running = True
        self.paused = False

    def stop_execution(self):
        """Stop the execution system"""
        self.running = False
        self.paused = False

    def pause_execution(self):
        """Pause the execution system"""
        self.paused = True

    def resume_execution(self):
        """Resume the paused execution system"""
        self.paused = False

    def trigger_immediate_cycle(self):
        """Trigger an immediate cycle (implement as needed)"""
        # This would be implemented based on your threading model
        pass

    def set_interval(self, minutes: float):
        """Set the cycle interval"""
        self.interval_minutes = minutes

    def initialize_system(self):
        """Initialize all system components"""
        print(f"[{datetime.now()}] Initializing Market Making System...")

        # Start API server
        self.api.start_server()
        time.sleep(2)  # Give server time to start

        # Setup LLM
        groq_key = get_groq_key()
        os.environ["GROQ_API_KEY"] = get_groq_key()
        os.environ["OPENAI_API_BASE"] = "https://api.cerebras.ai/v1/chat/completions"
        os.environ["OPENAI_API_KEY"] = os.getenv("CEREBRAS_API_KEY")

        max_response_tokens = 256

        # MCP Server configuration
        servers = [
            {
                "url": os.getenv("MCP_SEVER_API", "http://localhost:3000/mcp"),
                "transport": "streamable-http",
            }
        ]

        # Initialize MCP adapters
        try:
            self.bot_tools = MCPServerAdapter([servers[0]])
            print(
                f"Market Analyzer tools available: {[tool.name for tool in self.bot_tools.tools]}"
            )

            # Create LLMs for each agent with API integration
            researcher_llm = RetryingLLM(
                max_tokens=max_response_tokens,
                retries=5,
                backoff=2,
                model="cerebras/llama-3.3-70b",
                temperature=0.7,
                api=self.api,
                agent_name="Market Researcher",
            )

            pricer_llm = RetryingLLM(
                max_tokens=max_response_tokens,
                retries=5,
                backoff=2,
                model="cerebras/llama-3.3-70b",
                temperature=0.7,
                api=self.api,
                agent_name="Pricing Strategist",
            )

            executive_llm = RetryingLLM(
                max_tokens=max_response_tokens,
                retries=5,
                backoff=2,
                model="cerebras/llama-3.3-70b",
                temperature=0.7,
                api=self.api,
                agent_name="Executive Trader",
            )

            # ===== AGENTS WITH MONITORING =====
            self.market_researcher = MonitoredAgent(
                role="Senior Market Research and Analysis Specialist",
                goal="""Conduct comprehensive market research and analysis across all supported markets. 
                Identify trading opportunities, basically identify market with spread from 1 percent higher, analyze asset fundamentals, and provide detailed market insights 
                for informed decision-making.""",
                backstory="""You are an experienced market researcher with deep expertise in cryptocurrency markets, 
                asset analysis, and market structure. You specialize in identifying profitable trading opportunities 
                by analyzing market data, asset characteristics, and Spread in a particular market orderbook, to see where you can provide
                liquidiy and profit from spread, bid/ask. Your research forms the 
                foundation for trading strategies and investment decisions. You have extensive knowledge of technical 
                analysis, fundamental analysis, and quantitative methods for evaluating digital assets.""",
                tools=self.bot_tools.tools,
                verbose=True,
                llm=researcher_llm,
                max_iter=4,
                api=self.api,
            )

            self.pricer = MonitoredAgent(
                role="Senior Market Price Strategist",
                goal="""
                        Ascertain the Amount in Hyper fill vault, 
                        decide based on amount in vault what order size to use for the buy and sell side of the market order,
                        query the proper mid price based on bid and ask price of the particular pair in question
                    """,
                backstory="""You are a Senior Market Price Strategist with deep experience in crypto markets and market-making.
                You analyze orderbooks, bid/ask spreads,
                    and pool reserves to compute reliable mid-prices,
                    then size buy and sell orders based on the vault balance and measured liquidity.
                    You prioritize safe execution, balanced inventory, 
                    and profitable spread capture while observing risk limits and market impact.""",
                tools=self.bot_tools.tools,
                verbose=True,
                llm=pricer_llm,
                max_iter=4,
                api=self.api,
            )

            self.executive_trader = MonitoredAgent(
                role="Executive Trading Operations Manager",
                goal="""Execute market making operations based on research and pricing analysis. 
                Manage vault assets, deploy trading bots, and oversee the complete trading workflow 
                from asset allocation to active market making.""",
                backstory="""You are a seasoned Executive Trading Operations Manager with expertise in 
                automated trading systems and risk management. You translate market research and pricing 
                strategies into actionable trading operations. You have deep knowledge of DeFi protocols, 
                smart contract interactions, and automated market making systems. Your role is to execute 
                the strategic decisions made by the research and pricing teams, ensuring proper asset 
                management, bot deployment, and continuous monitoring of trading operations. You prioritize 
                capital efficiency, risk management, and operational excellence.""",
                tools=self.bot_tools.tools,
                verbose=True,
                llm=executive_llm,
                max_iter=6,
                api=self.api,
            )

            self.create_tasks()
            self.create_crew()

        except Exception as e:
            print(f"Error initializing system: {e}")
            import traceback

            traceback.print_exc()

    def create_tasks(self):
        """Create the workflow tasks"""
        self.market_discovery_task = Task(
            description="""
            Perform comprehensive market discovery and analysis:
            
            1. Get all supported markets using get_supported_markets
            2. For each supported market, fetch its orderbook
            3. Analyze the asset landscape and identify key characteristics:
            4. If the spread is drifting (non-stationary / ADF p-value ≥ 0.05 or a persistent trend), stop passive liquidity provision — widen quotes, reduce size, or hedge — until stationarity returns.

            Focus on understanding the complete market ecosystem available for analysis.
            """,
            expected_output="""
            A comprehensive market discovery report containing:
            - Complete list of supported markets
            - Asset count and breakdown per market
            - One asset with the highest Promise
            """,
            agent=self.market_researcher,
        )

        self.pricing_task = Task(
            description="""
            Based on current bid and ask, set the bid and ask price at around a profitable percentage from mid price for asset pair:
            
            1. Get the Balance of the Vault and the underlying asset which should be SEI
            2. Randomly decide based on amount in vault what order size to put in
            3. Ascertain the proper mid price for that pair
            4. Properly set the starting bid and ask price spread gap percentage
            5. This is a spread strategy we want to get the best prices possible for profitability also with frequent trades
            
            Focus on understanding spread strategy and setting the best price possible.
            """,
            expected_output="""
            A comprehensive pricing strategy report containing:
            - Recommended order size based on vault balance
            - The specific trading pair to enter
            - The optimal spread percentage for market entry
            - fetch vault balance details
            - current amount in vault
            - Calculated mid price and recommended bid/ask prices
            """,
            agent=self.pricer,
        )

        self.executive_trading_task = Task(
            description="""
            Execute the complete market making workflow based on the research and pricing analysis from previous tasks:
            
            1. Review the market research findings and selected trading pair
            2. Validate the pricing strategy and order sizing recommendations
            3. Check current vault balance and ensure sufficient funds
            4. Move required SEI token from vault to trading wallet there is a tool for that, you just need to insert the asset amount to move
            5. Deploy the market maker bot with the recommended configuration:
               - Use the identified trading pair from research
               - Apply the calculated order size from pricing analysis
               - Set the optimal spread percentage
               - Configure the reference price based on mid-price analysis
            6. Monitor initial bot deployment and ensure proper operation
            7. Provide comprehensive execution report
            
            Execute the full workflow using the start_market_making_workflow tool or individual tools as needed.
            Ensure all operations are executed safely with proper error handling.
            """,
            expected_output="""
            A comprehensive execution report containing:
            - Confirmation of vault asset movement
            - Market maker bot deployment status
            - Active trading pair and configuration details
            - Initial order placement confirmation
            - Risk management checks completed
            - Next steps for monitoring and optimization
            """,
            agent=self.executive_trader,
            context=[self.market_discovery_task, self.pricing_task],
        )

    def create_crew(self):
        """Create the crew with all agents and tasks"""
        self.market_analysis_crew = Crew(
            agents=[self.market_researcher, self.pricer, self.executive_trader],
            tasks=[
                self.market_discovery_task,
                self.pricing_task,
                self.executive_trading_task,
            ],
            verbose=True,
            process=Process.sequential,
            memory=False,
            llm=self.market_researcher.llm,  # Use one of the LLMs as default
        )

    def run_single_cycle(self):
        """Run a single market making cycle"""
        if self.paused:
            self.api.log_thought_sync(
                "System", "paused", "Cycle skipped - system is paused"
            )
            return None

        self.cycle_count += 1
        self.last_cycle_time = datetime.now().isoformat()

        print(f"\n{'='*80}")
        print(f"STARTING MARKET MAKING CYCLE #{self.cycle_count}")
        print(f"Time: {datetime.now()}")
        print(f"{'='*80}\n")

        # Log cycle start to API
        self.api.log_thought_sync(
            "System", "cycle_start", f"Starting cycle #{self.cycle_count}"
        )

        try:
            result = self.market_analysis_crew.kickoff()

            self.api.log_thought_sync(
                "System",
                "cycle_complete",
                f"Cycle #{self.cycle_count} completed successfully",
            )

            print(f"\n{'='*80}")
            print(f"CYCLE #{self.cycle_count} COMPLETE")
            print(f"{'='*80}")
            print(f"Result: {result}")

            return result

        except Exception as e:
            error_msg = f"Cycle #{self.cycle_count} failed: {str(e)}"
            self.api.log_thought_sync("System", "cycle_error", error_msg)
            print(f"Error in cycle #{self.cycle_count}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def run_continuous(self, interval_minutes=2):
        """Run the market making system continuously"""
        self.interval_minutes = interval_minutes
        interval_seconds = interval_minutes * 60

        print(f"\n{'='*80}")
        print(f"STARTING CONTINUOUS MARKET MAKING SYSTEM")
        print(f"Cycle Interval: {interval_minutes} minutes")
        print(f"API Available at: http://localhost:{self.api.port}")
        print(f"API Docs: http://localhost:{self.api.port}/docs")
        print(f"Press Ctrl+C to stop gracefully")
        print(f"{'='*80}\n")

        self.api.log_thought_sync(
            "System", "startup", f"System starting with {interval_minutes}min intervals"
        )

        while self.running:
            try:
                # Calculate next cycle time
                self.next_cycle_time = datetime.fromtimestamp(
                    time.time() + interval_seconds
                ).isoformat()

                # Run cycle
                self.run_single_cycle()

                if not self.running:
                    break

                # Wait for next cycle
                if not self.paused:
                    print(f"\nWaiting {interval_minutes} minutes until next cycle...")
                    self.api.log_thought_sync(
                        "System",
                        "waiting",
                        f"Waiting {interval_minutes} minutes for next cycle",
                    )

                # Use the current interval (might have been changed via API)
                current_interval = self.interval_minutes * 60
                for i in range(int(current_interval)):
                    if not self.running:
                        break
                    time.sleep(1)

            except KeyboardInterrupt:
                print(f"\nReceived interrupt signal...")
                break
            except Exception as e:
                print(f"Unexpected error in main loop: {e}")
                self.api.log_thought_sync(
                    "System", "error", f"Main loop error: {str(e)}"
                )
                time.sleep(10)  # Wait before retrying

        print(f"\n{'='*80}")
        print("MARKET MAKING SYSTEM STOPPED")
        print(f"{'='*80}")
        self.api.log_thought_sync("System", "shutdown", "System shutdown complete")


# Initialize and run the system
if __name__ == "__main__":
    try:
        system = MarketMakingSystem(api_port=5000)
        system.run_continuous(interval_minutes=2)

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nMarket Making System Shutdown Complete")
