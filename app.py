from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from core.utils import get_groq_key
import os
import time
import litellm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["MCP_AUTO_CONFIRM"] = "true"


class RetryingLLM(LLM):
    def __init__(self, retries=5, backoff=2, **kwargs):
        super().__init__(**kwargs)
        self.retries = retries
        self.backoff = backoff

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
        for i in range(self.retries):
            try:
                return super().call(*args, **kwargs)
            except litellm.RateLimitError as e:
                wait = self.backoff**i
                print(
                    f"[LLM] Rate limit hit. Waiting {wait}s before retry... ({i+1}/{self.retries})"
                )
                time.sleep(wait)
        raise Exception("LLM call failed after retries due to persistent rate limits.")


# Initialize components once at startup
groq_key = get_groq_key()
os.environ["GROQ_API_KEY"] = groq_key
os.environ["OPENAI_API_BASE"] = "https://api.cerebras.ai/v1/chat/completions"
os.environ["OPENAI_API_KEY"] = os.getenv("CEREBRAS_API_KEY")

llm = RetryingLLM(
    max_tokens=256,
    retries=5,
    backoff=2,
    model="cerebras/llama-3.3-70b",
    temperature=0.7,
)

servers = [
    {
        "url": os.getenv("MCP_SEVER_API", "http://localhost:3000/mcp"),
        "transport": "streamable-http",
    },
]

# Initialize tools globally
# agent_tools = MCPServerAdapter([servers[0]])
# agentic_tools = agent_tools.tools


try:
    agent_tools = MCPServerAdapter([servers[0]])
    agentic_tools = agent_tools.tools
    print(agent_tools, "JIJA")
except Exception as e:
    print(f"Failed to initialize MCPServerAdapter: {e}")
    # Provide fallback tools or initialize with empty tools
    agentic_tools = []

# Create agents
market_researcher = Agent(
    role="Senior Market Research and Analysis Specialist",
    goal="Conduct comprehensive market research and analysis across all supported markets.",
    backstory="You are an experienced market researcher with deep expertise in cryptocurrency markets."
    "After your research you return proper breakdown of the risks rewards"
    "Sort of educating the users on the strategies risk rewards such that any one fully grasp"
    "And understand market concepts in a very simple manner",
    tools=agentic_tools,
    verbose=False,
    llm=llm,
    max_iter=4,
)

pricer = Agent(
    role="Senior Market Price Strategist",
    goal="Ascertain the Amount in Hyper fill vault, decide order size and query proper mid price.",
    backstory="You are a Senior Market Price Strategist with deep experience in crypto markets.",
    tools=agentic_tools,
    verbose=False,
    llm=llm,
    max_iter=4,
)

# ===== EXECUTIVE TRADING AGENT =====
executive_trader = Agent(
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
    # tools=executive_tools,
    tools=agentic_tools,
    verbose=False,
    llm=llm,
    max_iter=6,
)


# Create tasks
market_discovery_task = Task(
    description="Perform comprehensive market discovery and analysis using get_supported_markets.",
    expected_output="A comprehensive market discovery report with supported markets and asset analysis."
    "you can only use the supported markets returned please use the tools",
    agent=market_researcher,
)

pricing_task = Task(
    description="Set ask price at profitable percentage from mid price based on vault balance."
    "the primary asset in vault is the base token so you prolly should place ask",
    expected_output="A comprehensive pricing strategy with order size, mid price, and ask prices.",
    agent=pricer,
)

# ===== EXECUTIVE TRADING TASK =====
executive_trading_task = Task(
    description="""
        Execute the complete market making workflow based on the research and pricing analysis from previous tasks:
        
        1. Review the market research findings and selected trading pair
        2. Validate the pricing strategy and order sizing recommendations
        3. Check current vault balance and ensure sufficient funds
        4. Move required assets from vault to trading wallet if needed
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
    agent=executive_trader,
    context=[
        market_discovery_task,
        pricing_task,
    ],  # Access to previous task outputs
)


# Create crew
market_analysis_crew = Crew(
    agents=[market_researcher, pricer, executive_trader],
    tasks=[market_discovery_task, pricing_task, executive_trading_task],
    verbose=True,
    process=Process.sequential,
    memory=False,
    llm=llm,
)


latest_result = None

@app.get("/")
def read_root():
    return {"message": "Market Making Bot API"}

@app.get("/status")
def status():
    try:
        return {"status": "success", "result": latest_result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/start-bot")
def start_bot():
    try:
        global latest_result
        result = market_analysis_crew.kickoff()
        latest_result = str(result)
        return {"status": "success", "result": latest_result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
