from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from core.utils import get_groq_key
import os
import time
import litellm


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


# Setup LLM
groq_key = get_groq_key()
print(groq_key, "GROQ_KEY")
os.environ["GROQ_API_KEY"] = get_groq_key()
os.environ["OPENAI_API_BASE"] = "https://api.cerebras.ai/v1/chat/completions"
os.environ["OPENAI_API_KEY"] = os.getenv("CEREBRAS_API_KEY")
max_response_tokens = 256
llm = RetryingLLM(
    # model="groq/llama-3.3-70b-versatile",
    # temperature=0.7,
    max_tokens=max_response_tokens,
    retries=5,  # try 5 times
    backoff=2,  # exponential: 1s, 2s, 4s, 8s...
    model="cerebras/llama-3.3-70b",  # or "llama-3.1-8b"
    temperature=0.7,
    # max_tokens=512,
)

# MCP Server configuration
servers = [
    {
        "url": os.getenv("MCP_SEVER_API", "http://localhost:3000/mcp"),
        "transport": "streamable-http",
    },  # Market Analyzer Server
]


# Initialize MCP adapters
agent_tools = None


try:
    # Initialize the adapters
    agent_tools = MCPServerAdapter([servers[0]])
    # executive_adapter = MCPServerAdapter([servers[2]])

    # Get tools from the adapters
    agentic_tools = agent_tools.tools

    print(f"Market Analyzer tools available: {[tool.name for tool in agentic_tools]}")
    # print(f"Pricer tools available: {[tool.name for tool in pricer_tools]}")
    # print(f"Executive tools available: {[tool.name for tool in executive_tools]}")

    # ===== MARKET RESEARCH AND ANALYSIS AGENT =====
    market_researcher = Agent(
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
        tools=agentic_tools,
        verbose=False,
        llm=llm,
        max_iter=4,
    )

    # ===== PRICING STRATEGIST AGENT =====
    pricer = Agent(
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
        tools=agentic_tools,
        verbose=False,
        llm=llm,
        max_iter=6,
    )

    # ===== COMPREHENSIVE MARKET DISCOVERY TASK =====
    market_discovery_task = Task(
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
        agent=market_researcher,
    )

    # ===== PRICING STRATEGY TASK =====
    pricing_task = Task(
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

    # ===== CREW SETUP =====
    market_analysis_crew = Crew(
        agents=[
            market_researcher,
            pricer,
            #   executive_trader
        ],
        tasks=[
            market_discovery_task,
            pricing_task,
            #   executive_trading_task
        ],
        verbose=False,
        process=Process.sequential,
        memory=False,
        llm=llm,
    )

    # ===== EXECUTE COMPLETE MARKET MAKING WORKFLOW =====
    print("\n" + "=" * 80)
    print("STARTING COMPREHENSIVE MARKET MAKING WORKFLOW")
    print("=" * 80 + "\n")

    result = market_analysis_crew.kickoff()

    print("\n" + "=" * 80)
    print("MARKET MAKING WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"\nFinal Execution Report:\n{result}")

except Exception as e:
    print(f"Error in market making system: {e}")
    print("Ensure all MCP servers are running:")
    print("- Market Analyzer: localhost:2000")
    print("- Pricer: localhost:3000")
    print("- Executive: localhost:4000")
    import traceback

    traceback.print_exc()
finally:
    # Clean up adapters if needed
    if agent_tools:
        pass

print("\n" + "=" * 50)
print("Market Making System Ready for Automation")
print("=" * 50)
