from fastapi import FastAPI
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter
from core.utils import get_groq_key
import os
import time
import litellm

app = FastAPI()

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
agent_tools = MCPServerAdapter([servers[0]])
agentic_tools = agent_tools.tools

# Create agents
market_researcher = Agent(
    role="Senior Market Research and Analysis Specialist",
    goal="Conduct comprehensive market research and analysis across all supported markets.",
    backstory="You are an experienced market researcher with deep expertise in cryptocurrency markets.",
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

# Create tasks
market_discovery_task = Task(
    description="Perform comprehensive market discovery and analysis using get_supported_markets.",
    expected_output="A comprehensive market discovery report with supported markets and asset analysis.",
    agent=market_researcher,
)

pricing_task = Task(
    description="Set bid and ask price at profitable percentage from mid price based on vault balance.",
    expected_output="A comprehensive pricing strategy with order size, mid price, and bid/ask prices.",
    agent=pricer,
)

# Create crew
market_analysis_crew = Crew(
    agents=[market_researcher, pricer],
    tasks=[market_discovery_task, pricing_task],
    verbose=False,
    process=Process.sequential,
    memory=False,
    llm=llm,
)

@app.get("/")
def read_root():
    return {"message": "Market Making Bot API"}

@app.get("/start-bot")
def start_bot():
    try:
        result = market_analysis_crew.kickoff()
        return {"status": "success", "result": str(result)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)