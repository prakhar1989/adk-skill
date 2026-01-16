# ADK Tools Reference

## Table of Contents
1. [Function Tools](#function-tools)
2. [Built-in Tools](#built-in-tools)
3. [AgentTool](#agenttool)
4. [MCP Tools](#mcp-tools)
5. [OpenAPI Tools](#openapi-tools)
6. [Tool Context](#tool-context)

---

## Function Tools

The simplest way to give agents capabilities. ADK auto-wraps Python functions.

### Basic Function Tool

```python
from google.adk.agents import Agent

def get_stock_price(ticker: str) -> dict:
    """Get the current stock price for a ticker symbol.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'GOOGL').
    
    Returns:
        dict with 'price' and 'currency' keys.
    """
    # Your implementation
    prices = {"AAPL": 150.0, "GOOGL": 140.0}
    price = prices.get(ticker.upper(), 0.0)
    return {"price": price, "currency": "USD", "ticker": ticker}

agent = Agent(
    name="stock_agent",
    model="gemini-3.0-flash-preview",
    instruction="Help users check stock prices using the get_stock_price tool.",
    tools=[get_stock_price],  # ADK wraps this automatically
)
```

### Function Tool Best Practices

```python
def search_products(
    query: str,
    category: str = "all",
    max_results: int = 10,
    in_stock_only: bool = True,
) -> dict:
    """Search the product catalog.
    
    CRITICAL: The docstring is parsed by ADK to generate the tool schema.
    The LLM reads this to understand when and how to use the tool.
    
    Args:
        query: Search terms to match against product names and descriptions.
        category: Product category filter. Options: 'all', 'electronics', 
                  'clothing', 'home'. Default: 'all'.
        max_results: Maximum number of products to return (1-50). Default: 10.
        in_stock_only: If True, only return products currently in stock.
    
    Returns:
        dict containing:
            - 'status': 'success' or 'error'
            - 'products': list of matching products
            - 'total_count': total matches (may exceed max_results)
    """
    # Implementation
    return {
        "status": "success",
        "products": [...],
        "total_count": 42,
    }
```

### Async Function Tools

```python
import aiohttp

async def fetch_weather(city: str) -> dict:
    """Fetch real-time weather data for a city.
    
    Args:
        city: City name (e.g., 'London', 'New York').
    
    Returns:
        Weather data including temperature and conditions.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.weather.com/{city}") as resp:
            data = await resp.json()
            return {
                "status": "success",
                "temperature": data["temp"],
                "conditions": data["conditions"],
            }

# Async functions work the same way
agent = Agent(
    name="weather_agent",
    model="gemini-3.0-flash-preview",
    tools=[fetch_weather],
)
```

### Explicit FunctionTool

For more control, wrap functions explicitly:

```python
from google.adk.tools import FunctionTool

def calculate(expression: str) -> dict:
    """Evaluate a math expression."""
    try:
        result = eval(expression)  # Use safer eval in production!
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

calc_tool = FunctionTool(
    func=calculate,
    # Override auto-generated schema if needed
)

agent = Agent(
    name="calculator",
    model="gemini-3.0-flash-preview",
    tools=[calc_tool],
)
```

---

## Built-in Tools

Pre-built tools from ADK and Google services.

### Google Search

```python
from google.adk.tools import google_search

agent = Agent(
    name="search_agent",
    model="gemini-3.0-flash-preview",
    instruction="Search the web to answer questions.",
    tools=[google_search],
)
```

### Code Execution

```python
from google.adk.code_executors import BuiltInCodeExecutor

# Enabled via code_executor parameter, not tools
agent = Agent(
    name="coder",
    model="gemini-3.0-flash-preview",
    code_executor=BuiltInCodeExecutor(),
    instruction="Write and execute Python code to solve problems.",
)
```

### Vertex AI Search (RAG)

```python
from google.adk.tools import VertexAiSearchTool

rag_tool = VertexAiSearchTool(
    data_store_id="projects/PROJECT/locations/LOCATION/collections/default_collection/dataStores/DATA_STORE_ID",
)

agent = Agent(
    name="rag_agent",
    model="gemini-3.0-flash-preview",
    tools=[rag_tool],
)
```

---

## AgentTool

Use another agent as a tool, enabling explicit invocation.

### Basic AgentTool

```python
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool

# Specialist agent
researcher = LlmAgent(
    name="researcher",
    model="gemini-3.0-flash-preview",
    description="Researches topics and provides detailed summaries.",
    instruction="Research the given topic thoroughly.",
)

# Wrap as a tool
research_tool = agent_tool.AgentTool(agent=researcher)

# Parent agent uses it as a tool
writer = LlmAgent(
    name="writer",
    model="gemini-3.0-flash-preview",
    instruction="Write articles. Use the researcher tool to gather information first.",
    tools=[research_tool],
)
```

### AgentTool vs Sub-Agents

```python
# Option 1: Sub-agents with LLM-driven delegation
# - Parent LLM decides when to transfer
# - Control passes fully to sub-agent
coordinator = LlmAgent(
    name="coordinator",
    sub_agents=[specialist_a, specialist_b],  # LLM chooses via transfer_to_agent
)

# Option 2: AgentTool for explicit invocation
# - Parent LLM calls agent like a function
# - Gets result back and continues
coordinator = LlmAgent(
    name="coordinator",
    tools=[
        agent_tool.AgentTool(agent=specialist_a),
        agent_tool.AgentTool(agent=specialist_b),
    ],
)
```

### Hierarchical AgentTools

```python
# Low-level tools
web_search = LlmAgent(name="web_search", tools=[google_search])
summarizer = LlmAgent(name="summarizer", instruction="Summarize text concisely.")

# Mid-level agent combining tools
researcher = LlmAgent(
    name="researcher",
    model="gemini-3.0-flash-preview",
    tools=[
        agent_tool.AgentTool(agent=web_search),
        agent_tool.AgentTool(agent=summarizer),
    ],
)

# Top-level agent
report_writer = LlmAgent(
    name="report_writer",
    model="gemini-3.0-flash-preview",
    instruction="Write reports. Use the researcher tool to gather and summarize info.",
    tools=[agent_tool.AgentTool(agent=researcher)],
)
```

---

## MCP Tools

Integrate tools via the Model Context Protocol.

### Remote MCP Server

```python
from google.adk.tools import MCPToolset

# GitHub MCP server (remote)
github_tools = MCPToolset(
    connection_params={
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": f"Bearer {GITHUB_TOKEN}"},
    },
    tool_filter=["get_issues", "create_issue", "get_pull_requests"],  # Limit exposed tools
)

agent = LlmAgent(
    name="github_agent",
    model="gemini-3.0-flash-preview",
    tools=[github_tools],
)
```

### Local MCP Server

```python
from google.adk.tools import MCPToolset
import subprocess

# Start local MCP server process
mcp_process = subprocess.Popen(
    ["python", "-m", "my_mcp_server"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
)

local_tools = MCPToolset(
    connection_params={
        "command": "python",
        "args": ["-m", "my_mcp_server"],
    },
)

agent = LlmAgent(
    name="local_agent",
    model="gemini-3.0-flash-preview",
    tools=[local_tools],
)
```

---

## OpenAPI Tools

Generate tools from OpenAPI/Swagger specs.

```python
from google.adk.tools import OpenAPIToolset

# From URL
api_tools = OpenAPIToolset.from_url(
    "https://api.example.com/openapi.json",
    # Optional: filter to specific operations
    operation_ids=["getUser", "createOrder", "listProducts"],
)

# From file
api_tools = OpenAPIToolset.from_file(
    "path/to/openapi.yaml",
)

agent = LlmAgent(
    name="api_agent",
    model="gemini-3.0-flash-preview",
    tools=[api_tools],
)
```

---

## Tool Context

Access session state and artifacts from within tools.

### Using ToolContext

```python
from google.adk.tools import ToolContext

def save_preference(
    key: str, 
    value: str, 
    tool_context: ToolContext,  # ADK injects this automatically
) -> dict:
    """Save a user preference.
    
    Args:
        key: Preference name.
        value: Preference value.
        tool_context: Injected by ADK - provides access to session.
    
    Returns:
        Confirmation of saved preference.
    """
    # Access session state
    tool_context.state[f"pref:{key}"] = value
    
    # Access artifacts
    # tool_context.save_artifact(name, data, mime_type)
    
    return {"status": "saved", "key": key, "value": value}


def get_user_context(tool_context: ToolContext) -> dict:
    """Get current user context from session.
    
    Args:
        tool_context: Injected by ADK.
    
    Returns:
        Current session context.
    """
    return {
        "user_id": tool_context.state.get("user_id"),
        "preferences": {
            k.replace("pref:", ""): v 
            for k, v in tool_context.state.items() 
            if k.startswith("pref:")
        },
    }

agent = Agent(
    name="preference_agent",
    model="gemini-3.0-flash-preview",
    tools=[save_preference, get_user_context],
)
```

### Artifacts in Tools

```python
from google.adk.tools import ToolContext

def generate_chart(
    data: list,
    chart_type: str,
    tool_context: ToolContext,
) -> dict:
    """Generate a chart and save as artifact.
    
    Args:
        data: Data points for the chart.
        chart_type: Type of chart ('bar', 'line', 'pie').
        tool_context: Injected by ADK.
    
    Returns:
        Confirmation with artifact reference.
    """
    import matplotlib.pyplot as plt
    import io
    
    # Generate chart
    fig, ax = plt.subplots()
    if chart_type == "bar":
        ax.bar(range(len(data)), data)
    elif chart_type == "line":
        ax.plot(data)
    
    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    
    # Save as artifact
    tool_context.save_artifact(
        name="chart.png",
        data=buf.read(),
        mime_type="image/png",
    )
    
    return {"status": "success", "artifact": "chart.png"}
```
