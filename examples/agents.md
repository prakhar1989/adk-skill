# ADK Agents Reference

## Table of Contents
1. [LlmAgent](#llmagent)
2. [Workflow Agents](#workflow-agents)
3. [Custom Agents](#custom-agents)
4. [Agent Configuration](#agent-configuration)

---

## LlmAgent

The core agent type powered by an LLM for reasoning and tool use.

### Basic Definition

```python
from google.adk.agents import LlmAgent, Agent  # Agent is an alias for LlmAgent

agent = LlmAgent(
    name="my_agent",                    # Required: unique identifier
    model="gemini-3.0-flash-preview",           # Required: model string
    description="What this agent does", # For delegation decisions
    instruction="System prompt here",   # Behavior guidance
    tools=[tool1, tool2],               # Functions or Tool instances
    sub_agents=[child1, child2],        # For multi-agent hierarchies
    output_key="result",                # Auto-save response to state
)
```

### Full Parameter Reference

```python
from google.adk.agents import LlmAgent
from google.genai import types
from pydantic import BaseModel

class OutputSchema(BaseModel):
    answer: str
    confidence: float

agent = LlmAgent(
    # Identity
    name="advanced_agent",
    model="gemini-3.0-flash-preview",
    description="Handles complex queries with structured output.",

    # Behavior
    instruction="""You are a helpful assistant.
    Guidelines:
    - Be concise
    - Use tools when needed
    - Output in the required format
    """,

    # Tools and sub-agents
    tools=[my_tool_function],
    sub_agents=[specialist_agent],

    # State management
    output_key="agent_response",  # Saves final text to state

    # Schema constraints
    input_schema=InputSchema,     # Expect JSON input matching schema
    output_schema=OutputSchema,   # Force JSON output matching schema

    # Context control
    include_contents='default',   # 'default' or 'none' (stateless)

    # LLM configuration
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=1000,
        top_p=0.9,
    ),
)
```

### Instruction Templates

Instructions support state variable substitution:

```python
agent = LlmAgent(
    name="processor",
    instruction="""Process the data in {input_data}.
    User preferences: {user_prefs}
    Previous result: {previous_result?}  # ? = optional, no error if missing
    """,
)
```

### Model Options

```python
# Google AI Studio (requires GOOGLE_API_KEY)
model="gemini-3.0-flash-preview"
model="gemini-3.0-pro"
model="gemini-3.0-flash-preview"

# Vertex AI (requires GOOGLE_CLOUD_PROJECT)
model="gemini-3.0-flash-preview"  # Same string, different auth

# Via LiteLLM (for other providers)
from google.adk.models import LiteLlm
model=LiteLlm(model="gpt-4o")
model=LiteLlm(model="claude-3-5-sonnet-20241022")
```

---

## Workflow Agents

Deterministic orchestration agents that control sub-agent execution flow.

### SequentialAgent

Executes sub-agents in order. Ideal for pipelines.

```python
from google.adk.agents import SequentialAgent, LlmAgent

# Each agent can read state written by previous agents
fetch = LlmAgent(name="fetch", instruction="Fetch data.", output_key="raw_data")
process = LlmAgent(name="process", instruction="Process {raw_data}.", output_key="processed")
report = LlmAgent(name="report", instruction="Create report from {processed}.")

pipeline = SequentialAgent(
    name="data_pipeline",
    sub_agents=[fetch, process, report],
)
```

### ParallelAgent

Executes sub-agents concurrently. Use distinct output_keys to avoid race conditions.

```python
from google.adk.agents import ParallelAgent, SequentialAgent, LlmAgent

# These run simultaneously
news_fetcher = LlmAgent(name="news", output_key="news")
weather_fetcher = LlmAgent(name="weather", output_key="weather")
stocks_fetcher = LlmAgent(name="stocks", output_key="stocks")

parallel_fetch = ParallelAgent(
    name="parallel_fetch",
    sub_agents=[news_fetcher, weather_fetcher, stocks_fetcher],
)

# Combine with sequential for fan-out/gather pattern
aggregator = LlmAgent(
    name="aggregator",
    instruction="Combine {news}, {weather}, and {stocks} into a briefing.",
)

root_agent = SequentialAgent(
    name="briefing_pipeline",
    sub_agents=[parallel_fetch, aggregator],
)
```

### LoopAgent

Repeats sub-agents until max_iterations or escalate=True.

```python
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
from google.adk.events import Event, EventActions
from typing import AsyncGenerator

class TerminationChecker(BaseAgent):
    """Custom agent that checks state and escalates to exit loop."""

    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        quality = ctx.session.state.get("quality_score", 0)
        should_stop = quality >= 8  # Exit when quality is good enough
        yield Event(
            author=self.name,
            actions=EventActions(escalate=should_stop)
        )

improver = LlmAgent(
    name="improver",
    instruction="Improve the draft in {draft}. Rate quality 1-10.",
    output_key="draft",
)

quality_rater = LlmAgent(
    name="rater",
    instruction="Rate the {draft} quality from 1-10. Output just the number.",
    output_key="quality_score",
)

refinement_loop = LoopAgent(
    name="refinement",
    max_iterations=5,  # Safety limit
    sub_agents=[improver, quality_rater, TerminationChecker(name="gate")],
)
```

---

## Custom Agents

Extend BaseAgent for non-LLM logic or custom orchestration.

### Basic Custom Agent

```python
from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from typing import AsyncGenerator

class DataTransformer(BaseAgent):
    """Custom agent that transforms data without using an LLM."""

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Read from state
        raw_data = ctx.session.state.get("raw_data", {})

        # Process (no LLM needed)
        transformed = self._transform(raw_data)

        # Write to state
        ctx.session.state["transformed_data"] = transformed

        # Yield response event
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=f"Transformed {len(raw_data)} records.")]
            ),
        )

    def _transform(self, data: dict) -> dict:
        # Your transformation logic
        return {k: v.upper() if isinstance(v, str) else v for k, v in data.items()}

# Use in a pipeline
transformer = DataTransformer(name="transformer", description="Transforms raw data.")
```

### Custom Agent with Escalation

```python
class ConditionalRouter(BaseAgent):
    """Routes based on state, can escalate to parent."""

    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        task_type = ctx.session.state.get("task_type", "unknown")

        if task_type == "urgent":
            # Escalate to parent agent
            yield Event(
                author=self.name,
                actions=EventActions(escalate=True),
                content=types.Content(
                    parts=[types.Part(text="Escalating urgent task to supervisor.")]
                ),
            )
        else:
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"Handling {task_type} task normally.")]
                ),
            )
```

---

## Agent Configuration

### Global Instructions

Apply instructions to all agents in a hierarchy:

```python
root_agent = LlmAgent(
    name="root",
    model="gemini-3.0-flash-preview",
    global_instruction="Always be polite and professional.",  # Applies to all sub-agents
    instruction="You coordinate specialized agents.",
    sub_agents=[agent_a, agent_b],
)
```

### Transfer Control

Control how agents can delegate:

```python
specialist = LlmAgent(
    name="specialist",
    model="gemini-3.0-flash-preview",
    disallow_transfer_to_parent=True,   # Can't escalate up
    disallow_transfer_to_peers=True,    # Can't transfer to siblings
)
```

### Planner Integration

Enable multi-step reasoning:

```python
from google.adk.planners import BuiltInPlanner, PlanReActPlanner
from google.genai.types import ThinkingConfig

# Use Gemini's built-in thinking
agent = LlmAgent(
    name="thinker",
    model="gemini-2.5-flash",
    planner=BuiltInPlanner(
        thinking_config=ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1024,
        )
    ),
    tools=[...],
)

# Or use Plan-ReAct pattern
agent = LlmAgent(
    name="planner",
    model="gemini-3.0-flash-preview",
    planner=PlanReActPlanner(),
    tools=[...],
)
```

### Code Execution

Allow agents to write and run code:

```python
from google.adk.code_executors import BuiltInCodeExecutor

calculator = LlmAgent(
    name="calculator",
    model="gemini-3.0-flash-preview",
    code_executor=BuiltInCodeExecutor(),
    instruction="Write Python code to solve math problems.",
)
```
