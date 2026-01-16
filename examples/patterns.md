# ADK Multi-Agent Patterns

## Table of Contents
1. [Coordinator/Dispatcher](#coordinatordispatcher)
2. [Sequential Pipeline](#sequential-pipeline)
3. [Parallel Fan-Out/Gather](#parallel-fan-outgather)
4. [Hierarchical Decomposition](#hierarchical-decomposition)
5. [Generator-Critic](#generator-critic)
6. [Iterative Refinement](#iterative-refinement)
7. [Human-in-the-Loop](#human-in-the-loop)
8. [Running Agents Programmatically](#running-agents-programmatically)

---

## Coordinator/Dispatcher

Route requests to specialized agents based on content.

```python
from google.adk.agents import LlmAgent

# Specialist agents with clear descriptions
billing_agent = LlmAgent(
    name="billing",
    model="gemini-2.0-flash",
    description="Handles billing, payments, invoices, and subscription questions.",
    instruction="Help users with billing inquiries. Be precise about amounts and dates.",
)

technical_agent = LlmAgent(
    name="technical",
    model="gemini-2.0-flash",
    description="Handles technical support, bugs, errors, and how-to questions.",
    instruction="Help users solve technical problems. Ask for error messages and logs.",
)

sales_agent = LlmAgent(
    name="sales",
    model="gemini-2.0-flash",
    description="Handles pricing, demos, enterprise plans, and purchasing questions.",
    instruction="Help users understand our products and pricing. Be helpful but not pushy.",
)

# Coordinator routes based on descriptions
root_agent = LlmAgent(
    name="help_desk",
    model="gemini-2.0-flash",
    instruction="""You are the help desk coordinator.
    
    Analyze each request and route to the appropriate specialist:
    - billing: payment issues, invoices, subscriptions
    - technical: bugs, errors, how-to questions
    - sales: pricing, demos, purchasing
    
    If unclear, ask a clarifying question before routing.
    """,
    sub_agents=[billing_agent, technical_agent, sales_agent],
)
```

---

## Sequential Pipeline

Process data through ordered stages.

```python
from google.adk.agents import SequentialAgent, LlmAgent

# Stage 1: Extract
extractor = LlmAgent(
    name="extractor",
    model="gemini-2.0-flash",
    instruction="""Extract key information from the input document.
    Output as JSON with keys: title, author, date, main_points, entities.
    """,
    output_key="extracted_data",
)

# Stage 2: Analyze
analyzer = LlmAgent(
    name="analyzer",
    model="gemini-2.0-flash",
    instruction="""Analyze the extracted data in {extracted_data}.
    Identify:
    - Sentiment (positive/negative/neutral)
    - Key themes
    - Notable patterns
    Output as JSON.
    """,
    output_key="analysis",
)

# Stage 3: Summarize
summarizer = LlmAgent(
    name="summarizer",
    model="gemini-2.0-flash",
    instruction="""Create a concise summary based on:
    - Extracted data: {extracted_data}
    - Analysis: {analysis}
    
    Format: 2-3 paragraph executive summary.
    """,
    output_key="summary",
)

# Stage 4: Format
formatter = LlmAgent(
    name="formatter",
    model="gemini-2.0-flash",
    instruction="""Format the summary in {summary} as a professional report.
    Include sections: Overview, Key Findings, Recommendations.
    """,
)

root_agent = SequentialAgent(
    name="document_processor",
    sub_agents=[extractor, analyzer, summarizer, formatter],
)
```

---

## Parallel Fan-Out/Gather

Execute independent tasks simultaneously, then combine results.

```python
from google.adk.agents import ParallelAgent, SequentialAgent, LlmAgent

# Parallel research agents
market_researcher = LlmAgent(
    name="market_research",
    model="gemini-2.0-flash",
    instruction="Research market trends and competitive landscape for {topic}.",
    output_key="market_data",
)

technical_researcher = LlmAgent(
    name="tech_research",
    model="gemini-2.0-flash",
    instruction="Research technical aspects and feasibility for {topic}.",
    output_key="tech_data",
)

financial_researcher = LlmAgent(
    name="financial_research",
    model="gemini-2.0-flash",
    instruction="Research financial implications and ROI for {topic}.",
    output_key="financial_data",
)

# Fan-out: run all three in parallel
research_phase = ParallelAgent(
    name="parallel_research",
    sub_agents=[market_researcher, technical_researcher, financial_researcher],
)

# Gather: synthesize results
synthesizer = LlmAgent(
    name="synthesizer",
    model="gemini-2.0-flash",
    instruction="""Synthesize the research findings:
    - Market: {market_data}
    - Technical: {tech_data}
    - Financial: {financial_data}
    
    Create a comprehensive recommendation report.
    """,
)

root_agent = SequentialAgent(
    name="research_pipeline",
    sub_agents=[research_phase, synthesizer],
)
```

---

## Hierarchical Decomposition

Break complex tasks into subtasks handled by specialized agents.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool

# Leaf-level specialists
code_writer = LlmAgent(
    name="code_writer",
    model="gemini-2.0-flash",
    description="Writes code implementations.",
    instruction="Write clean, documented code for the given requirements.",
)

code_reviewer = LlmAgent(
    name="code_reviewer",
    model="gemini-2.0-flash",
    description="Reviews code for bugs and improvements.",
    instruction="Review code for bugs, security issues, and style. Suggest improvements.",
)

test_writer = LlmAgent(
    name="test_writer",
    model="gemini-2.0-flash",
    description="Writes unit tests.",
    instruction="Write comprehensive unit tests with good coverage.",
)

# Mid-level: Development lead uses specialists as tools
dev_lead = LlmAgent(
    name="dev_lead",
    model="gemini-2.0-flash",
    description="Manages code development tasks.",
    instruction="""Coordinate code development:
    1. Use code_writer to implement features
    2. Use code_reviewer to review the code
    3. Use test_writer to create tests
    Ensure quality before marking complete.
    """,
    tools=[
        agent_tool.AgentTool(agent=code_writer),
        agent_tool.AgentTool(agent=code_reviewer),
        agent_tool.AgentTool(agent=test_writer),
    ],
)

# Similarly for documentation
doc_writer = LlmAgent(name="doc_writer", description="Writes documentation.")
doc_reviewer = LlmAgent(name="doc_reviewer", description="Reviews documentation.")

doc_lead = LlmAgent(
    name="doc_lead",
    model="gemini-2.0-flash",
    description="Manages documentation tasks.",
    tools=[
        agent_tool.AgentTool(agent=doc_writer),
        agent_tool.AgentTool(agent=doc_reviewer),
    ],
)

# Top-level: Project manager
root_agent = LlmAgent(
    name="project_manager",
    model="gemini-2.0-flash",
    instruction="""Manage the project:
    - Use dev_lead for code-related tasks
    - Use doc_lead for documentation tasks
    Coordinate to deliver complete features.
    """,
    tools=[
        agent_tool.AgentTool(agent=dev_lead),
        agent_tool.AgentTool(agent=doc_lead),
    ],
)
```

---

## Generator-Critic

Generate content, then review and improve it.

```python
from google.adk.agents import SequentialAgent, LlmAgent

generator = LlmAgent(
    name="generator",
    model="gemini-2.0-flash",
    instruction="""Generate a response to the user's request.
    Be creative and thorough.
    """,
    output_key="draft",
)

critic = LlmAgent(
    name="critic",
    model="gemini-2.0-flash",
    instruction="""Review the draft in {draft}.
    
    Evaluate:
    - Accuracy: Are facts correct?
    - Completeness: Does it fully address the request?
    - Clarity: Is it easy to understand?
    - Tone: Is it appropriate?
    
    Output a JSON with scores (1-10) and specific issues found.
    """,
    output_key="critique",
)

refiner = LlmAgent(
    name="refiner",
    model="gemini-2.0-flash",
    instruction="""Improve the draft based on the critique.
    
    Original draft: {draft}
    Critique: {critique}
    
    Address all issues while preserving good elements.
    """,
)

root_agent = SequentialAgent(
    name="review_pipeline",
    sub_agents=[generator, critic, refiner],
)
```

---

## Iterative Refinement

Loop until quality threshold is met.

```python
from google.adk.agents import LoopAgent, SequentialAgent, LlmAgent, BaseAgent
from google.adk.events import Event, EventActions
from typing import AsyncGenerator

class QualityGate(BaseAgent):
    """Exit loop when quality score >= threshold."""
    
    def __init__(self, name: str, threshold: int = 8):
        super().__init__(name=name, description="Quality gate checker")
        self.threshold = threshold
    
    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        score = ctx.session.state.get("quality_score", 0)
        try:
            score = int(score)
        except:
            score = 0
        
        should_exit = score >= self.threshold
        yield Event(
            author=self.name,
            actions=EventActions(escalate=should_exit),
        )

# Initial draft generator (runs once before loop)
initial_drafter = LlmAgent(
    name="initial_draft",
    model="gemini-2.0-flash",
    instruction="Create an initial draft for the user's request.",
    output_key="current_draft",
)

# Improvement agent (runs each iteration)
improver = LlmAgent(
    name="improver",
    model="gemini-2.0-flash",
    instruction="""Improve the draft in {current_draft}.
    Previous feedback: {feedback}
    Make it better while keeping what works.
    """,
    output_key="current_draft",  # Overwrites previous draft
)

# Quality evaluator
evaluator = LlmAgent(
    name="evaluator",
    model="gemini-2.0-flash",
    instruction="""Evaluate {current_draft} on a scale of 1-10.
    Provide specific feedback for improvement.
    Output format:
    Score: [number]
    Feedback: [specific improvements needed]
    """,
    output_key="feedback",
)

# Score extractor (parses the score from feedback)
score_extractor = LlmAgent(
    name="score_extractor",
    model="gemini-2.0-flash",
    instruction="""Extract just the numeric score from: {feedback}
    Output only the number, nothing else.
    """,
    output_key="quality_score",
)

# Refinement loop
refinement_loop = LoopAgent(
    name="refinement",
    max_iterations=5,
    sub_agents=[improver, evaluator, score_extractor, QualityGate(name="gate", threshold=8)],
)

root_agent = SequentialAgent(
    name="iterative_writer",
    sub_agents=[initial_drafter, refinement_loop],
)
```

---

## Human-in-the-Loop

Pause for human approval or input.

```python
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.tools import FunctionTool

# This would integrate with your external approval system
async def request_human_approval(
    action: str,
    details: str,
    urgency: str = "normal",
) -> dict:
    """Request human approval for an action.
    
    Args:
        action: What action needs approval.
        details: Detailed description for the reviewer.
        urgency: 'low', 'normal', or 'high'.
    
    Returns:
        dict with 'approved' (bool) and 'comments' (str).
    """
    # In production, this would:
    # 1. Send notification to approval system
    # 2. Wait for human response (webhook, polling, etc.)
    # 3. Return the decision
    
    # Placeholder - always approves
    return {
        "approved": True,
        "comments": "Looks good, proceed.",
        "approver": "human@example.com",
    }

approval_tool = FunctionTool(func=request_human_approval)

# Agent that prepares actions for approval
preparer = LlmAgent(
    name="preparer",
    model="gemini-2.0-flash",
    instruction="""Analyze the request and prepare an action plan.
    Output:
    - Proposed action
    - Risk assessment
    - Required approvals
    """,
    output_key="action_plan",
)

# Agent that requests approval
approver = LlmAgent(
    name="approval_requester",
    model="gemini-2.0-flash",
    instruction="""Based on {action_plan}, request human approval.
    Use the request_human_approval tool with appropriate details.
    """,
    tools=[approval_tool],
    output_key="approval_result",
)

# Agent that executes approved actions
executor = LlmAgent(
    name="executor",
    model="gemini-2.0-flash",
    instruction="""Check approval in {approval_result}.
    If approved: Execute the action from {action_plan}.
    If rejected: Explain why the action was not taken.
    """,
)

root_agent = SequentialAgent(
    name="approved_workflow",
    sub_agents=[preparer, approver, executor],
)
```

---

## Running Agents Programmatically

Execute agents from Python code (not just CLI).

### Basic Execution

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Define agent
agent = LlmAgent(
    name="assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
)

# Setup runner
session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
)

async def chat(user_message: str, user_id: str = "user1", session_id: str = "session1"):
    """Send a message and get the response."""
    
    # Ensure session exists
    session = await session_service.get_session(
        app_name="my_app",
        user_id=user_id,
        session_id=session_id,
    )
    if not session:
        await session_service.create_session(
            app_name="my_app",
            user_id=user_id,
            session_id=session_id,
        )
    
    # Create message
    content = types.Content(
        role="user",
        parts=[types.Part(text=user_message)],
    )
    
    # Run and collect response
    final_response = None
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content:
            final_response = event.content.parts[0].text
    
    return final_response

# Usage
async def main():
    response = await chat("What is the capital of France?")
    print(response)
    
    # Continue conversation in same session
    response = await chat("What about Germany?")
    print(response)

asyncio.run(main())
```

### With Initial State

```python
async def chat_with_context(
    user_message: str,
    initial_state: dict = None,
    user_id: str = "user1",
    session_id: str = "session1",
):
    """Chat with pre-populated state."""
    
    # Create session with initial state
    session = await session_service.create_session(
        app_name="my_app",
        user_id=user_id,
        session_id=session_id,
        state=initial_state or {},
    )
    
    content = types.Content(
        role="user",
        parts=[types.Part(text=user_message)],
    )
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content:
            return event.content.parts[0].text

# Usage with context
response = await chat_with_context(
    "Summarize my preferences",
    initial_state={
        "user_name": "Alice",
        "preferences": {"theme": "dark", "language": "en"},
        "history": ["visited homepage", "viewed products"],
    },
)
```

### Streaming Responses

```python
async def chat_streaming(user_message: str):
    """Stream response tokens as they arrive."""
    
    content = types.Content(
        role="user",
        parts=[types.Part(text=user_message)],
    )
    
    async for event in runner.run_async(
        user_id="user1",
        session_id="session1",
        new_message=content,
    ):
        # Print partial responses as they stream
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text, end="", flush=True)
        
        if event.is_final_response():
            print()  # Newline at end
            break
```
