# ADK Runtime Reference

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Sessions](#sessions)
3. [Session Services](#session-services)
4. [Runner](#runner)
5. [Events](#events)
6. [Programmatic Execution](#programmatic-execution)
7. [Building APIs and UIs](#building-apis-and-uis)

---

## Core Concepts

When you define an agent, you describe *what* it can do. The runtime infrastructure handles *how* it executes and maintains state.

| Component | Responsibility |
|-----------|---------------|
| **Agent** | *What* to do — instructions, tools, sub-agents |
| **Session** | *Memory* — conversation history, state, artifacts |
| **SessionService** | *Storage* — where sessions persist (memory, database, cloud) |
| **Runner** | *Execution* — orchestrates the agent loop, yields events |

```
┌─────────────────────────────────────────────────────┐
│                     Your Code                        │
│  ┌───────────┐    ┌────────┐    ┌────────────────┐  │
│  │   Agent   │───▶│ Runner │───▶│ SessionService │  │
│  └───────────┘    └────────┘    └────────────────┘  │
│                        │                 │          │
│                        ▼                 ▼          │
│                   [Events]          [Sessions]      │
└─────────────────────────────────────────────────────┘
```

---

## Sessions

A **Session** represents a single conversation thread with persistent memory.

### What Sessions Store

```python
session.id              # Unique session identifier
session.user_id         # User this session belongs to
session.state           # Dict[str, Any] - persists across turns
session.events          # Conversation history (user + agent messages)
session.artifacts       # Binary data (images, files)
```

### Session Identity

Sessions are identified by three values:
- **app_name**: Your application identifier
- **user_id**: Which user owns this conversation
- **session_id**: Which specific conversation thread

```python
# One user can have multiple sessions (like separate chat threads)
# app_name="customer_support", user_id="alice", session_id="chat_001"
# app_name="customer_support", user_id="alice", session_id="chat_002"
```

### State Management

State is how agents share data across turns and between each other:

```python
# Agent writes to state via output_key
agent_a = LlmAgent(
    name="researcher",
    output_key="research_results",  # Saves response to state["research_results"]
)

# Agent reads from state via instruction templates
agent_b = LlmAgent(
    name="writer",
    instruction="Write an article based on {research_results}",  # Reads state
)

# You can also pre-populate state when creating a session
session = await session_service.create_session(
    app_name="my_app",
    user_id="user_123",
    session_id="session_456",
    state={
        "user_name": "Alice",
        "preferences": {"tone": "formal"},
        "context": "Previous conversation summary...",
    },
)
```

---

## Session Services

SessionService determines where sessions are stored.

### InMemorySessionService

For development and testing. Data lost on restart.

```python
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()
```

### DatabaseSessionService

For production. Persists to SQL database.

```python
from google.adk.sessions import DatabaseSessionService

session_service = DatabaseSessionService(
    connection_string="postgresql://user:pass@host:5432/dbname",
)
```

### VertexAISessionService

For Google Cloud deployments with managed storage.

```python
from google.adk.sessions import VertexAISessionService

session_service = VertexAISessionService(
    project="my-gcp-project",
    location="us-central1",
)
```

### Session Service Operations

```python
# Create a new session
session = await session_service.create_session(
    app_name="my_app",
    user_id="user_123",
    session_id="session_456",
    state={"initial": "data"},  # Optional initial state
)

# Get existing session
session = await session_service.get_session(
    app_name="my_app",
    user_id="user_123",
    session_id="session_456",
)

# List user's sessions
sessions = await session_service.list_sessions(
    app_name="my_app",
    user_id="user_123",
)

# Delete a session
await session_service.delete_session(
    app_name="my_app",
    user_id="user_123",
    session_id="session_456",
)
```

---

## Runner

The **Runner** executes agents and manages the interaction loop.

### Basic Setup

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

agent = Agent(
    name="assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
    tools=[my_tool],
)

session_service = InMemorySessionService()

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
)
```

### Running the Agent

```python
from google.genai import types

# Create user message
message = types.Content(
    role="user",
    parts=[types.Part(text="What's the weather in Tokyo?")],
)

# Run and iterate through events
async for event in runner.run_async(
    user_id="user_123",
    session_id="session_456",
    new_message=message,
):
    # Process events (see Events section below)
    if event.is_final_response():
        print(event.content.parts[0].text)
```

### Runner Configuration

```python
from google.adk.runners import Runner, RunConfig

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
    # Optional: artifact service for binary data
    artifact_service=my_artifact_service,
)

# Run with configuration
async for event in runner.run_async(
    user_id="user_123",
    session_id="session_456",
    new_message=message,
    run_config=RunConfig(
        max_llm_calls=10,  # Limit LLM calls per run
    ),
):
    ...
```

---

## Events

The Runner yields **Events** as the agent executes. Events represent everything that happens: LLM responses, tool calls, tool results, state changes, etc.

### Event Structure

```python
event.id            # Unique event identifier
event.author        # Who created this event (agent name or "user")
event.content       # The Content (text, tool calls, etc.)
event.actions       # EventActions (state changes, escalate flag, etc.)
event.timestamp     # When this occurred
```

### Event Types

```python
async for event in runner.run_async(...):
    # Check what kind of event this is
    
    if event.content and event.content.parts:
        for part in event.content.parts:
            # Text response from agent
            if part.text:
                print(f"Agent said: {part.text}")
            
            # Agent wants to call a tool
            if part.function_call:
                print(f"Calling tool: {part.function_call.name}")
            
            # Tool returned a result
            if part.function_response:
                print(f"Tool result: {part.function_response.response}")
    
    # Final response - conversation turn complete
    if event.is_final_response():
        final_text = event.content.parts[0].text
        break
```

### Streaming vs Final

```python
async for event in runner.run_async(...):
    # Partial/streaming content (as tokens arrive)
    if event.partial:
        print(event.content.parts[0].text, end="", flush=True)
    
    # Complete response
    if event.is_final_response():
        print()  # Newline
        final_response = event.content.parts[0].text
```

---

## Programmatic Execution

Complete examples for running agents in your own code.

### Simple Chat Function

```python
import asyncio
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Setup (do once)
agent = Agent(
    name="assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="my_app", session_service=session_service)

async def chat(
    message: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
) -> str:
    """Send a message and get the response."""
    
    # Ensure session exists
    session = await session_service.get_session(
        app_name="my_app", user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name="my_app", user_id=user_id, session_id=session_id
        )
    
    # Create message content
    content = types.Content(
        role="user",
        parts=[types.Part(text=message)],
    )
    
    # Run agent and get final response
    final_response = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
    
    return final_response

# Usage
async def main():
    # Single conversation
    response = await chat("What is Python?")
    print(response)
    
    # Continues same session (has memory of previous turn)
    response = await chat("What are its main uses?")
    print(response)
    
    # Different session (fresh conversation)
    response = await chat("Hello!", session_id="other_session")
    print(response)

asyncio.run(main())
```

### Chat with Streaming Output

```python
async def chat_streaming(
    message: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
) -> str:
    """Send a message and stream the response."""
    
    session = await session_service.get_session(
        app_name="my_app", user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name="my_app", user_id=user_id, session_id=session_id
        )
    
    content = types.Content(role="user", parts=[types.Part(text=message)])
    
    full_response = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    # Print as tokens arrive
                    print(part.text, end="", flush=True)
                    full_response += part.text
        
        if event.is_final_response():
            print()  # Newline at end
            break
    
    return full_response
```

### Chat with Context/State

```python
async def chat_with_context(
    message: str,
    context: dict = None,
    user_id: str = "default_user",
    session_id: str = None,
) -> tuple[str, dict]:
    """Chat with initial context, returns response and final state."""
    
    # Generate unique session ID if not provided
    if session_id is None:
        import uuid
        session_id = str(uuid.uuid4())
    
    # Create session with initial state
    await session_service.create_session(
        app_name="my_app",
        user_id=user_id,
        session_id=session_id,
        state=context or {},
    )
    
    content = types.Content(role="user", parts=[types.Part(text=message)])
    
    final_response = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
    
    # Get final state after agent ran
    session = await session_service.get_session(
        app_name="my_app", user_id=user_id, session_id=session_id
    )
    final_state = dict(session.state) if session else {}
    
    return final_response, final_state

# Usage
response, state = await chat_with_context(
    message="Summarize my preferences",
    context={
        "user_name": "Alice",
        "preferences": {"theme": "dark", "language": "en"},
    },
)
print(f"Response: {response}")
print(f"State after: {state}")
```

---

## Building APIs and UIs

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

app = FastAPI()

# Initialize agent and runner
agent = Agent(
    name="api_assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful API assistant.",
)
session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="api_app", session_service=session_service)

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Ensure session exists
    session = await session_service.get_session(
        app_name="api_app",
        user_id=request.user_id,
        session_id=request.session_id,
    )
    if not session:
        await session_service.create_session(
            app_name="api_app",
            user_id=request.user_id,
            session_id=request.session_id,
        )
    
    # Run agent
    content = types.Content(
        role="user",
        parts=[types.Part(text=request.message)],
    )
    
    final_response = ""
    async for event in runner.run_async(
        user_id=request.user_id,
        session_id=request.session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
    
    return ChatResponse(
        response=final_response,
        session_id=request.session_id,
    )

@app.post("/sessions/{user_id}")
async def create_session(user_id: str, initial_state: dict = None):
    import uuid
    session_id = str(uuid.uuid4())
    
    await session_service.create_session(
        app_name="api_app",
        user_id=user_id,
        session_id=session_id,
        state=initial_state or {},
    )
    
    return {"session_id": session_id}

@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str):
    sessions = await session_service.list_sessions(
        app_name="api_app",
        user_id=user_id,
    )
    return {"sessions": [s.id for s in sessions]}
```

### Streaming with Server-Sent Events (SSE)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from google.genai import types

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    async def event_generator():
        session = await session_service.get_session(
            app_name="api_app",
            user_id=request.user_id,
            session_id=request.session_id,
        )
        if not session:
            await session_service.create_session(
                app_name="api_app",
                user_id=request.user_id,
                session_id=request.session_id,
            )
        
        content = types.Content(
            role="user",
            parts=[types.Part(text=request.message)],
        )
        
        async for event in runner.run_async(
            user_id=request.user_id,
            session_id=request.session_id,
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        # SSE format
                        yield f"data: {part.text}\n\n"
            
            if event.is_final_response():
                yield "data: [DONE]\n\n"
                break
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
```

### Using ADK's Built-in API Server

ADK provides a ready-made API server for simpler deployments:

```bash
# Start the API server
adk api_server my_agent --port 8080

# Endpoints available:
# POST /run          - Run agent (blocking)
# POST /run_sse      - Run agent (streaming SSE)
# GET  /docs         - Interactive API docs
```

```python
# Client code to call the API server
import requests

response = requests.post(
    "http://localhost:8080/run",
    json={
        "user_id": "user_123",
        "session_id": "session_456",
        "message": "Hello, agent!",
    },
)
print(response.json())
```
