# Ollama Interaction Service

This Python service acts as a bridge between the main Go application (LocalAGI) and an Ollama instance. It allows LocalAGI agents to use Ollama models for chat completions and tool usage, leveraging the `instructor` library for structured output (tool calls).

## Prerequisites

1.  **Ollama Installed and Running:** Ensure you have [Ollama](httpsa://ollama.ai/) installed and running.
2.  **Ollama Models:** Pull the desired models. For example:
    ```bash
    ollama pull llama3
    ```
    The service defaults to using "llama3" if no model is specified in the request from the Go application.

## Setup and Running

### Dependencies

The service uses FastAPI, Uvicorn, Ollama-Python, and Instructor. Dependencies are listed in `requirements.txt`.

### Option 1: Using Docker (Recommended)

1.  **Build the Docker image:**
    From the root of the LocalAGI repository, navigate to this directory (`ollama_service`) and run:
    ```bash
    docker build -t localagi-ollama-service .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 --network host localagi-ollama-service
    ```
    *   `-p 8000:8000`: Maps port 8000 of the container to port 8000 on your host.
    *   `--network host`: (Optional, but often easiest for development) Allows the service inside Docker to connect to Ollama running on your host machine (e.g., `http://localhost:11434`). If Ollama is running in its own Docker container, you might need to adjust network settings (e.g., use a shared Docker network).

### Option 2: Running Locally with Uvicorn

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the FastAPI application with Uvicorn:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The service will be available at `http://localhost:8000`.

## Configuration

-   `OLLAMA_HOST_URL`: Specifies the URL of the Ollama instance the service should connect to.
    -   Defaults to `http://localhost:11434`.
    -   You can override this when running the Docker container, e.g.:
        ```bash
        docker run -e OLLAMA_HOST_URL="http://my_ollama_server:11434" -p 8000:8000 localagi-ollama-service
        ```
    -   If running with Uvicorn directly, set the environment variable in your shell:
        ```bash
        export OLLAMA_HOST_URL="http://my_ollama_server:11434"
        uvicorn main:app --host 0.0.0.0 --port 8000
        ```
-   `OLLAMA_SERVICE_API_KEY`: (Optional) An API key to secure the service.
    -   Defaults to empty (no authentication).
    -   If set, clients must send this key in the `X-API-Key` header.
    -   Example Docker override:
        ```bash
        docker run -e OLLAMA_SERVICE_API_KEY="your_secret_key" -e OLLAMA_HOST_URL="http://my_ollama_server:11434" -p 8000:8000 localagi-ollama-service
        ```

## API

The service exposes one main endpoint:

*   `POST /chat`
    *   **Purpose:** Receives chat history and available tools from the Go application, interacts with Ollama, and returns either a text response or a tool call instruction.
    *   **Request Body (JSON):**
        ```json
        {
            "conversation_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "tool", "content": "Tool output here..."} 
            ],
            "tools": [ // Optional: list of tool definitions
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city and state"}
                        },
                        "required": ["location"]
                    }
                }
            ],
            "model_name": "llama3" // Optional: defaults to "llama3"
        }
        ```
    *   **Response Body (JSON):**
        *   For text response:
            ```json
            {
                "type": "text",
                "content": "This is the LLM's response."
            }
            ```
        *   For tool call:
            ```json
            {
                "type": "tool_call",
                "tool_name": "get_weather",
                "tool_arguments": {"location": "Boston, MA"}
            }
            ```

## How it Works with LocalAGI

When an agent in the LocalAGI Go application is configured with the URL of this running service, its LLM calls (for chat and determining tool usage) are directed here. This service then:
1. Receives the request from the Go agent.
2. Uses the `ollama-python` library to communicate with your Ollama instance.
3. Employs the `instructor` library to enable structured outputs, allowing Ollama to effectively decide when to use tools and what parameters to use.
4. Returns the response (text or tool call) to the Go agent.
