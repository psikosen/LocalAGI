import os
import ollama
import instructor
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, create_model
from typing import List, Union, Optional, Dict, Any, AsyncGenerator
import json
import logging # Added for logging

# --- Pydantic Models for Request and Response ---

class Message(BaseModel):
    role: str
    content: str

class ToolParameterProperty(BaseModel):
    type: str
    description: Optional[str] = None

class ToolParameters(BaseModel):
    type: str
    properties: Dict[str, ToolParameterProperty]
    required: List[str] = []

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: ToolParameters

class ChatRequest(BaseModel):
    conversation_history: List[Message]
    tools: Optional[List[ToolDefinition]] = None
    model_name: str = "llama3"

class TextResponseData(BaseModel):
    type: str = "text"
    content: str

class ToolCallResponseData(BaseModel):
    type: str = "tool_call"
    tool_name: str
    tool_arguments: Dict[str, Any]

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Logging Setup ---
# Uvicorn typically manages basicConfig, but we can set a logger instance.
# If running standalone, you might need: logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Example: If you want to ensure a specific level even if Uvicorn defaults differ
# import sys
# handler = logging.StreamHandler(sys.stdout)
# handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
# logger.addHandler(handler)
# logger.setLevel(logging.DEBUG) # Or INFO, as needed

# --- Configuration ---
OLLAMA_HOST_URL = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
OLLAMA_SERVICE_API_KEY = os.environ.get("OLLAMA_SERVICE_API_KEY")

# --- Ollama Client Initialization ---
ollama_client = ollama.Client(host=OLLAMA_HOST_URL)
instructor_client = instructor.patch(ollama_client)

# --- Helper Pydantic Model for simple text responses when using instructor ---
class DefaultTextResponse(BaseModel):
    """A simple model for text responses from the LLM."""
    content: str

# Removed response_model from decorator for StreamingResponse
@app.post("/chat")
async def chat_endpoint(request: ChatRequest, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Handles chat requests, routing them to Ollama and processing responses,
    which may include text or tool calls, streamed as Server-Sent Events.
    Includes API Key authentication if OLLAMA_SERVICE_API_KEY is set.
    """
    logger.info(f"Received /chat request. Model: {request.model_name}, Tools provided: {len(request.tools) if request.tools else 0}")

    if OLLAMA_SERVICE_API_KEY: # Check if API key is configured for the service
        if not x_api_key:
            logger.warn("X-API-Key header missing for protected endpoint.")
            raise HTTPException(status_code=401, detail="X-API-Key header missing")
        if x_api_key != OLLAMA_SERVICE_API_KEY:
            logger.warn(f"Invalid API Key received: {x_api_key[:5]}...") # Log a redacted version
            raise HTTPException(status_code=403, detail="Invalid API Key")

    pydantic_tool_models = []
    tool_names_map = {} # To map Pydantic model names back to original tool names

    if request.tools:
        for tool_def in request.tools:
            fields = {}
            for param_name, param_props in tool_def.parameters.properties.items():
                field_type = str
                if param_props.type == "string": field_type = str
                elif param_props.type == "integer": field_type = int
                elif param_props.type == "number": field_type = float
                elif param_props.type == "boolean": field_type = bool
                fields[param_name] = (field_type, ...)
            
            model_name = tool_def.name.replace("-", "_").replace(" ", "_")
            DynamicToolModel = create_model(model_name, **fields, __doc__=tool_def.description)
            pydantic_tool_models.append(DynamicToolModel)
            tool_names_map[model_name] = tool_def.name

    response_options = list(pydantic_tool_models)
    response_options.append(DefaultTextResponse)
    final_response_model = Union[tuple(response_options)] if response_options else DefaultTextResponse

    ollama_messages = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]

    async def stream_generator() -> AsyncGenerator[str, None]:
        try:
            logger.info(f"Calling Ollama model {request.model_name} via instructor.")
            response_stream = instructor_client.chat(
                model=request.model_name,
                messages=ollama_messages,
                response_model=final_response_model,
                stream=True
            )

            last_sent_text_content = "" 

            async for chunk in response_stream:
                if isinstance(chunk, DefaultTextResponse):
                    if chunk.content and chunk.content != last_sent_text_content:
                        new_content_part = chunk.content[len(last_sent_text_content):]
                        if new_content_part:
                            logger.debug(f"Streaming text chunk: {new_content_part}") # Using new_content_part for brevity
                            yield "data: " + json.dumps({"type": "text_chunk", "content": new_content_part}) + "\n\n"
                            last_sent_text_content = chunk.content
                
                elif hasattr(chunk, '__class__') and chunk.__class__.__name__ in tool_names_map:
                    tool_model_name = chunk.__class__.__name__
                    original_tool_name = tool_names_map.get(tool_model_name, tool_model_name)
                    tool_arguments = chunk.model_dump()
                    logger.info(f"Identified tool call: {original_tool_name}, arguments: {tool_arguments}")
                    yield "data: " + json.dumps({
                        "type": "tool_call",
                        "tool_name": original_tool_name,
                        "tool_arguments": tool_arguments
                    }) + "\n\n"
                    return 

        except ollama.ResponseError as e:
            status_code = e.status_code if hasattr(e, 'status_code') else 500
            error_payload = {"type": "error", "detail": f"Ollama API error: {e.error}", "status_code": status_code}
            logger.error(f"Streaming error to client: {error_payload}, Status Code: {status_code}", exc_info=True)
            yield "data: " + json.dumps(error_payload) + "\n\n"
        except Exception as e:
            error_payload = {"type": "error", "detail": f"An internal server error occurred: {str(e)}"}
            logger.exception("An unexpected error occurred during stream:") # exc_info=True is default with .exception
            yield "data: " + json.dumps(error_payload) + "\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
