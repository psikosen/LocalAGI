import ollama
import instructor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from typing import List, Union, Optional, Dict, Any

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

# --- Ollama Client Initialization ---
ollama_client = ollama.Client()
# Patch the Ollama client with instructor
instructor_client = instructor.patch(ollama_client)

# --- Helper Pydantic Model for simple text responses when using instructor ---
class DefaultTextResponse(BaseModel):
    """A simple model for text responses from the LLM."""
    content: str

@app.post("/chat", response_model=Union[TextResponseData, ToolCallResponseData])
async def chat_endpoint(request: ChatRequest):
    """
    Handles chat requests, routing them to Ollama and processing responses,
    which may include text or tool calls.
    """
    try:
        pydantic_tool_models = []
        tool_names_map = {} # To map Pydantic model names back to original tool names if needed

        if request.tools:
            for tool_def in request.tools:
                fields = {}
                for param_name, param_props in tool_def.parameters.properties.items():
                    # Pydantic needs Python types, not JSON schema type strings
                    field_type = str  # Default to string
                    if param_props.type == "string":
                        field_type = str
                    elif param_props.type == "integer":
                        field_type = int
                    elif param_props.type == "number":
                        field_type = float
                    elif param_props.type == "boolean":
                        field_type = bool
                    # Add more type mappings if necessary (e.g., array, object)

                    # Field description can be passed to Field if needed, or used in model docstring
                    fields[param_name] = (field_type, ...) # '...' makes it a required field

                # Create a Pydantic model for this tool
                # Ensure the model name is a valid Python identifier
                model_name = tool_def.name.replace("-", "_").replace(" ", "_")
                DynamicToolModel = create_model(
                    model_name,
                    **fields,
                    __doc__=tool_def.description
                )
                pydantic_tool_models.append(DynamicToolModel)
                tool_names_map[model_name] = tool_def.name


        # The response model for instructor needs to allow for any of the defined tools OR a simple text response.
        # We add DefaultTextResponse to the list of possible models.
        response_options = list(pydantic_tool_models) # Make a mutable copy
        response_options.append(DefaultTextResponse)

        # Ensure there's at least one response option, even if it's just DefaultTextResponse
        if not response_options: # Should not happen due to append above, but as a safeguard
            final_response_model = DefaultTextResponse
        else:
            final_response_model = Union[tuple(response_options)]


        # Prepare messages for Ollama
        ollama_messages = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]

        # Call Ollama using the instructor-patched client
        # The `tools` parameter for ollama.chat needs to be in the format expected by Ollama's function calling
        # which `instructor` handles when `response_model` includes tool models.
        # `instructor` will internally convert Pydantic models to a format Ollama can understand for function calling.
        ollama_response = instructor_client.chat(
            model=request.model_name,
            messages=ollama_messages,
            response_model=final_response_model,
            # `tools` argument for instructor.patch'd ollama.chat should be a list of Pydantic models
            # if you want to explicitly define them. Otherwise, it infers from response_model.
            # For dynamic tools, relying on response_model containing the tool Pydantic models is typical.
        )

        # Process the response
        if isinstance(ollama_response, DefaultTextResponse):
            return TextResponseData(content=ollama_response.content)
        elif hasattr(ollama_response, '__class__') and ollama_response.__class__.__name__ in tool_names_map:
            # It's one of the dynamic tool models
            tool_model_name = ollama_response.__class__.__name__
            original_tool_name = tool_names_map.get(tool_model_name, tool_model_name) # Fallback to model name
            return ToolCallResponseData(
                tool_name=original_tool_name,
                tool_arguments=ollama_response.model_dump()
            )
        else:
            # This case should ideally not be reached if DefaultTextResponse covers all non-tool scenarios.
            # However, as a fallback, treat as text if possible.
            if hasattr(ollama_response, 'content'):
                 return TextResponseData(content=str(ollama_response.content))
            elif isinstance(ollama_response, BaseModel): # A Pydantic model but not one we mapped
                 # This indicates a potential issue with tool definition or response model logic
                 # For now, return its content if possible or dump the model.
                 try:
                     return TextResponseData(content=ollama_response.content)
                 except AttributeError:
                    # If it doesn't have .content, it might be an unhandled tool or complex object.
                    # Log this for debugging. For now, returning its dump might be too verbose or expose internal details.
                    # Raise an error or return a generic error message.
                    print(f"Warning: Unhandled response type: {type(ollama_response)}. Dump: {ollama_response.model_dump_json()}")
                    raise HTTPException(status_code=500, detail=f"Unhandled response type from LLM: {type(ollama_response).__name__}")
            else:
                # If the response is not a Pydantic model and not a DefaultTextResponse,
                # it's an unexpected situation.
                print(f"Error: Unexpected response type from instructor: {type(ollama_response)}")
                raise HTTPException(status_code=500, detail="Unexpected response structure from LLM.")

    except ollama.ResponseError as e:
        print(f"Ollama API Error: {e.status_code} - {e.error}")
        raise HTTPException(status_code=e.status_code, detail=f"Ollama API error: {e.error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Log the full traceback for server-side debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
