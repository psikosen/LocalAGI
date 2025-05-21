package types

import "github.com/sashabaranov/go-openai"

// OllamaToolParameterProperty defines the structure for individual tool parameter properties
type OllamaToolParameterProperty struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

// OllamaToolParameters defines the structure for tool parameters
type OllamaToolParameters struct {
	Type       string                              `json:"type"` // "object"
	Properties map[string]OllamaToolParameterProperty `json:"properties"`
	Required   []string                            `json:"required,omitempty"`
}

// OllamaToolDefinition defines the structure for a single tool
type OllamaToolDefinition struct {
	Name        string               `json:"name"`
	Description string               `json:"description"`
	Parameters  OllamaToolParameters `json:"parameters"`
}

// OllamaChatRequest defines the structure for a chat request to the Ollama service
type OllamaChatRequest struct {
	ConversationHistory []openai.ChatCompletionMessage `json:"conversation_history"`
	Tools               []OllamaToolDefinition         `json:"tools,omitempty"`
	ModelName           string                         `json:"model_name,omitempty"`
}

// OllamaToolCall defines the structure for a tool call within the Ollama service response
type OllamaToolCall struct {
	ToolName      string                 `json:"tool_name"`
	ToolArguments map[string]interface{} `json:"tool_arguments"`
}

// OllamaChatResponse defines the structure for a chat response from the Ollama service
type OllamaChatResponse struct {
	Type     string          `json:"type"` // "text" or "tool_call"
	Content  string          `json:"content,omitempty"`
	ToolCall *OllamaToolCall `json:"tool_call,omitempty"`
	Error    string          `json:"error,omitempty"`
}
