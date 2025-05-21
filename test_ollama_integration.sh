#!/bin/bash

# ##############################################################################
# test_ollama_integration.sh
#
# Draft script for basic integration testing of Ollama functionality
# with the LocalAGI Go application and the Python Ollama service.
#
# This script is a TEMPLATE and will need to be adapted to the specific
# API endpoints and response handling of your LocalAGI Go application.
# ##############################################################################

# --- Prerequisites (Comments) ---
# 1. Ensure Ollama is installed and running.
#    - Example: `ollama serve`
#    - Ensure the model you intend to test with is pulled, e.g., `ollama pull llama3`
# 2. Ensure the Python Ollama service (`ollama_service/main.py`) is running.
#    - For tests without auth, run it without OLLAMA_SERVICE_API_KEY set.
#    - For tests with auth, run it with OLLAMA_SERVICE_API_KEY="testsecret".
# 3. Ensure the LocalAGI Go application is running.

# --- Configuration ---
GO_APP_BASE_URL="http://localhost:8080" # Adjust if your Go app runs elsewhere
PYTHON_SERVICE_BASE_URL="http://localhost:8000" # Adjust if Python service runs elsewhere

# API Keys (placeholders - replace with actual test keys or manage securely)
OPENAI_API_URL_PLACEHOLDER="https://api.openai.com/v1"
OPENAI_KEY_PLACEHOLDER="sk-your_openai_api_key"
PYTHON_SERVICE_API_KEY="testsecret" # Used for Test Case 2

# Agent names
AGENT_OLLAMA_NO_AUTH="ollama_no_auth_agent"
AGENT_OLLAMA_AUTH="ollama_auth_agent"
AGENT_OPENAI_FALLBACK="openai_fallback_agent"

# --- Helper Functions ---
# Basic function to make a PUT request for agent configuration
# Usage: configure_agent <agent_name> <json_payload_string>
configure_agent() {
    local agent_name="$1"
    local payload="$2"
    echo "--- Configuring Agent: $agent_name ---"
    curl -X PUT -H "Content-Type: application/json" \
         -d "$payload" \
         "$GO_APP_BASE_URL/api/agent/$agent_name/config"
    echo -e "\nCheck Go application logs for confirmation of agent configuration."
    sleep 2 # Give a moment for the agent to potentially (re)initialize
}

# Basic function to send a chat message
# Usage: send_chat <agent_name> <message_string>
send_chat() {
    local agent_name="$1"
    local message_content="$2"
    echo "--- Sending chat to Agent: $agent_name ---"
    curl -X POST -H "Content-Type: application/json" \
         -d "{\"message\": \"$message_content\"}" \
         "$GO_APP_BASE_URL/api/chat/$agent_name"
    echo -e "\nCheck Go application output/logs/UI for the agent's response."
    echo "The response might be SSE or a JSON object."
}

# --- Test Case 1: Simple Chat via Ollama (No Auth on Python Service) ---
echo -e "\n\n--- Test Case 1: Simple Chat via Ollama (No Auth on Python Service) ---"
echo "Ensure Python Ollama service is running WITHOUT OLLAMA_SERVICE_API_KEY set."
echo "Example: uvicorn main:app --host 0.0.0.0 --port $PYTHON_SERVICE_PORT_NUMBER"
read -p "Press [Enter] to continue when ready..."

# Configure agent to use Python Ollama service (no API key)
# NOTE: Adjust the model name if "llama3" is not available or not desired.
agent_config_no_auth=$(cat <<EOF
{
    "name": "$AGENT_OLLAMA_NO_AUTH",
    "ollama_service_url": "$PYTHON_SERVICE_BASE_URL",
    "model": "llama3",
    "api_url": "", 
    "api_key": "" 
}
EOF
)
configure_agent "$AGENT_OLLAMA_NO_AUTH" "$agent_config_no_auth"

# Send a chat message to this agent
send_chat "$AGENT_OLLAMA_NO_AUTH" "Hello Ollama, how are you?"

# Verify: Manually check the Go application's output/logs or UI for a response.
# It should be a response from the Ollama model (e.g., llama3).

# --- Test Case 2: Simple Chat via Ollama (With Auth on Python Service) ---
echo -e "\n\n--- Test Case 2: Simple Chat via Ollama (With Auth on Python Service) ---"
echo "Ensure Python Ollama service is running WITH OLLAMA_SERVICE_API_KEY=\"$PYTHON_SERVICE_API_KEY\"."
echo "Example: OLLAMA_SERVICE_API_KEY=\"$PYTHON_SERVICE_API_KEY\" uvicorn main:app --host 0.0.0.0 --port $PYTHON_SERVICE_PORT_NUMBER"
read -p "Press [Enter] to continue when ready..."

# Configure agent to use Python Ollama service WITH API key
agent_config_auth=$(cat <<EOF
{
    "name": "$AGENT_OLLAMA_AUTH",
    "ollama_service_url": "$PYTHON_SERVICE_BASE_URL",
    "ollama_service_api_key": "$PYTHON_SERVICE_API_KEY",
    "model": "llama3",
    "api_url": "", 
    "api_key": ""
}
EOF
)
configure_agent "$AGENT_OLLAMA_AUTH" "$agent_config_auth"

# Send a chat message to this agent (should succeed)
send_chat "$AGENT_OLLAMA_AUTH" "Hello authenticated Ollama, can you hear me?"

# Verify: Manually check for a response from the Ollama model.

# Optional: Test with wrong/missing API key (demonstrative)
echo "--- Optional: Testing Auth Failure (Wrong API Key) ---"
# Reconfigure agent with a wrong API key
agent_config_wrong_key=$(cat <<EOF
{
    "name": "$AGENT_OLLAMA_AUTH",
    "ollama_service_url": "$PYTHON_SERVICE_BASE_URL",
    "ollama_service_api_key": "wrongsecretkey",
    "model": "llama3"
}
EOF
)
configure_agent "$AGENT_OLLAMA_AUTH" "$agent_config_wrong_key"
send_chat "$AGENT_OLLAMA_AUTH" "Hello, will this fail due to wrong key?"
# Verify: Expect an error or no response. Check Python service logs for 403. Check Go app logs.

echo "--- Optional: Testing Auth Failure (Missing API Key) ---"
# Reconfigure agent with missing API key (if Go app allows clearing it)
agent_config_missing_key=$(cat <<EOF
{
    "name": "$AGENT_OLLAMA_AUTH",
    "ollama_service_url": "$PYTHON_SERVICE_BASE_URL",
    "ollama_service_api_key": "", # Assuming empty string clears it
    "model": "llama3"
}
EOF
)
# Note: The Go application's behavior when ollama_service_api_key is set to ""
# after having a value would need to be defined. If it doesn't clear the key,
# this specific test might not reflect a truly "missing" key scenario for an
# already configured agent. A new agent without the key would be a clearer test.
configure_agent "$AGENT_OLLAMA_AUTH" "$agent_config_missing_key"
send_chat "$AGENT_OLLAMA_AUTH" "Hello, will this fail due to missing key?"
# Verify: Expect an error or no response. Check Python service logs for 401. Check Go app logs.


# --- Test Case 3: Fallback to OpenAI (or other configured LLM) ---
echo -e "\n\n--- Test Case 3: Fallback to OpenAI ---"
echo "This test assumes you have standard OpenAI (or other LLM) credentials configured if ollama_service_url is not set."
read -p "Press [Enter] to continue..."

# Configure agent WITHOUT ollama_service_url, relying on standard LLM config
agent_config_openai=$(cat <<EOF
{
    "name": "$AGENT_OPENAI_FALLBACK",
    "ollama_service_url": "", # Explicitly empty or absent
    "api_url": "$OPENAI_API_URL_PLACEHOLDER",
    "api_key": "$OPENAI_KEY_PLACEHOLDER",
    "model": "gpt-3.5-turbo" # Or any other non-Ollama model configured
}
EOF
)
configure_agent "$AGENT_OPENAI_FALLBACK" "$agent_config_openai"

# Send a chat message to this agent
send_chat "$AGENT_OPENAI_FALLBACK" "Hello, are you OpenAI or another LLM (not Ollama)?"

# Verify: Manually check the Go application's output/logs or UI.
# The response should come from the configured OpenAI model (or other LLM), NOT from Ollama.

echo -e "\n\n--- All draft tests outlined. Remember to adapt API endpoints and verification steps. ---"
