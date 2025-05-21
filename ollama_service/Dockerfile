# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY main.py .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variables
ENV OLLAMA_HOST_URL="http://localhost:11434" 
ENV OLLAMA_HOST="0.0.0.0" # For Uvicorn server binding
ENV OLLAMA_SERVICE_API_KEY="" # Optional API key for the service, auth disabled by default

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
