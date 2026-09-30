# AI Agent & Chatbot Experiments

A collection of Python scripts exploring AI agents and chatbots using OpenAI-compatible APIs with Groq and Ollama backends.

## Features

- **File Management Agent** (`agent.py`) — An autonomous AI agent that reads, writes, and organizes files in a `notes/` folder using a think-act-observe loop with tool calling.
- **Bengali Chatbot** (`chatbot.py`) — A conversational chatbot called "বন্ধু" (Friend) that responds in natural Bengali using Groq API.
- **Local Chatbot** (`local_chatbot.py`) — Same Bengali chatbot running locally via Ollama with Llama 3.2 (3B).
- **Tool Calling Demo** (`tools.py`) — Demonstrates function calling with calculator and current time tools.

## Tech Stack

- Python
- Groq API (Qwen 3.8-27B, GPT-OSS-120B)
- Ollama (Llama 3.2:3B)
- OpenAI Python SDK

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/ai-agent.git
   cd ai-agent
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv ai_venv
   source ai_venv/bin/activate
   pip install openai python-decouple
   ```

3. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

4. For local chatbot, install and start [Ollama](https://ollama.com):
   ```bash
   ollama pull llama3.2:3b
   ollama serve
   ```

## Usage

```bash
# Run the file management agent
python agent.py

# Run the Bengali chatbot (Groq)
python chatbot.py

# Run the local Bengali chatbot (Ollama)
python local_chatbot.py

# Run tool calling demo
python tools.py
```

## License

MIT
