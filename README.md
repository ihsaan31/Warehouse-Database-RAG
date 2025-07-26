# Warehouse RAG Chatbot 

## Overview
This is a Retrieval-Augmented Generation (RAG) chatbot built for a portfolio showcase, demonstrating expertise in building intelligent, context-aware chat applications. The chatbot interacts with a dummy warehouse database for the fictional "Central Storage Facility" located at 123 Industrial Park, Springfield, IL. It uses LangChain, Chroma vector store, and the Groq API to provide accurate and structured responses to queries about inventory data, such as item details, stock levels, and warehouse information.

## Features
- **Natural Language Interaction**: Users can query the dummy warehouse database using natural language (e.g., "How many electronics are in stock?" or "Show me low-stock items").
- **Context Retention**: Maintains chat history for coherent, context-aware conversations.
- **RAG Pipeline**: Retrieves relevant data from a Chroma vector store using Ollama embeddings, then generates precise responses.
- **Structured Responses**: Answers are formatted with bullet points, tables, or lists for clarity and readability.
- **Error Handling**: Handles vague or out-of-scope queries gracefully, suggesting relevant database information when applicable.
- **Portfolio Focus**: Showcases skills in AI integration, vector databases, and chatbot development with a clean, professional UI via Chainlit.

## Tech Stack
- **Python Libraries**:
  - `langchain`: For building the RAG pipeline.
  - `langchain_ollama`: For embeddings with `nomic-embed-text:latest`.
  - `langchain_groq`: For the Groq API with `llama3-8b-8192` model.
  - `langchain_chroma`: For vector storage and retrieval.
  - `chainlit`: For the interactive web interface.
  - `python-dotenv`: For environment variable management.
- **Dummy Data**: Simulates a warehouse database with fields like item IDs, names, quantities, unit prices, locations, and more.
- **Chroma Database**: Stores dummy data locally at `/mnt/c/work/portofolio/Chatbot/chromadb` (configurable).
- **Groq API**: Powers the language model for response generation.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up Environment**:
   - Create a `.env` file with your Groq API key:
     ```plaintext
     GROQ_API_KEY=your_groq_api_key
     ```
   - If not provided, the app will prompt for the key at runtime.
4. **Prepare Dummy Data**:
   - Initialize the Chroma database with dummy warehouse data (e.g., item IDs, quantities, categories) at the specified `persist_directory`.
5. **Run the Application**:
   ```bash
   chainlit run app.py
   ```
   - Access the chatbot at `http://localhost:8000`.

## Usage
- Interact via the Chainlit UI by asking questions like:
  - "What is the total value of tools in stock?"
  - "Which items are low in stock?"
  - "Who manages the warehouse?"
- The chatbot retrieves relevant dummy data and responds in a clear, professional format.
- Demonstrates handling of calculations (e.g., total stock value) and structured output (e.g., tables for multiple items).

## Project Structure
- **app.py**: Core application logic, including:
  - Session management with in-memory chat history.
  - RAG pipeline setup with LangChain and Chroma.
  - Chainlit event handlers for chat initialization and message processing.
- **MessageHistory Class**: Manages per-session chat history.
- **Prompt Template**: Guides the model to provide accurate, database-driven responses with a professional tone.

## Configuration
- **Model**: Groq's `llama3-8b-8192` (temperature=0, max_retries=2).
- **Embeddings**: Ollama's `nomic-embed-text:latest` for vectorizing dummy data.
- **Vector Store**: Chroma with MMR search, retrieving up to 5 documents per query.
- **Session ID**: Unique UUID per session for chat history tracking.

## Portfolio Highlights
This project demonstrates:
- **AI Integration**: Combining LLMs (Groq) with RAG for accurate, context-aware responses.
- **Vector Database Usage**: Efficient data retrieval using Chroma and Ollama embeddings.
- **UI/UX**: Clean, interactive interface with Chainlit for a professional user experience.
- **Error Handling**: Robust handling of ambiguous or out-of-scope queries.
- **Scalability**: Modular design suitable for real-world applications with actual data.

## Limitations
- Uses dummy data, simulating a warehouse database for demonstration purposes.
- Responses are limited to the data stored in the Chroma vector store.
- In-memory chat history is not persistent; production use would require a database.
- Assumes a local Chroma database and Ollama server; cloud deployment would need additional setup.

## Future Improvements
- Integrate a persistent database for chat history.
- Add support for real-time data updates in the vector store.
- Enhance the UI with visualizations (e.g., stock level charts).
- Expand dummy data to showcase more complex queries (e.g., supplier analytics).

