# Warehouse RAG Chatbot

## Overview
This application is a Retrieval-Augmented Generation (RAG) chatbot designed to assist users with queries about the warehouse database for the "Central Storage Facility" located at 123 Industrial Park, Springfield, IL. The chatbot leverages a combination of LangChain, Chroma vector store, and the Groq API to provide accurate and context-aware responses based on inventory data, including item details, stock status, and warehouse information.

## Features
- **Natural Language Queries**: Users can ask questions about inventory, such as item quantities, locations, or total stock value, in natural language.
- **Context-Aware Responses**: Maintains chat history to provide coherent and contextually relevant answers.
- **Retrieval-Augmented Generation**: Uses a Chroma vector store with Ollama embeddings to retrieve relevant data from the warehouse database before generating responses.
- **Structured Output**: Responses are formatted for clarity, using bullet points, tables, or lists when appropriate.
- **Error Handling**: Gracefully handles ambiguous queries or out-of-scope questions by informing users and suggesting relevant database-related information.

## Prerequisites
To run this application, ensure you have the following installed:
- Python 3.8+
- Required Python packages (listed in `requirements.txt`):
  - `langchain`
  - `langchain_ollama`
  - `langchain_groq`
  - `langchain_chroma`
  - `chainlit`
  - `python-dotenv`
  - `getpass`
- A valid Groq API key (set as `GROQ_API_KEY` in a `.env` file or entered during runtime).
- Access to an Ollama server for embeddings (model: `nomic-embed-text:latest`).
- A Chroma database set up at `/mnt/c/work/portofolio/Chatbot/chromadb` (or modify the `persist_directory` as needed).

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the environment:
   - Create a `.env` file in the project root and add your Groq API key:
     ```plaintext
     GROQ_API_KEY=your_groq_api_key
     ```
   - Alternatively, the application will prompt for the API key during runtime if not set.
4. Ensure the Chroma database is initialized with warehouse data at the specified `persist_directory`.

## Usage
1. Start the Chainlit server:
   ```bash
   chainlit run app.py
   ```
2. Open your browser and navigate to `http://localhost:8000` to interact with the chatbot.
3. Ask questions about the warehouse, such as:
   - "What is the total value of electronics in stock?"
   - "Where are the items with low stock located?"
   - "Who is the warehouse manager?"

## Code Structure
- **app.py**: Main application file containing the chatbot logic, including:
  - Session management with in-memory message history.
  - Integration with LangChain for RAG pipeline.
  - Chroma vector store setup for data retrieval.
  - Prompt template for structured responses.
- **MessageHistory class**: Manages chat history for each user session.
- **Chainlit event handlers**:
  - `@cl.on_chat_start`: Initializes the session, model, and RAG chain.
  - `@cl.on_message`: Processes user queries and streams responses.

## Configuration
- **Model**: Uses `llama3-8b-8192` from Groq with configurable parameters (temperature=0, max_retries=2).
- **Embeddings**: Uses `nomic-embed-text:latest` from Ollama for vector store embeddings.
- **Vector Store**: Chroma with MMR search type, retrieving up to 5 relevant documents per query.
- **Session Management**: Generates a unique `session_id` per user session to maintain chat history.

## Notes
- The Chroma database must be pre-populated with warehouse data (e.g., item IDs, quantities, unit prices, etc.).
- Modify the `persist_directory` in the code if your Chroma database is stored elsewhere.
- Ensure the Ollama server is running locally or on an accessible host for embeddings.
- For production, consider using a persistent storage solution instead of in-memory `session_message_histories`.

## Limitations
- The chatbot can only respond based on data in the Chroma vector store.
- Out-of-scope queries (e.g., future predictions or external supplier details) are handled by informing the user of the limitation.
- The application assumes a local Chroma database; cloud-based setups may require additional configuration.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for bugs, feature requests, or improvements.

## License
This project is licensed under the MIT License.