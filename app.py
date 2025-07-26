
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
from langchain_ollama.chat_models import ChatOllama
import os 
import chainlit as cl
import dotenv
dotenv.load_dotenv()
from langchain_groq import ChatGroq
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
import uuid

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")


class MessageHistory:
    def __init__(self, messages=None):
        self.messages = messages if messages is not None else []

    def add_messages(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    async def aget_messages(self):
        return self.messages

    # In-memory message history store
session_message_histories = {}

    # Get or create history per session
def get_session_history(session_id: str):
    if session_id not in session_message_histories:
        session_message_histories[session_id] = MessageHistory()
    return session_message_histories[session_id]

    # Wrap chain with message history
def create_chain_with_chat_history(final_chain: Runnable):
    final_chain = RunnableWithMessageHistory(
            final_chain,
            get_session_history,
            input_messages_key="question",
            output_messages_key="answer",
            history_messages_key="chat_history",
            config_configs=[
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    description="Unique identifier for the session.",
                    default="",
                    is_shared=True,
                ),
            ],
        )
    return final_chain


@cl.on_chat_start
async def on_chat_start():

    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    model = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    vector_store = Chroma(
    collection_name="warehouse_collection",  # Name of the collection
    embedding_function=embeddings,
    persist_directory="/mnt/c/work/portofolio/Chatbot/chromadb",  # Where to save data locally, remove if not necessary
)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    prompt = PromptTemplate.from_template(
        """
You are a highly efficient and knowledgeable Retrieval-Augmented Generation (RAG) chatbot designed to assist users with queries about the warehouse database for the "Central Storage Facility" located at 123 Industrial Park, Springfield, IL. Your primary goal is to provide accurate, concise, and relevant responses based on the provided warehouse database. You have access to detailed inventory data, including item IDs, names, categories, quantities, unit prices, locations within the warehouse, last updated dates, statuses, suppliers, minimum stock levels, and warehouse information such as total items, total value, last inventory check, manager, and contact details.

## Instructions
1. **Understand the Query**: Analyze the user's query to identify the specific information requested, such as item details, stock status, warehouse information, or calculations (e.g., total value of specific items).
2. **Retrieve Relevant Data**: Use the provided warehouse database to extract accurate and relevant information. Ensure you reference the correct fields (e.g., item_id, quantity, status, etc.) and avoid fabricating data.
3. **Generate a Response**: Provide a clear, concise, and accurate answer tailored to the user's query. Use natural language to present the information in an easily understandable format. If calculations are required (e.g., total value of items in a category), perform them accurately.
4. **Handle Ambiguities**: If the query is vague or ambiguous, ask for clarification while providing a partial answer based on the most likely interpretation, referencing the database.
5. **Out-of-Scope Queries**: If the query cannot be answered using the database (e.g., future predictions, external supplier details), politely inform the user that the information is not available in the provided data and suggest relevant database-related information if applicable.
6. **Formatting**: Present responses in a professional and structured manner. Use bullet points, tables, or numbered lists when appropriate to improve readability, especially for queries involving multiple items or complex data.


## Guidelines
- Always reference the database accurately and avoid assumptions beyond the provided data.
- For queries requiring calculations, show the breakdown (e.g., quantity × unit price) for transparency.
- If multiple items match a query (e.g., items in the same category), list all relevant items unless the user specifies otherwise.
- Use the warehouse information (e.g., manager, contact) when relevant to the query.
- Maintain a professional and helpful tone, ensuring the user feels supported in their warehouse management tasks.

Question: {question}
chat history: {chat_history}
stock : {docs}

"""
    )
    runnable = (
        RunnablePassthrough.assign(
            docs=lambda x: retriever.invoke(x["question"]),
        )
        | prompt
        | model
        | StrOutputParser()
    )

    chain_history_quality = create_chain_with_chat_history(final_chain=runnable)
    cl.user_session.set("chain_history_quality", chain_history_quality)


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    # Retrieve the chain from the user session instead of recreating it
    chain_history_quality = cl.user_session.get("chain_history_quality")

    session_history = get_session_history(session_id)
    # Use message.content to get the actual message text
    session_history.add_messages("user", message.content)

    msg = cl.Message(content="")
    full_response = ""
    async for chunk in chain_history_quality.astream(
        {"question": message.content},
        config={"configurable": {"session_id": session_id}}
    ):
        await msg.stream_token(chunk)
        full_response += chunk

    session_history.add_messages("assistant", full_response)
    await msg.send()