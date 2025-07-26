# from typing import Optional, Dict, List, Any
# import os
# import uuid
# import pickle
# import chainlit as cl
# from langchain.prompts import PromptTemplate
# from langchain.schema import StrOutputParser
# from langchain.schema.runnable import Runnable, RunnablePassthrough
# from langchain.schema.runnable.config import RunnableConfig
# from langchain_groq import ChatGroq
# from langchain_ollama.embeddings import OllamaEmbeddings
# from langchain_chroma import Chroma
# import dotenv

# # Load environment variables
# dotenv.load_dotenv()

# # --- Authentication ---
# @cl.password_auth_callback
# def auth_callback(username: str, password: str):
#     # Simple credential check for demo purposes
#     if (username, password) == ("admin", "admin"):
#         return cl.User(
#             identifier="admin", metadata={"role": "admin", "provider": "credentials"}
#         )
#     else:
#         return None

# # --- Persistent Data Layer using Pickle ---
# class PickleDataLayer(cl.data.BaseDataLayer):
#     def __init__(self, file_path: str = "chainlit_data.pkl"):
#         self.file_path = file_path
#         if os.path.exists(self.file_path):
#             with open(self.file_path, "rb") as f:
#                 self.store = pickle.load(f)
#         else:
#             # Initialize store with a schema-like structure
#             self.store = {
#                 "users": {},       # key = identifier, value = {id, identifier, createdAt, metadata}
#                 "threads": {},     # key = thread_id, value = thread dict
#                 "feedbacks": {},   # key = feedback_id, value = feedback dict
#                 "elements": {},    # key = element_id, value = element dict
#                 "steps": {},       # key = step_id, value = step dict
#                 "sessions": {},
#             }
#             self._persist()

#     def build_debug_url(self, id: str) -> str:
#         return ""

#     def _persist(self):
#         """Save the store to the pickle file."""
#         with open(self.file_path, "wb") as f:
#             pickle.dump(self.store, f)

#     # --- User Methods ---
#     async def get_user(self, identifier: str) -> Optional[Dict[str, Any]]:
#         return self.store["users"].get(identifier)

#     async def create_user(self, user: cl.User) -> Dict[str, Any]:
#         data = {
#             "id": str(uuid.uuid4()),
#             "identifier": user.identifier,
#             "createdAt": str(uuid.uuid1()),
#             "metadata": user.metadata,
#         }
#         self.store["users"][user.identifier] = data
#         self._persist()
#         return data

#     # --- Thread Methods ---
#     async def list_threads(self, pagination: Any, filters: Any) -> Any:
#         from chainlit.data.sql_alchemy import PaginatedResponse
#         threads = list(self.store["threads"].values())
#         total = len(threads)
#         page_info = {
#             "page": getattr(pagination, 'page', 1),
#             "pageSize": getattr(pagination, 'page_size', total),
#             "totalPages": 1,
#             "hasNextPage": False,
#             "startCursor": None,
#             "endCursor": None
#         }
#         return PaginatedResponse(data=threads, total=total, pageInfo=page_info)

#     async def delete_thread(self, thread_id: str) -> bool:
#         """Delete a thread from the store by its ID."""
#         if thread_id in self.store["threads"]:
#             del self.store["threads"][thread_id]
#             self._persist()
#             return True
#         return False

#     async def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
#         """Retrieve a thread from the store by its ID."""
#         return self.store["threads"].get(thread_id)

#     async def update_thread(self, thread: Dict[str, Any]) -> None:
#         """Update or add a thread in the store."""
#         thread_id = thread.get("id")
#         if not thread_id:
#             raise ValueError("Thread dictionary must contain an 'id' key")
#         self.store["threads"][thread_id] = thread
#         self._persist()

#     async def update_thread(self, thread: Dict[str, Any]) -> None:
#             """Update or add a thread in the store."""
#             thread_id = thread.get("id")
#             if not thread_id:
#                 raise ValueError("Thread dictionary must contain an 'id' key")
#             self.store["threads"][thread_id] = thread
#             self._persist()

#     async def delete_thread(self, thread_id: str) -> bool:
#         """Delete a thread from the store by its ID."""
#         if thread_id in self.store["threads"]:
#             del self.store["threads"][thread_id]
#             self._persist()
#             return True
#         return False

#     async def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
#         """Retrieve a thread from the store by its ID."""
#         return self.store["threads"].get(thread_id)

#     # --- Feedback Methods ---
#     async def upsert_feedback(self, feedback: Dict[str, Any]) -> str:
#         fid = feedback.get("id", str(uuid.uuid4()))
#         self.store["feedbacks"][fid] = feedback
#         self._persist()
#         return fid

#     async def delete_feedback(self, feedback_id: str) -> bool:
#         return self.store["feedbacks"].pop(feedback_id, None) is not None

#     # --- Element Methods ---
#     async def create_element(self, element: Dict[str, Any]) -> None:
#         eid = element.get("id", str(uuid.uuid4()))
#         self.store["elements"][eid] = element
#         self._persist()

#     async def get_element(self, thread_id: str, element_id: str) -> Optional[Dict[str, Any]]:
#         el = self.store["elements"].get(element_id)
#         return el if el and el.get("threadId") == thread_id else None

#     async def delete_element(self, element_id: str) -> None:
#         self.store["elements"].pop(element_id, None)
#         self._persist()

#     # --- Step Methods ---
#     async def create_step(self, step: Dict[str, Any]) -> None:
#         sid = step.get("id", str(uuid.uuid4()))
#         self.store["steps"][sid] = step
#         self._persist()

#     async def update_step(self, step: Dict[str, Any]) -> None:
#         self.store["steps"].update({step.get("id"): step})
#         self._persist()

#     async def delete_step(self, step_id: str) -> None:
#         self.store["steps"].pop(step_id, None)
#         self._persist()

#     async def get_thread_author(self, thread_id: str) -> Optional[str]:
#         thr = self.store["threads"].get(thread_id)
#         return thr.get("userId") if thr else None

#     async def delete_user_session(self, id: str) -> bool:
#         return self.store["sessions"].pop(id, None) is not None

# # Register the data layer
# @cl.data_layer
# def get_data_layer():
#     return PickleDataLayer()

# # --- Chainlit App Setup ---
# if "GROQ_API_KEY" not in os.environ:
#     os.environ["GROQ_API_KEY"] = cl.getpass("Enter your Groq API key:")

# # In-memory session histories
# session_histories: Dict[str, List[Dict[str, str]]] = {}

# class MessageHistory:
#     def __init__(self, messages: List[Dict[str, str]] = None):
#         self.messages = messages or []
#     def add_messages(self, role: str, content: str):
#         self.messages.append({"role": role, "content": content})
#     async def aget_messages(self):
#         return self.messages

# def get_session_history(session_id: str) -> MessageHistory:
#     if session_id not in session_histories:
#         session_histories[session_id] = []
#     return MessageHistory(session_histories[session_id])

# def create_chain_with_chat_history(final_chain: Runnable) -> Runnable:
#     def get_messages(input: Dict):
#         session_id = input.get("session_id", "")
#         history = get_session_history(session_id)
#         return history.messages

#     return (
#         RunnablePassthrough.assign(chat_history=get_messages)
#         | final_chain
#     ).with_config(
#         RunnableConfig(id="session_id", description="Session identifier", default="", is_shared=True)
#     )

# @cl.on_chat_start
# def on_chat_start():
#     session_id = str(uuid.uuid4())
#     cl.user_session.set("session_id", session_id)

#     # Setup RAG components
#     model = ChatGroq(model="llama3-8b-8192", temperature=0)
#     embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
#     vector_store = Chroma(collection_name="warehouse_collection",
#                          embedding_function=embeddings,
#                          persist_directory="./chromadb")
#     retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

#     prompt = PromptTemplate.from_template(
#         """
# # Warehouse DB RAG Chatbot Prompt

# Question: {question}
# chat history: {chat_history}
# stock : {docs}
# """
#     )
#     runnable = (
#         RunnablePassthrough.assign(docs=lambda x: retriever.invoke(x["question"]))
#         | prompt
#         | model
#         | StrOutputParser()
#     )
#     chain = create_chain_with_chat_history(runnable)
#     cl.user_session.set("chain", chain)

# @cl.on_message
# async def on_message(message: cl.Message):
#     session_id = cl.user_session.get("session_id")
#     chain = cl.user_session.get("chain")
#     history = get_session_history(session_id)
#     history.add_messages("user", message.content)

#     msg = cl.Message(content="")
#     response = ""
#     for chunk in chain.stream({"question": message.content}, config={"session_id": session_id}):
#         await msg.stream_token(chunk)
#         response += chunk
#     history.add_messages("assistant", response)
#     await msg.send()


# from typing import Optional, Dict, List, Any
# import os
# import uuid
# import pickle
# import chainlit as cl
# from langchain.prompts import PromptTemplate
# from langchain.schema import StrOutputParser
# from langchain.schema.runnable import Runnable, RunnablePassthrough
# from langchain.schema.runnable.config import RunnableConfig
# from langchain_groq import ChatGroq
# from langchain_ollama.embeddings import OllamaEmbeddings
# from langchain_chroma import Chroma
# from datetime import datetime
# import dotenv

# # Load environment variables
# dotenv.load_dotenv()

# # --- Authentication ---
# @cl.password_auth_callback
# def auth_callback(username: str, password: str):
#     # Simple credential check for demo purposes
#     if (username, password) == ("admin", "admin"):
#         return cl.User(
#             identifier="admin", metadata={"role": "admin", "provider": "credentials"}
#         )
#     else:
#         return None

# # --- Persistent Data Layer using Pickle ---
# class PickleDataLayer(cl.data.BaseDataLayer):
#     def __init__(self, file_path: str = "chainlit_data.pkl"):
#         self.file_path = file_path
#         if os.path.exists(self.file_path):
#             with open(self.file_path, "rb") as f:
#                 self.store = pickle.load(f)
#         else:
#             # Initialize store with a schema-like structure
#             self.store = {
#                 "users": {},       # key = identifier, value = {id, identifier, createdAt, metadata}
#                 "threads": {},     # key = thread_id, value = thread dict
#                 "feedbacks": {},   # key = feedback_id, value = feedback dict
#                 "elements": {},    # key = element_id, value = element dict
#                 "steps": {},       # key = step_id, value = step dict
#                 "sessions": {},
#             }
#             self._persist()

#     def build_debug_url(self, id: str) -> str:
#         return ""

#     def _persist(self):
#         """Save the store to the pickle file."""
#         with open(self.file_path, "wb") as f:
#             pickle.dump(self.store, f)

#     # --- User Methods ---
#     async def get_user(self, identifier: str) -> Optional[Dict[str, Any]]:
#         return self.store["users"].get(identifier)

#     async def create_user(self, user: cl.User) -> Dict[str, Any]:
#         data = {
#             "id": str(uuid.uuid4()),
#             "identifier": user.identifier,
#             "createdAt": str(uuid.uuid1()),
#             "metadata": user.metadata,
#         }
#         self.store["users"][user.identifier] = data
#         self._persist()
#         return data

#     # --- Thread Methods ---
#     async def list_threads(self, pagination: Any, filters: Any) -> Any:
#         from chainlit.data.sql_alchemy import PaginatedResponse
#         threads = list(self.store["threads"].values())
#         total = len(threads)
#         page_info = {
#             "page": getattr(pagination, 'page', 1),
#             "pageSize": getattr(pagination, 'page_size', total),
#             "totalPages": 1,
#             "hasNextPage": False,
#             "startCursor": None,
#             "endCursor": None
#         }
#         return PaginatedResponse(data=threads, total=total, pageInfo=page_info)

#     async def delete_thread(self, thread_id: str) -> bool:
#         """Delete a thread from the store by its ID."""
#         if thread_id in self.store["threads"]:
#             del self.store["threads"][thread_id]
#             self._persist()
#             return True
#         return False

#     async def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
#         """Retrieve a thread from the store by its ID."""
#         return self.store["threads"].get(thread_id)

#     async def update_thread(self, thread: Dict[str, Any], **kwargs) -> None:
#         """Update or add a thread in the store."""
#         thread_id = thread.get("id")
#         if not thread_id:
#             raise ValueError("Thread dictionary must contain an 'id' key")
#         self.store["threads"][thread_id] = thread
#         self._persist()

#     # --- Feedback Methods ---
#     async def upsert_feedback(self, feedback: Dict[str, Any]) -> str:
#         fid = feedback.get("id", str(uuid.uuid4()))
#         self.store["feedbacks"][fid] = feedback
#         self._persist()
#         return fid

#     async def delete_feedback(self, feedback_id: str) -> bool:
#         return self.store["feedbacks"].pop(feedback_id, None) is not None

#     # --- Element Methods ---
#     async def create_element(self, element: Dict[str, Any]) -> None:
#         eid = element.get("id", str(uuid.uuid4()))
#         self.store["elements"][eid] = element
#         self._persist()

#     async def get_element(self, thread_id: str, element_id: str) -> Optional[Dict[str, Any]]:
#         el = self.store["elements"].get(element_id)
#         return el if el and el.get("threadId") == thread_id else None

#     async def delete_element(self, element_id: str) -> None:
#         self.store["elements"].pop(element_id, None)
#         self._persist()

#     # --- Step Methods ---
#     async def create_step(self, step: Dict[str, Any]) -> None:
#         sid = step.get("id", str(uuid.uuid4()))
#         self.store["steps"][sid] = step
#         self._persist()

#     async def update_step(self, step: Dict[str, Any]) -> None:
#         self.store["steps"].update({step.get("id"): step})
#         self._persist()

#     async def delete_step(self, step_id: str) -> None:
#         self.store["steps"].pop(step_id, None)
#         self._persist()

#     async def get_thread_author(self, thread_id: str) -> Optional[str]:
#         thr = self.store["threads"].get(thread_id)
#         return thr.get("userId") if thr else None

#     async def delete_user_session(self, id: str) -> bool:
#         return self.store["sessions"].pop(id, None) is not None

# # Register the data layer
# @cl.data_layer
# def get_data_layer():
#     return PickleDataLayer()

# # --- Chainlit App Setup ---
# if "GROQ_API_KEY" not in os.environ:
#     os.environ["GROQ_API_KEY"] = cl.getpass("Enter your Groq API key:")

# # In-memory session histories
# session_histories: Dict[str, List[Dict[str, str]]] = {}

# class MessageHistory:
#     def __init__(self, messages: List[Dict[str, str]] = None):
#         self.messages = messages or []
#     def add_messages(self, role: str, content: str):
#         self.messages.append({"role": role, "content": content})
#     async def aget_messages(self):
#         return self.messages

# def get_session_history(session_id: str) -> MessageHistory:
#     if session_id not in session_histories:
#         session_histories[session_id] = []
#     return MessageHistory(session_histories[session_id])

# def create_chain_with_chat_history(final_chain: Runnable) -> Runnable:
#     def get_messages(input: Dict):
#         session_id = input.get("session_id", "")
#         history = get_session_history(session_id)
#         return history.messages

#     return (
#         RunnablePassthrough.assign(chat_history=get_messages)
#         | final_chain
#     ).with_config(
#         RunnableConfig(id="session_id", description="Session identifier", default="", is_shared=True)
#     )

# @cl.on_chat_start
# async def on_chat_start():  # Make this async
#     session_id = str(uuid.uuid4())
#     cl.user_session.set("session_id", session_id)

#     # Create and store a new thread
#     thread_id = str(uuid.uuid4())
#     thread = {
#         "id": thread_id,
#         "createdAt": datetime.now().isoformat(),
#         "name": "New Conversation",
#         "userId": cl.user_session.get("user").identifier if cl.user_session.get("user") else None,
#         "steps": [],
#         "elements": [],
#     }
#     data_layer = cl.get_data_layer()
#     await data_layer.update_thread(thread)
#     cl.user_session.set("thread_id", thread_id)

#     # Setup RAG components
#     model = ChatGroq(model="llama3-8b-8192", temperature=0)
#     embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
#     vector_store = Chroma(collection_name="warehouse_collection",
#                          embedding_function=embeddings,
#                          persist_directory="./chromadb")
#     retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

#     prompt = PromptTemplate.from_template(
#         """
# # Warehouse DB RAG Chatbot Prompt

# Question: {question}
# chat history: {chat_history}
# stock : {docs}
# """
#     )
#     runnable = (
#         RunnablePassthrough.assign(docs=lambda x: retriever.invoke(x["question"]))
#         | prompt
#         | model
#         | StrOutputParser()
#     )
#     chain = create_chain_with_chat_history(runnable)
#     cl.user_session.set("chain", chain)

# @cl.on_message
# async def on_message(message: cl.Message):
#     session_id = cl.user_session.get("session_id")
#     thread_id = cl.user_session.get("thread_id")  # Get thread ID
#     chain = cl.user_session.get("chain")
#     history = get_session_history(session_id)
#     history.add_messages("user", message.content)

#     # Create user step
#     user_step = {
#         "id": str(uuid.uuid4()),
#         "threadId": thread_id,
#         "name": "User Message",
#         "createdAt": datetime.now().isoformat(),
#         "type": "user_message",
#         "output": message.content,
#     }
#     data_layer = cl.get_data_layer()
#     await data_layer.create_step(user_step)

#     msg = cl.Message(content="")
#     response = ""
#     async for chunk in chain.astream(
#         {"question": message.content}, 
#         config={"session_id": session_id}
#     ):
#         await msg.stream_token(chunk)
#         response += chunk
    
#     history.add_messages("assistant", response)
#     await msg.send()

#     # Create assistant step
#     assistant_step = {
#         "id": str(uuid.uuid4()),
#         "threadId": thread_id,
#         "name": "Assistant Response",
#         "createdAt": datetime.now().isoformat(),
#         "type": "assistant_message",
#         "output": response,
#     }
#     await data_layer.create_step(assistant_step)

from typing import Dict, List, Optional, Any
import os
import uuid
import pickle
import asyncio
import chainlit as cl
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from chainlit.types import ThreadDict

# Load environment variables
load_dotenv()

# --- Authentication ---
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "admin", "provider": "credentials"})
    return None

# --- Persistent Data Layer using Pickle ---
class PickleDataLayer(cl.data.BaseDataLayer):
    def __init__(self, file_path: str = "chainlit_data.pkl"):
        self.file_path = file_path
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as f:
                self.store = pickle.load(f)
        else:
            self.store = {
                "users": {},
                "threads": {},
                "steps": {},
                "sessions": {},
            }
            self._persist()

    def build_debug_url(self, id: str) -> str:
        return ""

    def _persist(self):
        with open(self.file_path, "wb") as f:
            pickle.dump(self.store, f)

    # --- User Methods ---
    async def get_user(self, identifier: str) -> Optional[Dict[str, Any]]:
        return self.store["users"].get(identifier)

    async def create_user(self, user: cl.User) -> Dict[str, Any]:
        data = {
            "id": str(uuid.uuid4()),
            "identifier": user.identifier,
            "createdAt": datetime.now().isoformat(),
            "metadata": user.metadata,
        }
        self.store["users"][user.identifier] = data
        self._persist()
        return data

    # --- Thread Methods ---
    async def list_threads(self, pagination: Any, filters: Any) -> Any:
        from chainlit.data.sql_alchemy import PaginatedResponse
        threads = list(self.store["threads"].values())
        
        # Apply user filter if needed
        if filters and filters.userIdentifier:
            threads = [t for t in threads if t.get("userId") == filters.userIdentifier]
        
        # Sort by creation date (newest first)
        threads.sort(key=lambda x: x["createdAt"], reverse=True)
        
        total = len(threads)
        page = pagination.page if pagination else 1
        page_size = pagination.pageSize if pagination else total
        
        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        paginated_threads = threads[start:end]
        
        page_info = {
            "page": page,
            "pageSize": page_size,
            "totalPages": max(1, (total + page_size - 1) // page_size),
            "hasNextPage": end < total,
            "startCursor": None,
            "endCursor": None
        }
        return PaginatedResponse(data=paginated_threads, total=total, pageInfo=page_info)

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        thread = self.store["threads"].get(thread_id)
        if not thread:
            return None
        
        # Add steps to thread dictionary
        thread_steps = [
            step for step in self.store["steps"].values() 
            if step["threadId"] == thread_id
        ]
        # Sort steps by creation time
        thread_steps.sort(key=lambda x: x["createdAt"])
        thread["steps"] = thread_steps
        
        return thread

    async def delete_thread(self, thread_id: str) -> bool:
        if thread_id in self.store["threads"]:
            # Delete associated steps
            step_ids = [
                step_id for step_id, step in self.store["steps"].items()
                if step["threadId"] == thread_id
            ]
            for step_id in step_ids:
                del self.store["steps"][step_id]
            
            # Delete thread
            del self.store["threads"][thread_id]
            self._persist()
            return True
        return False

    async def update_thread(self, thread: ThreadDict, **kwargs) -> None:
        thread_id = thread["id"]
        # Don't store steps directly in thread
        thread_data = thread.copy()
        if "steps" in thread_data:
            del thread_data["steps"]
            
        self.store["threads"][thread_id] = thread_data
        self._persist()

    # --- Step Methods ---
    async def create_step(self, step: Dict[str, Any]) -> None:
        step_id = step.get("id", str(uuid.uuid4()))
        step["id"] = step_id
        self.store["steps"][step_id] = step
        self._persist()

    async def update_step(self, step: Dict[str, Any]) -> None:
        step_id = step["id"]
        self.store["steps"][step_id] = step
        self._persist()

    async def delete_step(self, step_id: str) -> None:
        self.store["steps"].pop(step_id, None)
        self._persist()

    async def get_thread_author(self, thread_id: str) -> Optional[str]:
        thread = self.store["threads"].get(thread_id)
        return thread.get("userId") if thread else None

    async def delete_user_session(self, id: str) -> bool:
        return self.store["sessions"].pop(id, None) is not None

# Register the data layer
@cl.data_layer
def get_data_layer():
    return PickleDataLayer()

# --- Session Management ---
# Fixed the type annotation syntax error
session_histories: Dict[str, List[Dict[str, str]]] = {}

def get_session_history(thread_id: str) -> List[Dict[str, str]]:
    if thread_id not in session_histories:
        session_histories[thread_id] = []
    return session_histories[thread_id]

# Initialize ChatGroq model
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = cl.getpass("Enter your Groq API key:")

model = ChatGroq(model="llama3-8b-8192", temperature=0)

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond to the user's questions clearly and concisely."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

# Create the chain
chain = (
    RunnablePassthrough.assign(
        chat_history=lambda x: [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" 
            else AIMessage(content=msg["content"])
            for msg in x["chat_history"]
        ]
    )
    | prompt
    | model
)

# --- Chainlit Handlers ---
@cl.on_chat_start
async def on_chat_start():
    # Create a new thread
    thread_id = str(uuid.uuid4())
    user = cl.user_session.get("user")
    
    thread = {
        "id": thread_id,
        "createdAt": datetime.now().isoformat(),
        "name": "New Conversation",
        "userId": user.identifier if user else "anonymous",
    }
    
    data_layer = cl.get_data_layer()
    await data_layer.update_thread(thread)
    cl.user_session.set("thread_id", thread_id)
    
    # Initialize session history
    session_histories[thread_id] = []
    await cl.Message(content="Chat started! How can I help you?").send()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    thread_id = thread["id"]
    cl.user_session.set("thread_id", thread_id)
    
    # Rebuild session history from stored steps
    history = []
    for step in thread["steps"]:
        if step["type"] == "user_message":
            history.append({"role": "user", "content": step["output"]})
        elif step["type"] == "assistant_message":
            history.append({"role": "assistant", "content": step["output"]})
    
    session_histories[thread_id] = history
    await cl.Message(content=f"Resumed conversation {thread_id}").send()

@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    history = get_session_history(thread_id)
    data_layer = cl.get_data_layer()
    
    # Add user message to history
    history.append({"role": "user", "content": message.content})
    
    # Create user step in data layer
    user_step = {
        "id": str(uuid.uuid4()),
        "threadId": thread_id,
        "name": "User Message",
        "createdAt": datetime.now().isoformat(),
        "type": "user_message",
        "output": message.content,
    }
    await data_layer.create_step(user_step)
    
    # Create assistant message
    msg = cl.Message(content="")
    await msg.send()
    
    # Generate response
    response_content = ""
    inputs = {"input": message.content, "chat_history": history}
    
    async for chunk in chain.astream(inputs):
        if hasattr(chunk, 'content'):
            token = chunk.content
            if token:
                response_content += token
                await msg.stream_token(token)
    
    # Update assistant message
    await msg.update()
    
    # Add assistant response to history
    history.append({"role": "assistant", "content": response_content})
    
    # Create assistant step in data layer
    assistant_step = {
        "id": str(uuid.uuid4()),
        "threadId": thread_id,
        "name": "Assistant Response",
        "createdAt": datetime.now().isoformat(),
        "type": "assistant_message",
        "output": response_content,
    }
    await data_layer.create_step(assistant_step)