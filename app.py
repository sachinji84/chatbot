import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional
import uuid
import datetime

# Initialize FastAPI app
app = FastAPI()

# Static and templates setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = "chromadb"  # Path where your ChromaDB is stored

# Initialize the OpenAI client with LangChain
llm_model = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4")

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage
documents = {}
conversation_history = {}

# Initialize vector database
vectordb = None


def initialize_vector_db():
    global vectordb
    if vectordb is None:
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
    return vectordb


initialize_vector_db()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the upload page."""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, document: UploadFile = File(...)):
    """Handle PDF upload and process the document."""
    if document.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_path = os.path.join(UPLOAD_FOLDER, document.filename)
    with open(file_path, "wb") as f:
        f.write(await document.read())

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(content)

        vectordb.add_texts(
            texts=chunks,
            ids=[f"{document.filename}-{i}" for i in range(len(chunks))]
        )

        vectordb.persist()

        doc_id = len(documents) + 1
        documents[doc_id] = document.filename

        return templates.TemplateResponse("chat.html", {"request": request, "doc_id": doc_id})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{doc_id}", response_class=HTMLResponse)
async def chat(doc_id: int, request: Request):
    """Render the chat page."""
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    return templates.TemplateResponse("chat.html", {"request": request, "doc_id": doc_id})


class ChatRequest(BaseModel):
    query: str


@app.post("/interact/{doc_id}")
async def interact_with_document(doc_id: int, chat_request: ChatRequest):
    """Handle chat interactions."""
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")

    user_query = chat_request.query
    search_results = vectordb.similarity_search_with_score(user_query, k=5)
    relevant_content = "\n".join([result[0].page_content for result in search_results])

    history = conversation_history.get(doc_id, "")

    template = (
        "You are a helpful assistant.\n"
        "Conversation history:\n{history}\n"
        "Always answer from given context. If information is not available, say so.\n"
        "Relevant document content:\n{relevant_content}\n"
        "User query: {user_query}"
    )

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["history", "relevant_content", "user_query"]
    )

    chain = prompt_template | llm_model

    try:
        response = chain.invoke({
            "history": history,
            "relevant_content": relevant_content,
            "user_query": user_query
        })

        conversation_history[doc_id] = history + f"User: {user_query}\nAssistant: {response.content}\n"

        return JSONResponse(content={"response": response.content})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session")
async def create_session(user_id: str, chatbot_name: str, first_query: str):
    """
    Create a new chat session and store metadata.
    """
    session_id = str(uuid.uuid4())
    session_name = first_query[:30]

    session_data = {
        "session_id": session_id,
        "user_id": user_id,
        "chatbot_name": chatbot_name,
        "session_name": session_name,
        "created_at": datetime.datetime.now().isoformat() + "Z"
    }

    try:
        vectordb.add_texts(
            texts=[str(session_data)],
            ids=[session_id]
        )
        vectordb.persist()
        return {"status": "success", "session_id": session_id, "session_name": session_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving session: {e}")


@app.get("/user/{user_id}/sessions")
async def list_sessions(user_id: str):
    """
    Retrieve all chat sessions for a user.
    """
    try:
        results = vectordb.similarity_search_with_score(user_id, k=100)
        sessions = [eval(res[0].page_content) for res in results if user_id in res[0].page_content]
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sessions: {e}")


@app.get("/session/{chatbot_name}/{session_id}/history")
async def fetch_messages(chatbot_name: str, session_id: str):
    """
    Retrieve all messages from a specific chat session.
    """
    try:
        results = vectordb.similarity_search_with_score(session_id, k=10)
        messages = [res[0].page_content for res in results if session_id in res[0].page_content]
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    #  print("FastAPI app is ready for use!")
