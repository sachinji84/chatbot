import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body  # Add this import

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify a list of allowed domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Set up the templates directory
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Directory for uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = "chromadb"  # Path where your ChromaDB will be stored

# Initialize the OpenAI client with LangChain
llm_model = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4")

# In-memory storage for simplicity (consider using a database in production)
documents = {}

# Initialize the vector database
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

# Initialize the vector store when the application starts
initialize_vector_db()

# Route to display upload page
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# Handle PDF upload and process the document
@app.post("/upload")
async def handle_upload(document: UploadFile = File(...)):  # Use 'document' here, not 'file'
     # After processing the file, add the filename to the documents dict
    documents[document.filename] = "Initial conversation history"  # Example of adding history for future interactions
    print(f"Document {document.filename} added to documents.")
    # Check if the uploaded file is a PDF
    if not document.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed"}

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, document.filename)
    with open(file_path, "wb") as f:
        content = await document.read()  # 'document' is the correct variable name
        f.write(content)

    # Process the document (e.g., extract text from PDF)
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])

        # Split and add to the vector database
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(content)

        vectordb.add_texts(
            texts=chunks,
            ids=[f"{document.filename}-{i}" for i in range(len(chunks))]
        )

        vectordb.persist()

        print(f"Processed {len(chunks)} chunks.")

        return RedirectResponse(url=f"/chat/{document.filename}", status_code=303)

    except Exception as e:
        return {"error": str(e)}

# Route to render chat page
@app.get("/chat/{filename}")
async def chat(request: Request, filename: str):
    return templates.TemplateResponse("chat.html", {"request": request, "filename": filename})

# Handle chat interactions
@app.post("/interact/{filename}")
async def interact_with_document(filename: str, query: str = Body(...)):  # Ensure the query comes from the body
    print(f"Received query: {query}")  # Debugging line to see the incoming query
    # Check if the document exists in the vector store
    if filename not in documents:
        return {"error": "Document not found"}

    # Debugging: print the incoming query
    print(f"Received query for {filename}: {query}")

    # Search the vector store for relevant content
    search_results = vectordb.similarity_search_with_score(query, k=5)
    if not search_results:
        return {"response": "No relevant content found."}

    relevant_content = "\n".join([result[0].page_content for result in search_results])

    # Retrieve or initialize conversation history
    history = documents.get(filename, "")

    # Create the prompt template for generating responses
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
            "user_query": query
        })

        # Store conversation history for future interactions
        documents[filename] = history + f"User: {query}\nAssistant: {response.content}\n"

        # Debugging: print the response generated
        print(f"Generated response: {response.content}")

        return {"response": response.content}

    except Exception as e:
        return {"error": str(e)}

# Run the FastAPI application (using Uvicorn)
# If you're running the script directly with Python, you can use the following block:

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",  # Pass the app as an import string
        host="127.0.0.1",
        port=8000,
        reload=True  # Enable reload
    )
