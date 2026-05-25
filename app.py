import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

vector_store = None
retriever = None

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_document(file_path):
    extension = file_path.rsplit(".", 1)[1].lower()
    if extension == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf8")
    return loader.load()


def build_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)


def query_answer(question):
    global retriever
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    return qa_chain.run(question)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    global vector_store, retriever
    if "document" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["document"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF, TXT, and MD files are allowed."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    documents = load_document(file_path)
    vector_store = build_vector_store(documents)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    return jsonify({"message": f"Uploaded and processed {filename}. You can now ask questions."})


@app.route("/chat", methods=["POST"])
def chat():
    if retriever is None:
        return jsonify({"error": "Upload a document first."}), 400

    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        answer = query_answer(question)
        return jsonify({"answer": answer})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
