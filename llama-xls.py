import os
import re
import multiprocessing
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
from langchain_community.llms import Ollama


from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Enable GPU acceleration on Apple M2
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Use all available CPU cores
CPU_CORES = os.cpu_count()
model="mistral"
llm = Ollama(model=model, base_url="http://localhost:11434")



def parallel_split_text(df, chunk_size=100, chunk_overlap=50):
    """Split DataFrame into row-based chunks to maintain context."""
    chunks = []
    current_chunk = []
    current_size = 0

    for row in df.itertuples(index=False, name=None):
        # Combine key text columns
        row_text = ' '.join(map(str, [row[0], row[1], row[2], row[3], row[7], row[8], row[9], row[10]]))
        row_size = len(row_text)
        if current_size + row_size <= chunk_size:
            current_chunk.append(row_text)
            current_size += row_size
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [row_text]
            current_size = row_size

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def initialize_vectorstore(file_path, embeddings):
    """Initialize Chroma vectorstore only if necessary."""
    if os.path.exists("./chroma_db"):
        print("Loading existing Chroma vectorstore...")
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    print("Creating new Chroma vectorstore...")
    df = pd.read_excel(file_path, engine="openpyxl", dtype=str)
    df = df.fillna('')  # Handle NaN values
    chunks = parallel_split_text(df)
    metadatas = [{"index": i, "title": row[0]} for i, row in enumerate(df.values)]
    return Chroma.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas, persist_directory="./chroma_db")


def process_csv_file(file_path):
    """Load and process the dataset, generate embeddings, and create a retriever."""
    global vectorstore, retriever

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", show_progress=True, model=model)
    vectorstore = initialize_vectorstore(file_path, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def ollama_llm(question, context):
    # Create a memory object which will store the conversation history.
    memory = ConversationBufferMemory()

    # Create a chain with this memory object and the model object created earlier.
    chain = ConversationChain(
     llm=llm,
    memory=memory
   )

    """Query the LLM using provided question and context."""
    formatted_prompt = (
        f"Use ONLY the context below to answer the question.\n"
        f"If unsure, say 'I don’t know'.\n"
        f"Keep answers under 4 sentences.\n\n"
        f"Question: {question}\nContext: {context}"
    )

    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": "As a representative of BetterHome, a one-stop shop for home appliances, come up with recommendations based on the user request using the product catalogue data"},
        {"role": "user", "content": formatted_prompt}],
        options={"temperature": 0.2}
    )

    return response["message"]["content"].strip()


def compress_context(docs, max_tokens=800):
    """Compress context to improve LLM performance."""
    combined = " ".join(doc.page_content for doc in docs)
    return combined[:max_tokens] if len(combined) > max_tokens else combined


def ask_question(question):
    """Retrieve relevant documents and get an answer from the LLM."""
    retrieved_docs = retriever.invoke(question)
    context = compress_context(retrieved_docs)
    return ollama_llm(question, context)


if __name__ == "__main__":
    print("Starting script...")

    # File path
    FILE_PATH = "/Users/narasimhan/workspace/python-workbench/betterhome/products.xlsx"

    # Initialize vectorstore and retriever
    process_csv_file(FILE_PATH)

    # Interactive loop
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = ask_question(question)
        print(f"Answer: {answer}\n")
