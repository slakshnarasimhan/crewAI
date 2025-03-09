import os
import sys
import time
import pandas as pd
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

import ollama

# Enable GPU acceleration on Apple M2
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

model = "mistral"
llm = Ollama(model=model, base_url="http://localhost:11434")
PRINT_MODE = "word"  # Change to "char" for character-by-character printing

def parallel_split_text(df):
    """Split DataFrame into smaller chunks with clear product information."""
    chunks = []
    for _, row in df.iterrows():
        brand = row.get("Brand", "").strip().lower()
        category = row.get("Type", "").strip().lower()
        product_name = row.get("Product Name", "")
        price = row.get("Price", "")
        retail_price = row.get("Retail Price", "")
        return_policy = row.get("Return Policy", "")
        warranty = row.get("Warranty", "")
        description = row.get("Description", "")
        
        product_info = f"Product: {product_name} | Type: {category} | Brand: {brand} "
        if price:
            product_info += f" | Price: {price}"
        if retail_price:
            product_info += f" | Retail Price: {retail_price}"
        if return_policy:
            product_info += f" | Return Policy: {return_policy}"
        if warranty:
            product_info += f" | Warranty: {warranty}"

        if product_name:
            chunks.append(product_info)

    return chunks

def initialize_vectorstore(file_path, embeddings):
    """Initialize Chroma vectorstore with products and expected answers."""
    if os.path.exists("./chroma_db-mistral"):
        print("Loading existing Chroma vectorstore...")
        return Chroma(persist_directory="./chroma_db-mistral", embedding_function=embeddings)

    print("Creating new Chroma vectorstore...")
    df = pd.read_csv(file_path, dtype=str).fillna('')
    chunks = parallel_split_text(df)

    metadatas = [{"index": i, "category": chunk.split("|")[0].split(":")[1].strip().lower()} 
                 for i, chunk in enumerate(chunks)]

    return Chroma.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas, persist_directory="./chroma_db-mistral")

def process_csv_file(file_path):
    """Load and process the dataset, generate embeddings, and create a retriever."""
    global vectorstore, retriever

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    df = pd.read_csv(file_path, dtype=str).fillna('')
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", show_progress=True, model=model)
    vectorstore = initialize_vectorstore(file_path, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 50, "fetch_k": 100})

if __name__ == "__main__":
    print("Starting script...")
    FILE_PATH = "products-clean.csv"  # Update to the CSV file path
    process_csv_file(FILE_PATH)

