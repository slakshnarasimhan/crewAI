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
model="llama3.2"
llm = Ollama(model=model, base_url="http://localhost:11434")

def parallel_split_text(df, chunk_size=100, chunk_overlap=50):
    """Split DataFrame into smaller chunks with clear product information."""
    chunks = []
    for _, row in df.iterrows():
        category = row.get("Product Category", "").strip().lower()
        brand = row.get("Brand", "")
        product_name = row.get("Product Name", "")
        price = row.get("Price", "")
        retail_price = row.get("Retail Price", "")
        description = row.get("Description", "")
        features = row.get("Features", "")
        warranty = row.get("Warranty", "")


        # Combine product name and price into one chunk
        product_info = f"Product: {product_name} | Category: {category} | Brand: {brand}"
        if price:
            product_info += f" | Price: {price}"
        if retail_price:
            product_info += f" | Retail Price: {retail_price}"
        if description:
            product_info += f" | Description: {description}"
        if features:
            product_info += f" | Features: {features}"
        if warranty:
            product_info += f" | Warranty: {warranty}"

        # Only append if product name is available
        if product_name:
            chunks.append(product_info)

    return chunks



def initialize_vectorstore(file_path, embeddings):
    """Initialize Chroma vectorstore with products and expected answers."""
    if os.path.exists("./chroma_db"):
        print("Loading existing Chroma vectorstore...")
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    print("Creating new Chroma vectorstore...")
    df = pd.read_excel(file_path, engine="openpyxl", dtype=str)
    df = df.fillna('')

    chunks = parallel_split_text(df)

    # Expected answers should be added as metadata without overriding product data
    expected_answers = [
        "Product Category: Chimney | Brands: Faber, Elica, Glen, Kaff",
        "Product Category: Water Heater | Brands: AO Smith, Racold, Havells",
        "Product Category: Fan | Brands: Crompton, Havells, Orient"
    ]
    chunks.extend(expected_answers)

    metadatas = [{"index": i, "category": chunk.split("|")[0].split(":")[1].strip().lower()} 
                 for i, chunk in enumerate(chunks)]

    return Chroma.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas, persist_directory="./chroma_db")


def process_csv_file(file_path):
    """Load and process the dataset, generate embeddings, and create a retriever."""
    global vectorstore, retriever

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", show_progress=True, model=model)
    vectorstore = initialize_vectorstore(file_path, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 50, "fetch_k": 20})

def ollama_llm(question, context):
    """Query the LLM using provided question, context, and examples."""
    few_shots = """
    Examples:
    Q: What chimney brands are available?
    A: The available chimney brands are Faber, Elica, Glen, and Kaff.

    Q: What water heater brands do you have?
    A: The available water heater brands are AO Smith, Racold, and Havells.

    Q: What fan brands do you have?
    A: The available fan brands are Crompton, Havells, and Orient.

    Q: What's the price range of the hob tops you offer?
    A: The price range of hob tops is between 20000 and 50000 INR.

    Q: Show all product names and their prices.
    A: 1. Product: Faber Chimney | Price: 79191
       2. Product: Glen Chimney | Price: 40041
       3. Product: AO Smith Water Heater | Price: 29285
    """

    formatted_prompt = (
        f"Use ONLY the provided product catalogue context to answer the question.\n"
        f"If the context doesn't include the requested information, respond with 'I don’t know'.\n"
        f"Keep the answer concise and accurate.\n\n"
        f"{few_shots}\n"
        f"Question: {question}\nContext: {context}"
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a product expert at BetterHome, providing product information strictly from the product catalogue."},
            {"role": "user", "content": formatted_prompt}
        ],
        options={"temperature": 0.2}
    )

    return response["message"]["content"].strip()


    formatted_prompt = (
    f"Using ONLY the provided product catalogue context, answer the question.\n"
    f"Focus on the specific product category mentioned in the question.\n"
    f"Focus on price information when asked about price range.\n"
    f"If no relevant information is found, respond with 'I don’t know'.\n"
    f"Keep the answer concise and list product brands when applicable.\n\n"
    f"Question: {question}\nContext: {context}"
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a product expert for BetterHome. Provide product recommendations strictly based on the uploaded catalogue."},
            {"role": "user", "content": formatted_prompt}
        ],
        options={"temperature": 0.2}
    )

    return response["message"]["content"].strip()


def compress_context(docs, max_tokens=5000):
    """Compress context to improve LLM performance."""
    combined = " ".join(doc.page_content for doc in docs)
    return combined[:max_tokens] if len(combined) > max_tokens else combined

def ask_question(question):
    """Retrieve relevant documents and get an answer from the LLM."""
    # Identify product category from the question using simple keyword matching
    category_keywords = {
    "chimney": "Chimney",
    "water heater": "Water Heater",
    "fan": "Fan",
    "microwave": "Microwave",
    "hob top": "Hob Top",
    "refrigerator": "Refrigerator"
    }

    # Initialize category_filter to avoid UnboundLocalError
    category_filter = None

    # Determine category based on question
    for keyword, category in category_keywords.items():
        if keyword in question.lower():
            category_filter = category
            break

    # Determine category based on question
    if category_filter:
       retrieved_docs = retriever.invoke(question, search_kwargs={"filter": {"category": category_filter.lower()}})
    else:
       retrieved_docs = retriever.invoke(question)

    for keyword, category in category_keywords.items():
        if keyword in question.lower():
            category_filter = category
            break

    # Retrieve documents with or without category filter
    if category_filter:
        retrieved_docs = retriever.invoke(question, search_kwargs={"filter": {"category": category_filter}})
    else:
        retrieved_docs = retriever.invoke(question)

    #print("\n--- Retrieved Documents ---")
    #for doc in retrieved_docs:
    #    print(doc.page_content)
    #print("---------------------------\n")

    context = compress_context(retrieved_docs)
    return ollama_llm(question, context)


if __name__ == "__main__":
    print("Starting script...")

    # File path
    FILE_PATH = "/Users/narasimhan/workspace/python-workbench/betterhome/sample.xlsx"

    # Initialize vectorstore and retriever
    process_csv_file(FILE_PATH)
    #df = pd.read_excel(FILE_PATH, engine="openpyxl", dtype=str)
    #print(df.columns) 

    #print(ask_question("What chimney brands are available?"))
    #print(ask_question("What water heater brands do you have?"))
    #print(ask_question("whats the price range of the hob tops you offer."))
    #print(ask_question("show all product names and their prices"))

    # Interactive loop
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = ask_question(question)
        print(f"Answer: {answer}\n")
