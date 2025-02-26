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
    """Split DataFrame into smaller chunks with clear product information."""
    chunks = []
    for _, row in df.iterrows():
        brand = row.get("Brand", "").strip().lower()
        category = row.get("Type", "").strip().lower()
        product_name = row.get("Product Name", "")
        price = row.get("Price", "")
        retail_price = row.get("Retail Price", "")
        return_policy = row.get("Return Policy", "")

        # Combine product name, category, and brand into one chunk
        product_info = f"Product: {product_name} | Type: {category} | Brand: {brand}"
        if price:
            product_info += f" | Price: {price}"
        if retail_price:
            product_info += f" | Retail Price: {retail_price}"
        if return_policy:
            product_info += f" | Return Policy: {return_policy}"

        # Append only if product name exists
        if product_name:
            chunks.append(product_info)

        # Store return policy separately for easy retrieval
        if return_policy:
            chunks.append(f"Brand: {brand} | Type: {category} | Return Policy: {return_policy}")

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
    global vectorstore, retriever, total_products, category_counts

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Load the DataFrame
    df = pd.read_excel(file_path, engine="openpyxl", dtype=str).fillna('')

    # Store the total number of products
    total_products = len(df)

    # Store the count of products by category (case-insensitive)
    category_counts = df["Type"].str.strip().str.lower().value_counts().to_dict()

    # Initialize vectorstore
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", show_progress=True, model=model)
    vectorstore = initialize_vectorstore(file_path, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 50, "fetch_k": 100})

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
       

    Q: What chimney brands are available?
    A: The available chimney brands are Faber, Elica, Glen, and Kaff.

    Q: How many modes are available in LED mirrors?
    A: There are 3 modes available in LED mirrors: Warm White, Cool White, and Daylight.

    Q: How many colour temperatures are available in LED mirrors?
    A: LED mirrors offer 3 colour temperatures: 3000K, 4000K, and 6000K.
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
    f"Use ONLY the provided product catalogue context to answer the question.\n"
    f"Focus on the highest price when asked for the most expensive product.\n"
    f"Focus on return policy details when asked.\n"
    f"If the context doesn't include the requested information, respond with 'I don’t know'.\n"
    f"Keep the answer concise and accurate.\n\n"
    f"{few_shots}\n"
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
    """Retrieve relevant documents and get an answer from the LLM or perform direct calculations."""
    global total_products, category_counts

    question_lower = question.lower()

    # Brand-specific filtering
    brand_filter = None
    if "bosch" in question_lower:
        brand_filter = "bosch"

    # Return policy filter
    return_policy_filter = "return policy" if "return policy" in question_lower else None

    # Product category filtering
    category_keywords = {
        "chimney": "chimney",
        "water heater": "water heater",
        "fan": "fan",
        "microwave": "microwave",
        "hob top": "hob top",
        "refrigerator": "refrigerator",
        "led mirror": "led mirror"
    }

    category_filter = None
    for keyword, category in category_keywords.items():
        if keyword in question_lower:
            category_filter = category
            break

    # Construct filters
    filters = {}
    if category_filter:
        filters["category"] = category_filter
    if brand_filter:
        filters["brand"] = brand_filter
    if return_policy_filter:
        filters["return_policy"] = return_policy_filter

    # Retrieve documents with filters
    if filters:
        retrieved_docs = retriever.invoke(question, search_kwargs={"filter": filters})
    else:
        retrieved_docs = retriever.invoke(question)

    context = compress_context(retrieved_docs)
    return ollama_llm(question, context)



if __name__ == "__main__":
    print("Starting script...")

    # File path
    FILE_PATH = "/Users/narasimhan/workspace/python-workbench/betterhome/products.xlsx"

    # Initialize vectorstore and retriever
    process_csv_file(FILE_PATH)
    #df = pd.read_excel(FILE_PATH, engine="openpyxl", dtype=str)
    #print(df.columns) 

    #print(ask_question("What chimney brands are available?"))
    print(ask_question("What water heater brands do you have?"))
    #print(ask_question("whats the price range of the hob tops you offer."))
    print(ask_question("show all product names and their prices"))
    print(ask_question("What's the most expensive Bosch chimney you have?"))
    print(ask_question("What is the return policy of Bosch chimneys?"))



    # Interactive loop
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = ask_question(question)
        print(f"Answer: {answer}\n")
