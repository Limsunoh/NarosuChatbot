import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.llms import OpenAI as LlmOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document  # 아직 community에서 지원하지 않음

# Load .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__, template_folder="templates")

# Load Excel database
db_path = "db/db_products.xlsx"  # Adjust path to your actual file
df = pd.read_excel(db_path)

# Helper function to create embedding text
def create_text_for_embedding(row):
    """
    Generates text for embeddings from the given row.
    """
    parts = [
        str(row["Category"]),
        str(row["Sub-category"]),
        str(row["Style"]),
        str(row["Season"]),
        str(row["Material"]),
        str(row["Fit"]),
        str(row["Length"]),
        str(row["Color"]),
        str(row["Situation"]),
        str(row["Target"]),
        str(row["Size"]),
    ]
    return " ".join(parts)

# Create LangChain Documents
docs = []
for _, row in df.iterrows():
    content = create_text_for_embedding(row)
    meta = {
        "name": row["name"],
        "price": row["Price (KRW)"],
        "category": row["Category"],
    }
    doc = Document(page_content=content, metadata=meta)
    docs.append(doc)

# Embeddings and Vector Store
embedding_fn = OpenAIEmbeddings(
    openai_api_key=openai_api_key,
    model="text-embedding-ada-002"
)
vectorstore = FAISS.from_documents(docs, embedding_fn)

# LLM and Retrieval Chain
llm = LlmOpenAI(
    openai_api_key=openai_api_key,
    model_name="text-davinci-003",
    temperature=0.0
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "메시지가 입력되지 않았습니다."}), 400
    
    try:
        # Debugging input
        print(f"Received message: {user_input}")

        # Process user input with LangChain
        result = qa_chain.invoke({"input": user_input})
        answer = result["output"]
        source_docs = result["source_documents"]

        recommended_products = []
        for doc in source_docs:
            recommended_products.append({
                "name": doc.metadata["name"],
                "price": doc.metadata["price"],
                "category": doc.metadata["category"]
            })

        # Debugging output
        response_data = {"response": answer, "products": recommended_products}
        print("Sending response:", response_data)

        return jsonify(response_data)

    except Exception as e:
        # Debugging errors
        print("Error occurred:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
