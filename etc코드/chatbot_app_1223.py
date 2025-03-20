import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain.chains import RetrievalQA
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# .env 파일 로드
load_dotenv()

# Flask 앱 초기화
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # CORS 활성화

# OpenAI API Key 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

# LangChain 초기화
def initialize_rag_pipeline():
    """LangChain RAG 파이프라인 초기화"""
    # OpenAI 임베딩 초기화
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 데이터 로드 및 전처리
    file_path = "db/db_products.xlsx"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {file_path}")

    # 엑셀 데이터를 읽어오기
    data = pd.read_excel(file_path)
    if data.empty:
        raise ValueError("엑셀 파일이 비어 있습니다. 데이터를 확인하세요.")

    # 데이터프레임을 LangChain 문서 형식으로 변환
    documents = [
        Document(
            page_content=(
                f"Category: {row['Category']}, Sub-category: {row['Sub-category']}, "
                f"Style: {row['Style']}, Season: {row['Season']}, Material: {row['Material']}, "
                f"Fit: {row['Fit']}, Length: {row['Length']}, Color: {row['Color']}, "
                f"Situation: {row['Situation']}, Target: {row['Target']}, "
                f"Size: {row['Size']}, Price: {row['Price (KRW)']}"
            ),
            metadata={"id": idx}
        )
        for idx, row in data.iterrows()
    ]

    # 텍스트 분리
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_documents = text_splitter.split_documents(documents)

    # FAISS 벡터스토어 초기화
    vectorstore = FAISS.from_documents(split_documents, embeddings)

    # OpenAI LLM 초기화
    llm = OpenAI(openai_api_key=openai_api_key)

    # RetrievalQA 체인 초기화
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_chain

# RAG 파이프라인 초기화
try:
    qa_chain = initialize_rag_pipeline()
except Exception as e:
    print(f"Error during RAG pipeline initialization: {e}")
    qa_chain = None

# 라우트 설정
@app.route("/")
def home():
    """홈 페이지"""
    return render_template("bot.html")  # templates/bot.html 파일 렌더링

@app.route("/chat", methods=["POST"])
def chat():
    """챗봇 API"""
    if qa_chain is None:
        return jsonify({"error": "RAG 파이프라인 초기화에 실패했습니다. 서버 로그를 확인하세요."}), 500

    data = request.json
    input_text = data.get("input_text", "")

    try:
        # LangChain RAG 파이프라인으로 응답 생성
        response = qa_chain.invoke({"query": input_text})
        return jsonify({"input_text": input_text, "response": response["result"]})
    except Exception as e:
        print(f"Error during chat processing: {str(e)}")
        return jsonify({"error": "서버 오류가 발생했습니다. 다시 시도해주세요."}), 500

# Flask 실행
if __name__ == "__main__":
    if not openai_api_key:
        print("Error: OpenAI API Key is not set. Please check your .env file.")
        exit(1)

    app.run(host="localhost", port=5500, debug=True)
