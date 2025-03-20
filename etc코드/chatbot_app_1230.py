import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# .env 파일 로드
load_dotenv()

# Flask 앱 초기화
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# OpenAI API Key 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API Key가 설정되지 않았습니다.")

# LangChain 초기화
def initialize_rag_pipeline():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 데이터 로드 및 전처리
    file_path = "db/Infoitems.xlsx"  # 교체된 파일 경로
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {file_path}")

    # 엑셀 데이터를 읽기
    data = pd.read_excel(file_path)
    if data.empty:
        raise ValueError("엑셀 파일이 비어 있습니다. 데이터를 확인하세요.")

    # 컬럼 이름 전처리
    data.columns = data.columns.str.strip()

    # 필요한 컬럼이 있는지 확인
    required_columns = ['No', 'ID', 'Title', 'Brand', 'Price', 'Option', 'Option 1', 'URL']
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"엑셀 데이터에 '{col}' 컬럼이 없습니다. 파일을 확인하세요.")

    # 데이터프레임을 LangChain 문서로 변환
    documents = [
        Document(
            page_content=f"Title: {row['Title']}",
            metadata={"price": row['Price'], "url": row['URL']}
        )
        for _, row in data.iterrows()
    ]

    # 텍스트 분리
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_documents = text_splitter.split_documents(documents)

    # FAISS 벡터 저장소 초기화
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
    print(f"RAG 파이프라인 초기화 오류: {e}")
    qa_chain = None

# 라우트 설정
@app.route("/")
def home():
    return render_template("bot.html")

@app.route("/chat", methods=["POST"])
def chat():
    if qa_chain is None:
        return jsonify({"error": "RAG 파이프라인 초기화 실패. 서버 로그를 확인하세요."}), 500

    data = request.json
    input_text = data.get("input_text", "")

    try:
        # RAG 기반 검색 실행
        response = qa_chain.invoke({"query": input_text})

        # 결과에서 Price와 URL 추출
        documents = response.get("documents", [])
        results = [
            {"title": doc.page_content, "price": doc.metadata["price"], "url": doc.metadata["url"]}
            for doc in documents
        ]

        return jsonify({"results": results})
    except Exception as e:
        print(f"Chat 처리 중 오류: {e}")
        return jsonify({"error": "서버 오류가 발생했습니다. 다시 시도해주세요."}), 500

if __name__ == "__main__":
    app.run(host="localhost", port=5500, debug=True)
