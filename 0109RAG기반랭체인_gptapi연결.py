from dotenv import load_dotenv
import os
import time
from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_chroma import Chroma
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

# 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
persist_directory = './chroma_db'

# OpenAI 임베딩 및 LLM 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
llm = OpenAI(api_key=API_KEY)

# 프롬프트 템플릿 정의
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="질문: {question}\n문맥: {context}\n정확하고 간결한 답변을 제공하세요."
)

# 엑셀 데이터 로드 및 변환 함수
def load_excel_to_documents(file_path):
    df = pd.read_excel(file_path)
    documents = []
    for _, row in df.iterrows():
        text = " ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
        documents.append(text)
    return documents

# 벡터 데이터베이스 로드 또는 생성
def initialize_vector_db(file_path, persist_directory):
    if os.path.exists(persist_directory):
        print("[INFO] 기존 벡터 데이터베이스 로드 중...")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("[INFO] 새로운 벡터 데이터베이스 생성 중...")
        documents = load_excel_to_documents(file_path)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_text(documents)
        vector_store = Chroma.from_texts(split_docs, embeddings, persist_directory=persist_directory)
        vector_store.persist()
    return vector_store

# 벡터DB 초기화
file_path = "./db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx"
vector_store = initialize_vector_db(file_path, persist_directory)

# 문서 결합 체인 생성
combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)

# 최신 Retrieval Chain 방식 적용
retrieval_chain = create_retrieval_chain(
    retriever=vector_store.as_retriever(),
    combine_docs_chain=combine_docs_chain
)

# Flask 애플리케이션 초기화
app = Flask(__name__, template_folder="templates")

# HTML 페이지 서빙 라우트
@app.route('/')
def index():
    return render_template('index.html')

# 챗봇 엔드포인트
@app.route('/chatbot', methods=['POST'])
def chat_with_postman():
    data = request.get_json()
    user_message = data.get("message")
    start_time = time.time()
    try:
        # 최신 Retrieval Chain 방식 적용
        response = retrieval_chain.invoke({"question": user_message, "context": ""})["answer"]
    except Exception as e:
        response = f"[ERROR] {e}"

    end_time = time.time()
    print(f"[INFO] API 응답 시간: {end_time - start_time:.2f}초")

    return jsonify({"response": response})

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5050
    print(f"\n[INFO] Flask 서버 실행 중...\n접속 주소: http://{host}:{port}\n엔드포인트:\n - 메인 페이지: http://{host}:{port}/\n - 챗봇 엔드포인트: http://{host}:{port}/chatbot (POST 요청 필요)")
    app.run(host=host, port=port)
