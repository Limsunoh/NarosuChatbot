from dotenv import load_dotenv
import os
import time
from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_chroma import Chroma
from langsmith import Client
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
import shutil

# ✅ 환경 변수 로드 및 LangSmith 연결
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
persist_directory = './chroma_db'

# ✅ LangSmith 연결 (API 키와 프로젝트 정보)
langsmith_client = Client(api_key=os.getenv('LANGCHAIN_API_KEY'))

# ✅ OpenAI 임베딩 및 LLM 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
llm = OpenAI(api_key=API_KEY, temperature=0.3, top_p=0.7)

# ✅ LangSmith 트레이싱 비활성화 (서버 연결 오류 방지)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# ✅ 사용자 정의 프롬프터 (LLM에 역할 부여)
PROMPT_JSON = """
당신은 상품 데이터를 분석하는 상품 추천 AI 챗봇 전문가입니다. 
- 사용자의 질문에 따라 적합한 상품을 추천합니다.
- 제공된 데이터의 모든 열을 반드시 확인하고, 검색을 수행합니다.
"""

# ✅ 핵심 열 자동 선택 (질문 기반)
def determine_core_columns(user_query):
    keyword_to_columns = {
        "가격": ["오너클랜판매가"],
        "배송비": ["배송비"],
        "색상": ["색상"],
        "카테고리": ["카테고리명"],
        "상품": ["원본상품명"],
        "키워드": ["키워드"]
    }
    core_columns = set()
    for keyword, columns in keyword_to_columns.items():
        if keyword.lower() in user_query.lower():
            core_columns.update(columns)
    if not core_columns:
        core_columns = {"원본상품명", "카테고리명", "키워드"}
    return list(core_columns)

# ✅ 엑셀 데이터 로드 및 전처리
def load_excel_to_documents(file_path):
    """엑셀 데이터를 로드하고, 텍스트 변환"""
    df = pd.read_excel(file_path, engine="openpyxl")  # ✅ openpyxl로 읽기 오류 해결
    core_columns = df.columns.tolist()
    documents = [
        " ".join([f"{col}: {row[col]}" for col in core_columns if pd.notna(row[col])])
        for _, row in df.iterrows()
    ]
    return df, documents, core_columns

# ✅ 벡터 데이터베이스 초기화 (토큰 초과 방지 적용)
def initialize_vector_db(file_path, persist_directory, reset=False):
    """ChromaDB 초기화 (토큰 초과 방지 적용)"""
    if reset and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("[INFO] 기존 벡터 데이터베이스 삭제 완료.")

    if os.path.exists(persist_directory):
        try:
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            print("[INFO] 기존 벡터 데이터베이스 로드 완료.")
        except Exception as e:
            print(f"[ERROR] 벡터DB 로드 오류: {e}")
            vector_store = None
    else:
        # ✅ 데이터 로드 및 전처리
        df, documents, core_columns = load_excel_to_documents(file_path)
        
        # ✅ 텍스트 청크 크기 감소 (토큰 초과 방지)
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        
        # ✅ 각 문서를 개별적으로 청크 나누기
        split_docs = []
        for doc in documents:
            split_docs.extend(text_splitter.split_text(doc))
        
        # ✅ 벡터DB 생성 (batch_size 설정)
        vector_store = Chroma.from_texts(
            split_docs, embeddings, persist_directory=persist_directory, batch_size=100
        )
        vector_store.persist()
        print("[INFO] 새 벡터 데이터베이스 생성 완료.")
    return vector_store

# ✅ 벡터DB 초기화 (데이터 재생성 가능)
file_path = "./db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx"
vector_store = initialize_vector_db(file_path, persist_directory, reset=False)

# ✅ Flask 애플리케이션 초기화
app = Flask(__name__, template_folder="templates")

# ✅ 메인 페이지 서빙 라우트
@app.route('/')
def index():
    """메인 페이지 렌더링"""
    return render_template('index.html')

# ✅ **LangSmith 트레이싱 포함 챗봇 엔드포인트 (유사도 점수 제거)**
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_query = data.get("message")
    start_time = time.time()

    try:
        # ✅ LangSmith 트레이싱 비활성화 적용
        core_columns = determine_core_columns(user_query)
        
        # ✅ 프롬프터 구성 및 LLM 호출
        full_prompt = f"{PROMPT_JSON}\n사용자 입력: {user_query}\n핵심 열: {core_columns}"
        llm.invoke(full_prompt)

        # ✅ RAG 기반 벡터DB 검색 수행 (유사도 점수 제거)
        search_results = vector_store.similarity_search(user_query, k=5)
        
        # ✅ 핵심 열만 필터링하여 결과 제공
        filtered_results = []
        for doc in search_results:
            filtered_data = {col: doc.page_content.split(f"{col}: ")[-1].split(" ")[0] for col in core_columns}
            filtered_results.append(filtered_data)

        # ✅ 유사도 점수 제거 후 상품만 반환
        response_data = filtered_results

    except Exception as e:
        response_data = {"error": str(e)}

    end_time = time.time()
    print(f"[INFO] 추천 상품 응답 시간: {end_time - start_time:.2f}초")
    return jsonify({"response": response_data})

# ✅ Flask 애플리케이션 실행 (포트 7070으로 변경)
if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5050  # 기존 포트 충돌 방지
    print(f"\n[INFO] Flask 서버 실행 중...\n접속 주소: http://{host}:{port}\n - 챗봇 엔드포인트: http://{host}:{port}/chatbot (POST 요청 필요)")
    app.run(host=host, port=port)
