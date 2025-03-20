from dotenv import load_dotenv
import os
import pandas as pd
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import faiss
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

# ✅ FastAPI 인스턴스 생성
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5050"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 메모리 기능 추가 (대화 히스토리 및 핵심 단어 저장)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memoryWord = []  # 핵심 단어 저장

# ✅ Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# ✅ 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# ✅ FAISS 인덱스 파일 경로 설정
faiss_file_path = f"faiss_index_{datetime.now().strftime('%Y%m%d')}.faiss"

# ✅ POST 요청을 위한 데이터 모델 정의
class QueryRequest(BaseModel):
    query: str

# ✅ 엑셀 데이터 로드 및 변환 (공백 제거)
def load_excel_to_texts(file_path):
    try:
        data = pd.read_excel(file_path)
        data.columns = data.columns.str.strip()
        texts = [" | ".join([f"{col}: {row[col]}" for col in data.columns]) for _, row in data.iterrows()]
        return texts, data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"엑셀 파일 로드 오류: {str(e)}")

# ✅ FAISS 인덱스 저장
def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

# ✅ FAISS 인덱스 로드
def load_faiss_index(file_path):
    try:
        return faiss.read_index(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS 인덱스 로딩 오류: {str(e)}")

# ✅ FAISS 인덱스 생성 및 저장
def create_and_save_faiss_index(file_path):
    texts, _ = load_excel_to_texts(file_path)
    임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
    embeddings = 임베딩.embed_documents(texts)
    index = faiss.IndexFlatL2(1536)
    index.add(np.array(embeddings, dtype=np.float32))
    save_faiss_index(index, faiss_file_path)

# ✅ FAISS 인덱스 로드 또는 생성
if not os.path.exists(faiss_file_path):
    create_and_save_faiss_index("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")
index = load_faiss_index(faiss_file_path)

# ✅ JSON 직렬화를 위한 int 변환 함수
def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    return obj

# ✅ LLM 기반 핵심 단어 추출 함수
def extract_keywords_with_llm(query):
    global memoryWord
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", api_key=API_KEY)
    prompt = f"다음 문장에서 핵심 키워드를 추출해 주세요: {query}"
    response = llm.invoke([SystemMessage(content=prompt)])
    extracted_keywords = response.content.split(", ")
    memoryWord.extend(extracted_keywords)
    memoryWord = list(set(memoryWord))
    return extracted_keywords

# ✅ 루트 경로 - HTML 페이지 렌더링
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ✅ POST 요청 처리 - `/chatbot`
@app.post("/chatbot")
def search_and_generate_response(request: QueryRequest):
    query = request.query
    print(f"🔍 사용자 검색어: {query}")

    # ✅ 핵심 단어 추출 및 누적
    extract_keywords_with_llm(query)
    print(f"✅ 누적된 핵심 단어: {memoryWord}")

    _, data = load_excel_to_texts("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")

    # ✅ OpenAI 임베딩 생성
    임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
    query_embedding = 임베딩.embed_query(query)

    # ✅ FAISS 검색 수행
    # D: 유사도 거리 배열 (값이 작을수록 더 유사함) / I: 인덱스 배열 (가장 유사한 데이터의 위치)
    D, I = index.search(np.array([query_embedding], dtype=np.float32), k=5)

    # ✅ 검색 결과 JSON 변환
    results = []
    for idx in I[0]:
        if idx >= len(data):
            continue
        result_row = data.iloc[idx]
        result_info = {
            "상품코드": str(result_row["상품코드"]),
            "원본상품명": result_row["원본상품명"],
            "오너클랜판매가": convert_to_serializable(result_row["오너클랜판매가"]),
            "배송비": convert_to_serializable(result_row["배송비"]),
            "이미지중": result_row["이미지중"],
            "원산지": result_row["원산지"]
        }
        results.append(result_info)

    combined_input = f"질문: {query}, 핵심단어: {','.join(memoryWord)}"

    # ✅ LLM을 사용하여 대화 흐름을 자연스럽게 유지
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", api_key=API_KEY)
    response = llm.invoke([
        SystemMessage(content=f"당신은 쇼핑몰에 대한 지식이 높고 사용자의 질문에 대해서 아주 친절하고 전문가인 챗봇입니다. 모든 답변은 간결하고 쉽게 답변합니다. 상품의 옵션들을 계속 해서 질문을 해서 사용자가 찾고자 하는 상품을 좁혀 나갑니다. 모든 단어는 and형식으로 둘다 유사한 단어가 포함한 것을 검색 합니다."),
        HumanMessage(content=f"{combined_input}"),
        AIMessage(content=f"임베딩 파일 안에 속성중에 몇개 골라서 어떤걸 원하는지 다시 질문한다.")
    ])

    # ✅ 메모리에 저장
    memory.save_context({"input": query}, {"output": response.content})

    # ✅ JSON 반환
    return {
        "query": query,
        "results": results,
        "llm_response": response.content,
        "chat_history": memory.load_memory_variables({}),
        "extracted_keywords": memoryWord
    }

# ✅ FastAPI 서버 실행 (포트 고정: 5050)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
