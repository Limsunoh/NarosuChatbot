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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory


# ✅ 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
REDIS_URL = "redis://localhost:6379/0"


# ✅ FastAPI 인스턴스 생성
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5050"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# ✅ Redis 기반 메시지 기록 관리 함수
def get_message_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)

# ✅ FAISS 인덱스 파일 경로 설정
faiss_file_path = f"faiss_index_{datetime.now().strftime('%Y%m%d')}.faiss"

# ✅ POST 요청을 위한 데이터 모델 정의
class QueryRequest(BaseModel):
    query: str

# ✅ JSON 직렬화를 위한 int 변환 함수
def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
        return obj.item()
    return obj

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
    try:
        faiss.write_index(index, file_path)
    except Exception as e:
        print(f"❌ FAISS 인덱스 저장 오류: {e}")

# ✅ FAISS 인덱스 로드
def load_faiss_index(file_path):
    try:
        return faiss.read_index(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS 인덱스 로딩 오류: {str(e)}")

# ✅ FAISS 인덱스 생성 및 저장 (IndexIVFFlat 적용)
def create_and_save_faiss_index(file_path):
    try:
        texts, _ = load_excel_to_texts(file_path)
        임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
        embeddings = 임베딩.embed_documents(texts)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        # ✅ IndexIVFFlat 사용
        d = embeddings.shape[1]
        nlist = 200  # 클러스터 개수
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(embeddings)
        index.add(embeddings)

        save_faiss_index(index, faiss_file_path)
    except Exception as e:
        print(f"❌ FAISS 인덱스 생성 및 저장 오류: {e}")

# ✅ FAISS 인덱스 로드 또는 생성
if not os.path.exists(faiss_file_path):
    create_and_save_faiss_index("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")
index = load_faiss_index(faiss_file_path)

# ✅ 정확도 계산 함수
def calculate_accuracies(distances):
    """
    거리 값 리스트를 받아 정확도를 계산하여 반환합니다.
    :param distances: FAISS에서 반환된 거리 값 리스트
    :return: 정확도 리스트 (0.00 ~ 1.00 범위)
    """
    return [round((1 - dist), 2) for dist in distances]



# ✅ LLM을 이용한 키워드 추출 및 대화 이력 반영
def extract_keywords_with_llm(query):
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)

    # 기존 대화 이력과 함께 LLM에 전달
    response = llm.invoke([
        SystemMessage(content="사용자의 대화 내역을 반영하여 상품 검색을 위한 핵심 키워드를 추출해주세요."),
        HumanMessage(content=f"질문: {query} \n ")
    ])

    # 키워드 업데이트
    keywords = [keyword.strip() for keyword in response.content.split(",")]
    combined_keywords = ", ".join(keywords)
    return combined_keywords



# ✅ 루트 경로 - HTML 페이지 렌더링
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ✅ POST 요청 처리 - `/chatbot`
@app.post("/chatbot")
def search_and_generate_response(request: QueryRequest):
    query = request.query
    print(f"🔍 사용자 검색어: {query}")

    
    try:
        # ✅ LLM을 통한 키워드 추출 및 임베딩 생성
        combined_keywords = extract_keywords_with_llm(query)
        print(f"✅ 추출된 키워드: {combined_keywords}")

        _, data = load_excel_to_texts("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")
        
        # ✅ OpenAI 임베딩 생성
        임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
        query_embedding = 임베딩.embed_query(combined_keywords)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # ✅ FAISS 검색 수행
        D, I = index.search(query_embedding, k=5)

        # ✅ FAISS 검색 결과 검사
        if I is None or len(I) == 0 or not hasattr(I, "__iter__"):
            return {
                "query": query,
                "results": [],
                "message": "검색 결과가 없습니다. 다른 키워드를 입력하세요!"
            }
        
        # ✅ 거리 값 기반으로 정확도 계산
        accuracies = calculate_accuracies(D)

        # ✅ 검색 결과 JSON 변환
        results = []
        for idx, accuracy in zip(I, accuracies):
            if idx >= len(data):
                continue
            result_row = data.iloc[idx]
            result_info = {
                "상품코드": str(result_row["상품코드"]),
                "원본상품명": result_row["원본상품명"],
                "오너클랜판매가": convert_to_serializable(result_row["오너클랜판매가"]),
                "배송비": convert_to_serializable(result_row["배송비"]),
                "이미지중": result_row["이미지중"],
                "원산지": result_row["원산지"],
                "정확도": accuracy  # ✅ 정확도 추가
            }
            results.append(result_info)
        
        # ✅ ChatPromptTemplate 및 RunnableWithMessageHistory 생성
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 친절한 쇼핑 챗봇입니다. 사용자 대화를 기억하고 친절하게 회장님 모시듯 응답하세요. 그리고 상품을 찾을 수 있게 계속해서 질문을 하세요. 30자 이내로 응답하세요."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        runnable = prompt | llm

        with_message_history = RunnableWithMessageHistory(
            runnable,  # 실행 가능한 객체
            get_message_history,  # 메시지 기록을 가져오는 함수
            input_messages_key="input",  # 입력 메시지의 키
            history_messages_key="history",  # 기록 메시지의 키
        )

        # ✅ JSON 반환
        return {
            "query": query,
            "results": results,
            "정확도" : accuracy,
            "runnable" : runnable
        }

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ FastAPI 서버 실행 (포트 고정: 5050)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
