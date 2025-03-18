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
from langchain_community.chat_message_histories import RedisChatMessageHistory


# ✅ FastAPI 인스턴스 생성
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5050"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
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

# ✅ LLM을 이용한 키워드 추출 및 대화 이력 반영
def extract_keywords_with_llm(query):
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)

    # ✅ 기본값으로 초기화
    chat_history = ""

    try:
        # ✅ 메모리 데이터를 로드하고 확인
        memory_data = memory.load_memory_variables({})
        if "chat_history" in memory_data and memory_data["chat_history"]:
            chat_history = str(memory_data["chat_history"])
    except Exception as e:
        print(f"메모리 로딩 오류: {e}")

    # 기존 대화 이력과 함께 LLM에 전달
    response = llm.invoke([
        SystemMessage(content="사용자의 대화 내역을 반영하여 상품 검색을 위한 핵심 키워드를 추출해주세요."),
        HumanMessage(content=f"질문: {query} \n 대화 이력: {chat_history}")
    ])
    # 키워드 업데이트
    keywords = [keyword.strip() for keyword in response.content.split(",")]
    memoryWord.clear()
    memoryWord.extend(keywords)
    memory.save_context({"input": query}, {"output": keywords})
    combined_keywords = ", ".join(keywords)
    return combined_keywords

# ✅ FAISS 인덱스 로드 또는 생성
if not os.path.exists(faiss_file_path):
    create_and_save_faiss_index("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")
index = load_faiss_index(faiss_file_path)

# ✅ 루트 경로 - HTML 페이지 렌더링
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ✅ 정확도 계산 함수
def calculate_accuracies(distances):
    """
    거리 값 리스트를 받아 정확도를 계산하여 반환합니다.
    :param distances: FAISS에서 반환된 거리 값 리스트
    :return: 정확도 리스트 (0.00 ~ 1.00 범위)
    """
    return [round((1 - dist), 2) for dist in distances]

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

        # ✅ 거리 임계값 필터링
        threshold = 0.5
        filtered_results = [(dist, int(idx)) for dist, idx in zip(D[0], I[0]) if dist <= threshold]

        # ✅ 필터링 후 결과 없음 방지
        if not filtered_results:
            print("⚠️ 임계값 기준 검색 결과 없음")
            return {
                "query": query,
                "results": [],
                "message": "검색 결과가 없습니다. 검색 기준을 완화하거나 다른 키워드를 입력해주세요."
            }

        # ✅ 결과 언패킹 방지
        try:
            D, I = zip(*filtered_results)
        except ValueError:
            print("⚠️ 검색 결과 없음")
            return {
                "query": query,
                "results": [],
                "message": "검색 결과가 없습니다. 다른 키워드를 입력해보세요!"
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

        # ✅ 검색 결과를 LangChain 메모리에 저장 (모든 속성 저장)
        memory.save_context(
            {"input": query},  # 사용자 질문
            {"output": results}  # 검색 결과 전체 저장
        )

        # ✅ LLM을 사용하여 대화 흐름을 자연스럽게 유지
        llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", api_key=API_KEY)
        response = llm.invoke([
            SystemMessage(content=f"오너클랜판매가:가격,원본상품명:상품의 제목. 당신은 쇼핑몰에 대한 지식이 높고 사용자의 질문에 대해서 아주 친절하고 전문가인 챗봇입니다. 모든 답변은 간결하고 쉽게 답변합니다."),
            HumanMessage(content=f"사용자 요청: {query} \n 검색된 결과:\n{json.dumps(results, ensure_ascii=False, indent=2)}\n 사용자에게 추가 질문을 만들어주세요."),
        ])

        # ✅ JSON 반환
        return {
            "query": query,
            "results": results,
            "llm_response": response.content,
            "chat_history": memory.load_memory_variables({}),
            "extracted_keywords": memoryWord,
            "정확도" : accuracy
        }

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ FastAPI 서버 실행 (포트 고정: 5050)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
