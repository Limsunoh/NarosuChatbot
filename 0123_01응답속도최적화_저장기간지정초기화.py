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
from langchain_core.runnables import ConfigurableFieldSpec
import redis
import requests
from typing import Union
import logging
import time



# ✅ 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
REDIS_URL = "redis://localhost:6379/0"
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
print(f"🔍 로드된 VERIFY_TOKEN: {VERIFY_TOKEN}")
print(f"🔍 로드된 PAGE_ACCESS_TOKEN: {PAGE_ACCESS_TOKEN}")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("response_time_logger")

# ✅ FAISS 인덱스 파일 경로 설정
faiss_file_path = f"faiss_index_01M.faiss"

def get_redis():
    return redis.Redis.from_url(REDIS_URL)

# ✅ FastAPI 인스턴스 생성
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5050", "https://satyr-inviting-quetzal.ngrok-free.app"],  # 외부 도메인 추가
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("response_time_logger")

# 응답 속도 측정을 위한 미들웨어 추가
@app.middleware("http")
async def measure_response_time(request: Request, call_next):
    start_time = time.time()  # 요청 시작 시간
    response = await call_next(request)  # 요청 처리
    process_time = time.time() - start_time  # 처리 시간 계산
    
    # '/chatbot' 엔드포인트에 대한 응답 속도 로깅
    if request.url.path == "/webhook":
        print(f"📊 [TEST] Endpoint: {request.url.path}, 처리 시간: {process_time:.4f} 초")  # print로 직접 확인
        logger.info(f"📊 [Endpoint: {request.url.path}] 처리 시간: {process_time:.4f} 초")
    
    response.headers["X-Process-Time"] = str(process_time)  # 응답 헤더에 처리 시간 추가
    return response

# ✅ Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# ✅ Redis 기반 메시지 기록 관리 함수
def get_message_history(session_id: str) -> RedisChatMessageHistory:
    """
    Redis를 사용하여 메시지 기록을 관리합니다.
    :param session_id: 사용자의 고유 세션 ID
    :return: RedisChatMessageHistory 객체
    """
    try:
        history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
        return history
    except Exception as e:
        print(f"❌ Redis 연결 오류: {e}")
        raise HTTPException(status_code=500, detail="Redis 연결에 문제가 발생했습니다.")

# 요청 모델
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

# ✅ LLM을 이용한 키워드 추출 및 대화 이력 반영
def extract_keywords_with_llm(query):
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)

    # 기존 대화 이력과 함께 LLM에 전달
    response = llm.invoke([
        SystemMessage(content="사용자의 대화 내역을 반영하여 상품 검색을 위한 정말로 핵심 키워드를 추출해주세요. 만약 단어 간에 띄어쓰기가 있다면 하나의 단어 일수도 있습니다 띄어쓰기가 있다면 단어끼리 붙여서도 문장을 분석해보세요요. 여러방법으로 생각해서 추출해주세요. 다른 나라 언어로 질문이 들어오면 질문을 먼저 한글로 번역해서 단어를 추출합니다."),
        HumanMessage(content=f"질문: {query} \n ")
    ])

    # 키워드 업데이트
    keywords = [keyword.strip() for keyword in response.content.split(",")]
    combined_keywords = ", ".join(keywords)
    return combined_keywords

store = {}  # 빈 딕셔너리를 초기화합니다.

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # 세션 ID에 해당하는 대화 기록이 저장소에 없으면 새로운 ChatMessageHistory를 생성합니다.
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    # 세션 ID에 해당하는 대화 기록을 반환합니다.
    return store[session_id]

def clear_message_history(session_id: str):
    """
    Redis에 저장된 특정 세션의 대화 기록을 초기화합니다.
    """
    try:
        history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
        history.clear()
        print(f"✅ 세션 {session_id}의 대화 기록이 초기화되었습니다.")
    except Exception as e:
        print(f"❌ Redis 초기화 오류: {e}")
        raise HTTPException(status_code=500, detail="대화 기록 초기화 중 오류가 발생했습니다.")






@app.get("/webhook")
async def verify_webhook(request: Request):
    try:
        
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        
        print(f"🔍 받은 Verify Token: {token}")
        print(f"🔍 서버 Verify Token: {VERIFY_TOKEN}")
        
        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("✅ 웹훅 인증 성공")
            return int(challenge)
        else:
            print("❌ 웹훅 인증 실패")
            return {"status": "error", "message": "Invalid token"}
    except Exception as e:
        print(f"❌ 인증 처리 오류: {e}")
        return {"status": "error", "message": str(e)}
    
@app.post("/webhook")
async def handle_webhook(request: Request):
    try:
        data = await request.json()
        # print(f"📬 웹훅 데이터 수신: {json.dumps(data, indent=2, ensure_ascii=False)}")
        # print(f"📬 웹훅 데이터 수신2: {data}")

        # 메시지 이벤트 처리
        if data.get("field") == "messages":  # field 값이 'messages'인지 확인
            value = data.get("value", {})  # value 필드 가져오기

            # Redis 세션 ID 설정
            sender_id = value.get("sender", {}).get("id")  # 발신자 ID
            # print(f"유저아이디 : {sender_id}")

            # 사용자 메시지 가져오기
            user_message = value.get("message", {}).get("text", "").strip()  # 메시지 텍스트
            # print(f"유저메세지 : {user_message}")

            if sender_id and user_message:
                # ✅ 챗봇 응답 생성
                bot_response = search_and_generate_response2(user_message, session_id=sender_id)

                # ✅ 사용자에게 응답 전송
                if isinstance(bot_response, dict) and "response" in bot_response:
                    response_text = bot_response["response"]
                else:
                    response_text = "죄송합니다. 요청 처리 중 오류가 발생했습니다."

                # # 페이스북 사용자에게 메시지 전송
                # send_message(sender_id, response_text)
                print(f"🤖 챗봇 응답: {response_text}")

        return response_text
    except Exception as e:
        print(f"❌ 웹훅 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))



def search_and_generate_response2(request: Union[QueryRequest, str], session_id: str = None) -> dict:
    query = request

    # Reset 요청 처리
    if query.lower() == "reset":
        if session_id:
            clear_message_history(session_id)
        return {"message": f"세션 {session_id}의 대화 기록이 초기화되었습니다."}

    print(f"🔍 사용자 검색어: {query}")

    try:
        # ✅ Redis 메시지 기록 관리
        session_history = get_message_history(session_id)
        # ✅ 기존 대화 내역 확인
        print(f"🔍 Redis 메시지 기록 (초기 상태): {session_history.messages}")

        # ✅ 기존 대화 내역 확인
        previous_queries = [
            msg.content for msg in session_history.messages if isinstance(msg, HumanMessage)
        ]
        print(f"🔍 Redis 메시지 기록: {previous_queries}")

        # ✅ LLM을 통한 키워드 추출 및 임베딩 생성
        combined_query = " ".join(previous_queries + [query])
        combined_keywords = extract_keywords_with_llm(combined_query)
        print(f"✅ 생성된 검색 키워드: {combined_keywords}")

        # ✅ Redis에 사용자 입력 추가
        session_history.add_message(HumanMessage(content=query))
        print(f"�� Redis 메시지 기록 (변경된 상태): {session_history.messages}")

        _, data = load_excel_to_texts("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")

        # ✅ OpenAI 임베딩 생성
        임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
        query_embedding = 임베딩.embed_query(combined_keywords)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # ✅ FAISS 검색 수행
        D, I = index.search(query_embedding, k=5)

        # ✅ FAISS 검색 결과 검사
        if I is None or I.size == 0:
            return {
                "query": query,
                "results": [],
                "message": "검색 결과가 없습니다. 다른 키워드를 입력하세요!",
                "message_history": [
                    {"type": type(msg).__name__, "content": msg.content if hasattr(msg, "content") else str(msg)}
                    for msg in session_history.messages
                ],
            }



        # ✅ 검색 결과 JSON 변환
        results = []
        for idx_list in I:  # 2차원 배열 처리
            for idx in idx_list:
                if idx >= len(data):  # 잘못된 인덱스 방지
                    continue
                result_row = data.iloc[idx]
                result_info = {
                    "상품코드": str(result_row["상품코드"]),
                    "제목": result_row["원본상품명"],
                    "가격": convert_to_serializable(result_row["오너클랜판매가"]),
                    "배송비": convert_to_serializable(result_row["배송비"]),
                    "이미지": result_row["이미지중"],
                    "원산지": result_row["원산지"]
                }
                results.append(result_info)

        # ✅ results를 텍스트로 변환
        if results:
            results_text = "\n".join(
                [
                    f"상품코드: {item['상품코드']}, 제목: {item['제목']}, 가격: {item['가격']}원, "
                    f"배송비: {item['배송비']}원, 원산지: {item['원산지']}, 이미지: {item['이미지']}"
                    for item in results
                ]
            )
        else:
            results_text = "검색 결과가 없습니다."
                
        message_history=[]
        
        # ✅ ChatPromptTemplate 및 RunnableWithMessageHistory 생성
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""항상 message_history의 대화이력을 보면서 대화의 문맥을 이해합니다. 당신은 쇼핑몰 챗봇으로, 친절하고 인간적인 대화를 통해 고객의 쇼핑 경험을 돕는 역할을 합니다. 아래는 최근 검색된 상품 목록입니다.
            목표: 사용자의 요구를 명확히 이해하고, 이전 대화의 맥락을 기억해 자연스럽게 이어지는 추천을 제공합니다.
            작동 방식:
            이전 대화 내용을 기반으로 적합한 상품을 연결합니다.
            이건 대화 이력 문장을 보고 문맥을 이해하며, 사용자가 무슨 내용을 작성하고 상품을 찾는지 집중적으로 답변을 작성합니다.
            스타일: 따뜻하고 공감하며, 마치 실제 쇼핑 도우미처럼 친절하고 자연스럽게 응답합니다.
            대화 전략:
            사용자가 원하는 상품을 구체화하기 위해 적절한 후속 질문을 합니다.
            대화의 흐름이 끊기지 않도록 부드럽게 이어갑니다.
            목표는 단순한 정보 제공이 아닌, 고객이 필요한 상품을 정확히 찾을 수 있도록 돕는 데 중점을 둡니다. 당신은 이를 통해 고객이 편안하고 만족스러운 쇼핑 경험을 누릴 수 있도록 최선을 다해야 합니다."""),
            MessagesPlaceholder(variable_name="message_history"),
            ("system", f"다음은 대화이력입니다 : \n{session_history.messages}"),
            ("system", f"다음은 상품결과입니다 : \n{results_text}"),
            ("human", query)
        ])
        
        runnable = prompt | llm

        with_message_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",  # 입력 메시지의 키
            history_messages_key="message_history",
        )

        

        # ✅ LLM 실행 및 메시지 기록 업데이트
        response = with_message_history.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )

        # ✅ Redis에 AI 응답 추가
        session_history.add_message(AIMessage(content=response.content))

        # ✅ 메시지 기록을 Redis에서 가져오기
        session_history = get_message_history(session_id)
        message_history = [
            {"type": type(msg).__name__, "content": msg.content if hasattr(msg, "content") else str(msg)}
            for msg in session_history.messages
        ]


        # ✅ 출력 디버깅
        print("*** Response:", response)
        print("*** Message History:", message_history)

        # ✅ JSON 반환
        return {
            "query": query,
            "results": results,
            "response": response.content,
            "message_history": message_history
        }

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


def generate_bot_response(user_message: str) -> str:
    """
    사용자의 메시지를 받아 챗봇 응답을 생성합니다.
    """
    try:
        # ✅ Redis를 이용한 세션 관리
        session_id = f"user_{user_message[:10]}"  # 간단한 세션 ID 생성 (필요 시 사용자 ID 사용)
        session_history = get_message_history(session_id)

        # ✅ Redis에서 기존 대화 이력 확인
        print(f"🔍 Redis 메시지 기록 (초기 상태): {session_history.messages}")

        # ✅ 사용자 입력을 기록에 추가
        session_history.add_message(HumanMessage(content=user_message))

        # ✅ LLM 기반 응답 생성
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "사용자 메시지에 따라 적절하고 친절한 응답을 생성하세요."),
            MessagesPlaceholder(variable_name="message_history"),
            ("human", user_message)
        ])
        runnable = prompt | llm
        response = runnable.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
        )

        # ✅ Redis에 챗봇 응답 저장
        session_history.add_message(AIMessage(content=response.content))

        return response.content
    except Exception as e:
        print(f"❌ 응답 생성 오류: {e}")
        return "죄송합니다. 오류가 발생했습니다. 나중에 다시 시도해주세요."

def send_message(recipient_id: str, message_text: str):
    """
    Facebook Send API를 사용하여 사용자에게 메시지를 전송합니다.
    """
    url = "https://graph.facebook.com/v22.0/me/messages"
    headers = {"Content-Type": "application/json"}
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text},
    }
    params = {"access_token": PAGE_ACCESS_TOKEN}
    response = requests.post(url, headers=headers, json=payload, params=params)
    
    if response.status_code == 200:
        print(f"✅ 메시지 전송 성공: {response.json()}")
    else:
        print(f"❌ 메시지 전송 실패: {response.status_code}, {response.text}")




# ✅ 루트 경로 - HTML 페이지 렌더링
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ✅ POST 요청 처리 - `/chatbot`
@app.post("/chatbot")
def search_and_generate_response(request: QueryRequest):
    query = request.query
    session_id = "redis123"  # 고정된 세션 ID


    reset_request = request.query.lower() == "reset"  # 'reset' 명령으로 초기화
    if reset_request:
        clear_message_history(session_id)
        return {
            "message": f"세션 {session_id}의 대화 기록이 초기화되었습니다."
        }



    print(f"🔍 사용자 검색어: {query}")

    try:
        # ✅ Redis 메시지 기록 관리
        session_history = get_message_history(session_id)
        # ✅ 기존 대화 내역 확인
        print(f"🔍 Redis 메시지 기록 (초기 상태): {session_history.messages}")

        # ✅ 기존 대화 내역 확인
        previous_queries = [
            msg.content for msg in session_history.messages if isinstance(msg, HumanMessage)
        ]
        print(f"🔍 Redis 메시지 기록: {previous_queries}")

        # ✅ LLM을 통한 키워드 추출 및 임베딩 생성
        combined_query = " ".join(previous_queries + [query])
        combined_keywords = extract_keywords_with_llm(combined_query)
        print(f"✅ 생성된 검색 키워드: {combined_keywords}")

        # ✅ Redis에 사용자 입력 추가
        session_history.add_message(HumanMessage(content=query))
        print(f"�� Redis 메시지 기록 (변경된 상태): {session_history.messages}")

        _, data = load_excel_to_texts("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")

        # ✅ OpenAI 임베딩 생성
        임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
        query_embedding = 임베딩.embed_query(combined_keywords)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # ✅ FAISS 검색 수행
        D, I = index.search(query_embedding, k=5)

        # ✅ FAISS 검색 결과 검사
        if I is None or I.size == 0:
            return {
                "query": query,
                "results": [],
                "message": "검색 결과가 없습니다. 다른 키워드를 입력하세요!",
                "message_history": [
                    {"type": type(msg).__name__, "content": msg.content if hasattr(msg, "content") else str(msg)}
                    for msg in session_history.messages
                ],
            }



        # ✅ 검색 결과 JSON 변환
        results = []
        for idx_list in I:  # 2차원 배열 처리
            for idx in idx_list:
                if idx >= len(data):  # 잘못된 인덱스 방지
                    continue
                result_row = data.iloc[idx]
                result_info = {
                    "상품코드": str(result_row["상품코드"]),
                    "제목": result_row["원본상품명"],
                    "가격": convert_to_serializable(result_row["오너클랜판매가"]),
                    "배송비": convert_to_serializable(result_row["배송비"]),
                    "이미지": result_row["이미지중"],
                    "원산지": result_row["원산지"]
                }
                results.append(result_info)

        # ✅ results를 텍스트로 변환
        if results:
            results_text = "\n".join(
                [
                    f"상품코드: {item['상품코드']}, 제목: {item['제목']}, 가격: {item['가격']}원, "
                    f"배송비: {item['배송비']}원, 원산지: {item['원산지']}, 이미지: {item['이미지']}"
                    for item in results
                ]
            )
        else:
            results_text = "검색 결과가 없습니다."
                
        message_history=[]
        
        # ✅ ChatPromptTemplate 및 RunnableWithMessageHistory 생성
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""항상 message_history의 대화이력을 보면서 대화의 문맥을 이해합니다. 당신은 쇼핑몰 챗봇으로, 친절하고 인간적인 대화를 통해 고객의 쇼핑 경험을 돕는 역할을 합니다. 아래는 최근 검색된 상품 목록입니다.
            목표: 사용자의 요구를 명확히 이해하고, 이전 대화의 맥락을 기억해 자연스럽게 이어지는 추천을 제공합니다.
            작동 방식:
            이전 대화 내용을 기반으로 적합한 상품을 연결합니다.
            이건 대화 이력 문장을 보고 문맥을 이해하며, 사용자가 무슨 내용을 작성하고 상품을 찾는지 집중적으로 답변을 작성합니다.
            스타일: 따뜻하고 공감하며, 마치 실제 쇼핑 도우미처럼 친절하고 자연스럽게 응답합니다.
            대화 전략:
            사용자가 원하는 상품을 구체화하기 위해 적절한 후속 질문을 합니다.
            대화의 흐름이 끊기지 않도록 부드럽게 이어갑니다.
            목표는 단순한 정보 제공이 아닌, 고객이 필요한 상품을 정확히 찾을 수 있도록 돕는 데 중점을 둡니다. 당신은 이를 통해 고객이 편안하고 만족스러운 쇼핑 경험을 누릴 수 있도록 최선을 다해야 합니다."""),
            MessagesPlaceholder(variable_name="message_history"),
            ("system", f"다음은 대화이력입니다 : \n{message_history}"),
            ("system", f"다음은 상품결과입니다 : \n{results_text}"),
            ("human", query)
        ])
        
        runnable = prompt | llm

        with_message_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",  # 입력 메시지의 키
            history_messages_key="message_history",
        )

        

        # ✅ LLM 실행 및 메시지 기록 업데이트
        response = with_message_history.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )

        # ✅ Redis에 AI 응답 추가
        session_history.add_message(AIMessage(content=response.content))

        # ✅ 메시지 기록을 Redis에서 가져오기
        session_history = get_message_history(session_id)
        message_history = [
            {"type": type(msg).__name__, "content": msg.content if hasattr(msg, "content") else str(msg)}
            for msg in session_history.messages
        ]


        # ✅ 출력 디버깅
        print("*** Response:", response)
        print("*** Message History:", message_history)

        # ✅ JSON 반환
        return {
            "query": query,
            "results": results,
            "response": response.content,
            "message_history": message_history
        }

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ✅ FastAPI 서버 실행 (포트 고정: 5050)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
