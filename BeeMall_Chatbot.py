import asyncio
import base64
import json
import logging
import os
import re
import time
import urllib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union, List
from urllib.parse import quote
import math

import numpy as np
import pandas as pd
import redis
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
    RedisChatMessageHistory,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema,
    DataType, Collection
)

executor = ThreadPoolExecutor()

# ✅ 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
REDIS_URL = "redis://localhost:6379/0"
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
MANYCHAT_API_KEY = os.getenv('MANYCHAT_API_KEY')
key = os.getenv("MANYCHAT_API_KEY")
if isinstance(key, str) and "\x3a" in key:
    key = key.replace("\x3a", ":")



# API_URL = os.getenv("API_URL", "").rstrip("/")  # 예: http://114.110.135.96:8011
API_URL = "https://fb-narosu.duckdns.org"  # 예: http://114.110.135.96:8011
print(f"🔍 로드된 VERIFY_TOKEN: {VERIFY_TOKEN}")
print(f"🔍 로드된 PAGE_ACCESS_TOKEN: {PAGE_ACCESS_TOKEN}")
print(f"🔍 로드된 API_KEY: {API_KEY}")
print(f"🔍 로드된 API_URL: {API_URL}")

# # ✅ FAISS 인덱스 파일 경로 설정
# faiss_file_path = f"04_28_faiss_3s.faiss"

# ─── Milvus import & 연결 ───────────────────────────────────────────────
# 올바른 공인 IP와 포트
connections.connect(
    alias="default",
    host="114.110.135.96",
    port="19530"
)
print("✅ Milvus에 연결되었습니다.")

# 컬렉션 이름
collection_name = "ownerclan_weekly_0428"

# 컬렉션 객체 생성 (조회 용도)
collection = Collection(name=collection_name)

# OpenAI Embedding 모델 (쿼리용)
emb_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")

# 💡 저장된 벡터 수 확인
)
print(f"\n📊 저장된 엔트리 수: {collection.num_entities}")
# ────────────────────────────────────────────────────────────────────────


EMBEDDING_MODEL = "text-embedding-3-small"

def get_redis():
    return redis.Redis.from_url(REDIS_URL)

# ✅ FastAPI 인스턴스 생성
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[API_URL,  # 실제 배포 URL
                  "http://localhost:5050",
                   "https://satyr-inviting-quetzal.ngrok-free.app", 
                   "https://viable-shark-faithful.ngrok-free.app"],  # 외부 도메인 추가
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("response_time_logger")
print(f"🔐 API KEY: {MANYCHAT_API_KEY}")

# 응답 속도 측정을 위한 미들웨어 추가
@app.middleware("http")
async def measure_response_time(request: Request, call_next):
    start_time = time.time()  # 요청 시작 시간
    response = await call_next(request)  # 요청 처리
    process_time = time.time() - start_time  # 처리 시간 계산

    response.headers["ngrok-skip-browser-warning"] = "1"
    response.headers["X-Frame-Options"] = "ALLOWALL"  # 또는 제거 방식도 가능 #BeeMall 챗봇 Iframe 막히는것 때문에 헤더 추가가
    response.headers["Content-Security-Policy"] = "frame-ancestors *" #BeeMall 챗봇 Iframe 막히는것 때문에 헤더 추가가

    # '/chatbot' 엔드포인트에 대한 응답 속도 로깅
    if request.url.path == "/webhook":
        print(f"📊 [TEST] Endpoint: {request.url.path}, 처리 시간: {process_time:.4f} 초")  # print로 직접 확인
        logger.info(f"📊 [Endpoint: {request.url.path}] 처리 시간: {process_time:.4f} 초")
    
    response.headers["X-Process-Time"] = str(process_time)  # 응답 헤더에 처리 시간 추가
    return response

# ✅ Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

'''# ✅ Redis 기반 메시지 기록 관리 함수
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
        raise HTTPException(status_code=500, detail="Redis 연결에 문제가 발생했습니다.")'''

# 요청 모델
class QueryRequest(BaseModel):
    query: str


# ✅ JSON 직렬화를 위한 int 변환 함수
def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
        return obj.item()
    return obj


def minimal_clean_with_llm(latest_input: str, previous_inputs: List[str]) -> str:
    """
    최신 입력과 Redis에서 가져온 과거 입력을 함께 LLM에게 전달하여,
    최소한의 정제 + 충돌 문맥 제거를 수행한 한 문장 반환
    """
    try:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("❌ [ERROR] OPENAI_API_KEY가 설정되지 않았습니다.")
        API_KEY = os.environ["OPENAI_API_KEY"]

        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)

        context_message = "\n".join(previous_inputs)

        system_prompt = """
            당신은 사용자의 과거 대화 맥락과 최신 입력을 기반으로 의미 있는 문장을 재구성하는 전문가입니다.\n
            다음 기준을 철저히 따르세요:\n
            1. 이전 입력 중 **최신 입력과 의미가 충돌하는 문장**은 완전히 제거합니다.\n
            2. **충돌이 없는 이전 입력은 유지**하며, **최신 입력을 반영**해 전체 흐름을 자연스럽게 이어가세요.\n
            3. 문장의 단어 순서나 표현은 원문을 최대한 유지합니다.\n
            4. 오타, 띄어쓰기, 맞춤법만 교정하세요.\n
            5. 어떤 언어로 입력되었든 **결과는 한국어 한 문장**으로 출력하세요.\n
            6. 절대로 결과에 설명을 추가하지 마세요. **한 문장만 출력**합니다.\n
            \n
            ---\n
            \n
            # 예시 1:\n
            이전 입력:\n
            - 강아지 옷 찿아줘\n
            - 밝은색 으로다시찾아\n
            - 겨울 용이면 더조아\n
            \n
            최신 입력:\n
            - 여름용으로 바꿔줘\n
            \n
            → 결과: "강아지 옷 여름용 밝은 색으로 찾아줘"\n
            \n
            ---\n
            \n
            # 예시 2:\n
            이전 입력:\n
            - 아이폰보여줘\n
            - 프로 모델 이면 좋겠 어\n
            - 실버 색상으로 봐줘\n
            \n
            최신 입력:\n
            - 갤럭시로 바꿔줘\n
            \n
            → 결과: "갤럭시 실버 색상으로 보여줘"\n
            \n
            ---\n
            \n
            # 예시 3:\n
            이전 입력:\n
            - 운동화250mm사이즈찿아줘\n
            - 흰 색 계열이 좋아\n
            - 쿠션감있는거 위주로\n
            \n
            최신 입력:\n
            - 260mm로 바꿔줘\n
            \n
            → 결과: "운동화 260mm 흰색 쿠션감 있는 걸로 찾아줘"\n
            """

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"이전 대화: {context_message}\n최신 입력: {latest_input}")
        ])

        if not hasattr(response, "content") or not isinstance(response.content, str):
            raise ValueError("❌ LLM 응답이 유효하지 않습니다.")

        return response.content.strip()

    except Exception as e:
        print(f"❌ [ERROR] minimal_clean_with_llm 실패: {e}")
        return latest_input  # 실패 시 최신 입력만 사용


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    
    return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)

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


# 🔥 상품 캐시 (전역 선언)
PRODUCT_CACHE = {}
# 🔗 구매하기 버튼 클릭 시 호출되는 ManyChat용 Hook 주소
MANYCHAT_HOOK_BASE_URL = f"{API_URL}/product-select"


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
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    start_time = time.time()

    try:
        # ✅ Step 1: 요청 데이터 파싱
        data = await request.json()
        parse_time = time.time() - start_time
        logger.info(f"📊 [Parse Time]: {parse_time:.4f} 초")

        # ✅ Step 2: 메시지 처리 시작
        process_start = time.time()

        if data.get("field") == "messages":
            value = data.get("value", {})

            sender_id = value.get("sender", {}).get("id")
            user_message = value.get("message", {}).get("text", "").strip()
            postback = value.get("postback", {})

            # ✅ postback 처리
            postback_payload = postback.get("payload")
            if postback_payload and postback_payload.startswith("BUY::"):
                product_code = postback_payload.split("::")[1]
                background_tasks.add_task(handle_product_selection, sender_id, product_code)
                return {
                    "version": "v2",
                    "content": {
                        "messages": [
                            {"type": "text", "text": f"✅ 상품 {product_code} 정보가 전송되었습니다!"}
                        ]
                    }
                }

            # ✅ reset 처리
            if sender_id and user_message:
                if user_message.lower() == "reset":
                    print(f"🔄 [RESET] 세션 {sender_id}의 대화 기록 초기화!")
                    clear_message_history(sender_id)
                    return {
                        "version": "v2",
                        "content": {
                            "messages": [
                                {
                                    "type": "text",
                                    "text": f"🔄 Chat reset complete!\n💬 Enter a keyword and let the AI work its magic 🛍️."
                                }
                            ]
                        },
                        "message": f"세션 {sender_id}의 대화 기록이 초기화되었습니다."
                    }

                # ✅ 일반 메시지 → AI 응답 처리
                background_tasks.add_task(process_ai_response, sender_id, user_message)

            process_time = time.time() - process_start
            logger.info(f"📊 [Processing Time 전체]: {process_time:.4f} 초")

        # 기본 응답
        return {
            "version": "v2",
            "content": {
                "messages": [
                    {
                        "type": "text",
                        "text": "🛍️ Just a moment, smart picks coming soon! ⏳"
                    }
                ]
            }
        }

    except Exception as e:
        print(f"❌ 웹훅 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# 🔁 추천 응답 처리 함수
async def process_ai_response(sender_id: str, user_message: str):
    try:
        print(f"🕒 [AI 처리 시작] 유저 ID: {sender_id}, 메시지: {user_message}")

        # ✅ 외부 응답 생성 (동기 → 비동기 실행)
        loop = asyncio.get_running_loop()
        bot_response = await loop.run_in_executor(executor, external_search_and_generate_response, user_message, sender_id)

        # ✅ 응답 확인 및 메시지 준비
        if isinstance(bot_response, dict):
            combined_message_text = bot_response.get("combined_message_text", "")
            results = bot_response.get("results", [])

            # ✅ 상품 캐시에 저장 (product_code → 상품 딕셔너리 전체 저장)
            for product in results:
                product_code = product.get("상품코드")
                if product_code:
                    PRODUCT_CACHE[product_code] = product

            messages_data = []

            # ✅ AI 응답 메시지 먼저 추가
            if combined_message_text:
                messages_data.append({
                    "type": "text",
                    "text": combined_message_text
                })

            # ✅ 카드형 메시지를 하나로 묶기 위한 elements 리스트
            cards_elements = []

            for product in results:
                product_code = product.get("상품코드", "None")

                # 가격과 배송비 정수 변환 후 포맷팅
                try:
                    price = int(float(product.get("가격", 0)))
                except:
                    price = 0
                try:
                    shipping = int(float(product.get("배송비", 0)))
                except:
                    shipping = 0

                cards_elements.append({
                    "title": f"✨ {product['제목']}",
                    "subtitle": (
                        f"가격: {price:,}원\n"
                        f"배송비: {shipping:,}원\n"
                        f"원산지: {product.get('원산지', '')}"
                    ),
                    "image_url": product.get("이미지", ""),
                    "buttons": [
                        {
                            "type": "url",
                            "caption": "🤩 View Product 🧾",
                            "url": product.get("상품링크", "#")
                        },
                        {
                            "type": "dynamic_block_callback",
                            "caption": "🛍️ Buy Now 💰",
                            "url": f"{API_URL}/product-select",
                            "method": "post",
                            "payload": {
                                "product_code": product_code,
                                "sender_id": sender_id
                            }
                        }
                    ]
                })

            # ✅ 전체 카드 메시지로 추가
            messages_data.append({
                "type": "cards",
                "image_aspect_ratio": "horizontal",  # 또는 "square"
                "elements": cards_elements
})

            # ✅ 메시지 전송
            send_message(sender_id, messages_data)
            print(f"✅ [Combined 메시지 전송 완료]: {combined_message_text}")
            print(f"버튼 생성용 product_code: {product_code}")
            print("✅ 최종 messages_data:", json.dumps(messages_data, indent=2, ensure_ascii=False))

        else:
            print(f"❌ AI 응답 오류 발생")

    except Exception as e:
        print(f"❌ AI 응답 처리 오류: {e}")

def clean_html_content(html_raw: str) -> str:
    try:
        html_cleaned = html_raw.replace('\n', '').replace('\r', '')
        html_cleaned = html_cleaned.replace(""", "\"").replace(""", "\"").replace("'", "'").replace("'", "'")
        if html_cleaned.count("<center>") > html_cleaned.count("</center>"):
            html_cleaned += "</center>"
        if html_cleaned.count("<p") > html_cleaned.count("</p>"):
            html_cleaned += "</p>"
        return html_cleaned
    except Exception as e:
        print(f"❌ HTML 정제 오류: {e}")
        return html_raw


'''####################################################################################################################
external_search_and_generate_response는 ManyChat 같은 외부 서비스와 연동되는 챗봇용 API이고, 구축된 UI 에는 사용되지 않음.
'''

# ✅ 외부 검색 및 응답 생성 함수
def external_search_and_generate_response(request: Union[QueryRequest, str], session_id: str = None) -> dict:
    try:
        # ✅ 입력 쿼리 추출 및 타입 확인
        query = request if isinstance(request, str) else request.query
        print(f"🔍 사용자 검색어: {query}")

        if not isinstance(query, str):
            raise TypeError(f"❌ [ERROR] 잘못된 query 타입: {type(query)}")

        # ✅ 세션 초기화 명령 처리
        if query.lower() == "reset":
            if session_id:
                clear_message_history(session_id)
            return {"message": f"세션 {session_id}의 대화 기록이 초기화되었습니다."}

        # ✅ Redis 세션 기록 불러오기 및 최신 입력 저장
        session_history = get_session_history(session_id)
        session_history.add_user_message(query)

        previous_queries = [msg.content for msg in session_history.messages if isinstance(msg, HumanMessage)]
        if query in previous_queries:
            previous_queries.remove(query)
        
        # ✅ 전체 중복 제거 (최신 입력을 제외한 나머지에서)
        previous_queries = list(dict.fromkeys(previous_queries))

        # ✅ LLM으로 정제된 쿼리 생성
        UserMessage = minimal_clean_with_llm(query, previous_queries)
        print("\n🧾 [최종 정제된 문장] →", UserMessage)
        print("📚 [원본 전체 문맥] →", " | ".join(previous_queries + [query]))

        # ✅ 임베딩 벡터 생성
        q_vec = np.array([emb_model.embed_query(UserMessage)], dtype=np.float32).tolist()

        # ✅ Milvus 벡터 검색 수행
        milvus_results = collection.search(
            data=q_vec,
            anns_field="emb",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=5,
            output_fields=[
                "product_code", "market_product_name", "market_price",
                "shipping_fee", "image_url", "description",
                "origin", "max_quantity"
            ]
        )

        # ✅ Milvus 검색 결과 가공
        results = []
        for hits in milvus_results:
            for hit in hits:
                try:
                    e = hit.entity

                    # ▶ 본문 → 미리보기 링크 생성
                    html_raw = e.get("description", "") or ""
                    html_cleaned = clean_html_content(html_raw)
                    if isinstance(html_raw, bytes):
                        html_raw = html_raw.decode("cp949")
                    encoded_html = base64.b64encode(html_cleaned.encode("utf-8", errors="ignore")).decode("utf-8")
                    safe_html = urllib.parse.quote_plus(encoded_html)
                    preview_url = f"{API_URL}/preview?html={safe_html}"
                except Exception as err:
                    print(f"⚠️ 본문 처리 중 오류: {err}")
                    preview_url = "https://naver.com"

                # ▶ 상품링크 결정
                product_link = e.get("product_link", "")
                if not product_link or product_link in ["링크 없음", "#", None]:
                    product_link = preview_url

                # ▶ 옵션 정보 파싱
                option_raw = str(e.get("composite_options", "")).strip()
                option_display = "없음"
                if option_raw.lower() not in ["", "nan"]:
                    parsed = []
                    for line in option_raw.splitlines():
                        try:
                            name, extra, _ = line.split(",")
                            extra = int(float(extra))
                            parsed.append(f"{name.strip()} {f'(＋{extra:,}원)' if extra>0 else ''}".strip())
                        except Exception:
                            parsed.append(line.strip())
                    option_display = "\n".join(parsed)

                # ▶ 결과 정리
                result_info = {
                    "상품코드":     str(e.get("product_code", "없음")),
                    "제목":         e.get("market_product_name", "제목 없음"),
                    "가격":         convert_to_serializable(e.get("market_price", 0)),
                    "배송비":       convert_to_serializable(e.get("shipping_fee", 0)),
                    "이미지":       e.get("image_url", "이미지 없음"),
                    "원산지":       e.get("origin", "정보 없음"),
                    "상품링크":     product_link,
                    "옵션":         option_display,
                    "조합형옵션":   option_raw,
                    "최대구매수량": convert_to_serializable(e.get("max_quantity", 0))
                }
                results.append(result_info)
                PRODUCT_CACHE[result_info["상품코드"]] = result_info


        message_history = [
            {"type": type(msg).__name__, "content": msg.content if hasattr(msg, "content") else str(msg)}
            for msg in session_history.messages
        ]

        raw_results_json = json.dumps(results[:5], ensure_ascii=False)
        raw_history_json = json.dumps(message_history, ensure_ascii=False)
        escaped_results = raw_results_json.replace("{", "{{").replace("}", "}}")
        escaped_history = raw_history_json.replace("{", "{{").replace("}", "}}")

        # ✅ LangChain 기반 프롬프트 및 LLM 실행 설정
        API_KEY = os.environ.get("OPENAI_API_KEY")
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
        당신은 쇼핑몰 챗봇으로, 친절하고 인간적인 대화를 통해 고객의 쇼핑 경험을 돕습니다.
        사용자의 언어에 맞게 번역해서 답변하세요(예시: 한국어->한국어, 영어->영어, 베트남어->베트남어 등)

        목표:
        - 사용자의 요구를 이해하고 대화의 맥락을 반영하여 적합한 상품을 추천합니다.

        작동 방식:
        - 대화 이력을 참고해 문맥을 파악하고 사용자의 요청에 맞는 상품을 연결합니다.
        - 필요한 경우 후속 질문으로 사용자의 요구를 구체화합니다.

        주의사항:
        - 아래 검색 결과는 LLM 내부 참고용입니다.
        - 상품을 나열하거나 직접 출력하지 마세요.
        - 키워드 요약이나 후속 질문을 위한 참고용으로만 활용하세요.
        """),
            MessagesPlaceholder(variable_name="message_history"),
            ("system", f"[검색 결과 - 내부 참고용 JSON]\n{escaped_results}"),
            ("system", f"[이전 대화 내용]\n{escaped_history}"),
            ("human", query)
        ])

        runnable = prompt | llm
        with_message_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="message_history",
        )

        # ✅ 응답 생성 및 시간 측정
        start_response = time.time()
        response = with_message_history.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"📊 [LLM 응답 시간] {time.time() - start_response:.2f}초")
        print("🤖 응답 결과:", response.content)

        # ✅ 최종 결과 반환 및 출력 로그
        result_payload = {
            "query": query,  # 사용자가 입력한 원본 쿼리
            "UserMessage": UserMessage,  # 정제된 쿼리
            "RawContext": previous_queries + [query],  # 전체 대화 맥락
            "results": results,  # 검색 결과 리스트
            "combined_message_text": response.content,  # LLM이 생성한 자연어 응답
            "message_history": message_history  # 전체 메시지 기록 (디버깅용)
        }
        print("\n📦 반환 객체 요약")
        print("query:", result_payload["query"])
        print("UserMessage:", result_payload["UserMessage"])
        print("RawContext:", result_payload["RawContext"])
        print("combined_message_text:", result_payload["combined_message_text"])
        print("results (count):", len(result_payload["results"]))
        print("message_history (count):", len(result_payload["message_history"]))

        return result_payload

    except Exception as e:
        print(f"❌ external_search_and_generate_response 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def send_message(sender_id: str, messages: list):  
    try:  
        url = "https://api.manychat.com/fb/sending/sendContent"
        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json"
        }

        # ✅ 메시지 구조 확인
        if not isinstance(messages, list):
            print(f"❌ [ERROR] messages는 리스트여야 합니다. 전달된 타입: {type(messages)}")
            return

        # ✅ LLM 응답 (첫 번째 메시지) 전송
        if messages:
            llm_text = messages[0]
            data = {
                "subscriber_id": sender_id,
                "data": {
                    "version": "v2",
                    "content": {
                        "messages": [llm_text],
                        "actions": [],
                        "quick_replies": []
                    }
                },
                "message_tag": "ACCOUNT_UPDATE"
            }
            response = requests.post(url, headers=headers, json=data)
            print(f"✅ [LLM 메시지 전송]: {response.json()}")

        # ✅ 카드 묶음 메시지 전송
        if len(messages) > 1:
            card_block = messages[1]
            data = {
                "subscriber_id": sender_id,
                "data": {
                    "version": "v2",
                    "content": {
                        "messages": [card_block],
                        "actions": [],
                        "quick_replies": []
                    }
                },
                "message_tag": "ACCOUNT_UPDATE"
            }

            response = requests.post(url, headers=headers, json=data)
            print(f"✅ [카드 메시지 전송]: {response.json()}")

    except Exception as e:
        print(f"❌ ManyChat 메시지 전송 오류: {e}")

class ManychatFieldUpdater:
    BASE_URL = "https://api.manychat.com/fb/subscriber/setCustomField"
    
    def __init__(self, subscriber_id: str, api_key: str):
        self.subscriber_id = subscriber_id
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def set_field(self, field_id: str, value):
        data = {
            "subscriber_id": self.subscriber_id,
            "field_id": field_id,
            "field_value": value
        }
        response = requests.post(self.BASE_URL, headers=self.headers, json=data)
        if response.status_code == 200:
            print(f"✅ {field_id} 저장 성공: {value}")
        else:
            print(f"❌ {field_id} 저장 실패: {response.status_code}, {response.text}")

    def set_unique_code(self, field_id: str, code: str):
        self.set_field(field_id, code)

    def set_product_name(self, field_id: str, name: str):
        self.set_field(field_id, name)

    def set_option(self, field_id: str, option: str):
        self.set_field(field_id, option)

    def set_price(self, field_id: str, price: int):
        self.set_field(field_id, price)

    def set_shipping(self, field_id: str, shipping: int):
        self.set_field(field_id, shipping)
    
    def set_product_selection_option(self, field_id: str, option: str):
        self.set_field(field_id, option)
    
    def set_extra_price(self, field_id: str, extra_price: int):
        self.set_field(field_id, extra_price)
    
    def set_product_max_quantity(self, field_id: str, max_quantity: int):
        self.set_field(field_id, max_quantity)
        
    def set_quantity(self, field_id: str, quantity: int):
        self.set_field(field_id, quantity)

    def set_total_price(self, field_id: str, total_price: int):
        self.set_field(field_id, total_price)


class Product_Selections(BaseModel):
    sender_id: str
    product_code: str


@app.post("/product-select")
def handle_product_selection(data: Product_Selections):
    try:
        sender_id = data.sender_id
        product_code = data.product_code

        if not sender_id or not product_code:
            return {
                "version": "v2",
                "content": {
                    "messages": [{"type": "text", "text": "❌ sender_id 또는 product_code가 없습니다."}]
                }
            }

        product = PRODUCT_CACHE.get(product_code)
        if not product:
            return {
                "version": "v2",
                "content": {
                    "messages": [{"type": "text", "text": f"❌ 상품코드 {product_code}에 대한 정보를 찾을 수 없습니다."}]
                }
            }
        
        # 가격, 옵션 정리
        price = int(float(product.get("가격", 0) or 0))
        shipping = int(float(product.get("배송비", 0) or 0))
        option_raw = product.get("조합형옵션", "").strip()

        option_display = "없음"
        if option_raw and option_raw.lower() != "nan":
            option_lines = option_raw.splitlines()
            parsed_options = []
            for line in option_lines:
                try:
                    name, extra_price, _ = line.split(",")
                    extra_price = int(float(extra_price))
                    price_str = f"(+{extra_price:,}원)" if extra_price > 0 else ""
                    parsed_options.append(f"{name.strip()} {price_str}".strip())
                except Exception:
                    parsed_options.append(line.strip())
            option_display = "\n".join(parsed_options)
        
        product["sender_id"] = sender_id
        
        # ✅ Manychat Field 업데이트
        updater = ManychatFieldUpdater(sender_id, MANYCHAT_API_KEY)
        updater.set_unique_code("12886380", product.get('상품코드'))
        updater.set_product_name("12886273", product.get('제목'))
        updater.set_option("12886363", option_display)
        updater.set_price("12890668", price)
        updater.set_shipping("12890670", shipping)
        updater.set_product_max_quantity("12922068", product.get('최대구매수량'))

        # ✅ 외부 Flow 트리거 (비동기처럼 요청 보내기)
        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        flow_payload = {
            "subscriber_id": sender_id,
            "flow_ns": "content20250417015933_369132"
        }
        try:
            res = requests.post(
                "https://api.manychat.com/fb/sending/sendFlow",
                headers=headers,
                json=flow_payload,
                timeout=5  # 실패해도 바로 리턴 안 끌려가게
            )
            print("✅ ManyChat Flow 전송 결과:", res.json())
        except Exception as e:
            print(f"❌ Flow 전송 실패: {e}")

        # ✅ 최종 클라이언트 응답 (Manychat Dynamic Block 규격)
        info_message = (
            f"상품코드\n{product.get('상품코드', '없음')}\n"
            f"제목\n{product.get('제목', '없음')}\n"
            f"원산지\n{product.get('원산지', '없음')}\n"
            f"------------------------------------------\n"
            f"가격\n{price:,}원\n"
            f"배송비\n{shipping:,}원\n"
            f"묶음배송수량\n{product.get('최대구매수량','0')}개\n"
            f"------------------------------------------\n"
            f"옵션\n{option_display}\n"
            f"------------------------------------------"
        ).strip()

        return {
            "version": "v2",
            "content": {
                "messages": [
                    {
                        "type": "text",
                        "text": info_message
                    }
                ]
            }
        }

    except Exception as e:
        print(f"❌ 상품 선택 처리 오류: {e}")
        return {
            "version": "v2",
            "content": {
                "messages": [{"type": "text", "text": f"❌ 서버 오류 발생: {str(e)}"}]
            }
        }



class Option_Selections(BaseModel):
    version: str
    field: str
    value: dict
    page: Optional[int] = 1


@app.post("/manychat-option-request")
def handle_option_request(data: Option_Selections):
    sender_id = data.value.get("sender_id") if isinstance(data.value, dict) else None
    product_code = data.value.get("product_code") if isinstance(data.value, dict) else None
    page = data.page or 1

    if not sender_id or not product_code:
        return {
            "version": "v2",
            "content": {
                "messages": [{"type": "text", "text": "❌ sender_id 또는 product_code가 없습니다."}]
            }
        }

    product = PRODUCT_CACHE.get(product_code)
    if not product:
        return {
            "version": "v2",
            "content": {
                "messages": [{"type": "text", "text": "❌ 상품 정보를 찾을 수 없습니다."}]
            }
        }

    options_raw = product.get("조합형옵션", "")
    if not options_raw or options_raw.lower() in ["nan", ""]:
        # ✅ 단일 옵션 상품일 경우 바로 다음 플로우로 이동
        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        flow_payload = {
            "subscriber_id": sender_id,
            "flow_ns": "content20250424050612_308842"
        }
        res = requests.post(
            "https://api.manychat.com/fb/sending/sendFlow",
            headers=headers,
            json=flow_payload
        )
        print("✅ 단일 옵션 상품 - Flow 전송 결과:", res.json())

        return {
            "version": "v2",
            "content": {
                "messages": [{"type": "text", "text": "🧾 This item has a single option — please select the quantity."}]
            }
        }

    options = options_raw.strip().split("\n")
    start_idx = (page - 1) * 27
    end_idx = start_idx + 27
    paged_options = options[start_idx:end_idx]

    message_batches = []
    current_buttons = []

    for opt in paged_options:
        try:
            name, extra_price, stock = opt.split(",")
            caption = f"{name.strip()} (+{int(float(extra_price)):,}원)" if float(extra_price) > 0 else name.strip()

            current_buttons.append({
                "type": "dynamic_block_callback",
                "caption": caption,
                "url": f"{API_URL}/manychat-option-select",
                "method": "post",
                "headers": {
                    "Content-Type": "application/json"
                    },
                "payload": {
                    "sender_id": sender_id,
                    "selected_option": caption
                }
            })

            if len(current_buttons) == 3:
                message_batches.append({
                    "type": "text",
                    "text": "📌 Pick your preferred option:",
                    "buttons": current_buttons
                })
                current_buttons = []

        except Exception as e:
            print(f"⚠️ 옵션 파싱 실패: {opt} → {e}")
            continue

    if current_buttons:
        message_batches.append({
            "type": "text",
            "text": "📌 Pick your preferred option:",
            "buttons": current_buttons
        })

    # 다음 페이지 버튼 추가
    if end_idx < len(options):
        message_batches.append({
            "type": "text",
            "text": "👀 View Next Option 🧾",
            "buttons": [
                {
                    "type": "dynamic_block_callback",
                    "caption": "👀 View Next Option 🧾",
                    "url": f"{API_URL}/manychat-option-request",
                    "method": "post",
                    "headers": {
                        "Content-Type": "application/json"
                        },
                    "payload": {
                        "version": "v2",
                        "field": "messages",
                        "value": {
                            "sender_id": sender_id,
                            "product_code": product_code
                        },
                        "page": page + 1
                    }
                }
            ]
        })

    return {
        "version": "v2",
        "content": {
            "messages": message_batches
        }
    }


@app.post("/manychat-option-select")
def handle_option_selection(payload: dict):
    sender_id = payload.get("sender_id")
    selected_option = payload.get("selected_option")

    if not sender_id or not selected_option:
        return {
            "version": "v2",
            "content": {
                "messages": [{"type": "text", "text": "❌ sender_id 또는 selected_option이 없습니다."}]
            }
        }

    # ✅ 추가금액 추출
    extra_price = 0
    match = re.search(r'\(\+([\d,]+)원\)', selected_option)
    if match:
        try:
            extra_price = int(match.group(1).replace(",", ""))
        except:
            extra_price = 0

    updater = ManychatFieldUpdater(sender_id, MANYCHAT_API_KEY)
    updater.set_product_selection_option("12904981", selected_option)
    updater.set_extra_price("12911810", extra_price)

    # ✅ 옵션 저장 후 Flow로 이동시키기
    headers = {
        "Authorization": f"Bearer {MANYCHAT_API_KEY}",
        "Content-Type": "application/json"
    }
    flow_payload = {
        "subscriber_id": sender_id,
        "flow_ns": "content20250424050612_308842"
    }
    res2 = requests.post(
        "https://api.manychat.com/fb/sending/sendFlow",
        headers=headers,
        json=flow_payload
    )
    print("✅ ManyChat Flow 전송 결과:", res2.json())

    return {
        "version": "v2",
        "content": {
            "messages": [
                {
                    "type": "text",
                    "text": f"✅ Option selected: {selected_option} (Extra: {extra_price:,})원)"
                }
            ]
        }
    }

class QuantityInput(BaseModel):
    sender_id: str
    product_quantity: int


def safe_int(val):
    try:
        return int(float(str(val).replace(",", "").replace("원", "").strip()))
    except:
        return 0


@app.post("/calculate_payment")
def calculate_payment(data: QuantityInput):
    try:
        sender_id = data.sender_id
        quantity = data.product_quantity

        if not sender_id or quantity is None:
            raise ValueError("❌ sender_id 또는 product_quantity 누락됨")

        # 🔍 캐시에서 상품 정보 불러오기
        product = None
        for p in PRODUCT_CACHE.values():
            if p.get("sender_id") == sender_id:
                product = p
                break

        if not product:
            raise ValueError("❌ 해당 유저의 상품 정보가 존재하지 않습니다.")

        # 🔢 기본 정보 추출
        price = safe_int(float(product.get("가격", 0)))
        extra_price = safe_int(float(product.get("추가금액", 0))) if "추가금액" in product else 0
        shipping = safe_int(float(product.get("배송비", 0)))
        max_quantity = safe_int(float(product.get("최대구매수량", 0)))

        # ✅ 총 가격 계산
        total_price = (price + extra_price) * quantity
        if max_quantity == 0:
            shipping_cost = shipping
        else:
            shipping_cost = shipping * math.ceil(quantity / max_quantity)

        total_price += shipping_cost

        # ✅ 천 단위 구분을 위한 포맷팅
        formatted_total_price = "{:,}".format(total_price)
        print(f"✅ 계산 완료 → 총금액: {formatted_total_price}원 (수량: {quantity}, 배송비: {shipping_cost:,}원)")

        # ✅ Manychat 필드 업데이트
        updater = ManychatFieldUpdater(sender_id, MANYCHAT_API_KEY)
        updater.set_quantity("12911653", quantity)  # Product_quantity 필드 ID
        updater.set_total_price("13013393", formatted_total_price)  # Total_price 필드 ID - 포맷팅된 값으로 저장

        # ✅ ManyChat 다음 Flow로 이동
        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        flow_payload = {
            "subscriber_id": sender_id,
            "flow_ns": "content20250501040123_213607"
        }
        res = requests.post(
            "https://api.manychat.com/fb/sending/sendFlow",
            headers=headers,
            json=flow_payload
        )
        print("✅ 최종결제금액 전송완료:", res.json())

        return {
            "Product_quantity": quantity,
            "Total_price": total_price
        }

    except Exception as e:
        print(f"❌ 결제 금액 계산 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ✅ 루트 경로 - HTML 페이지 렌더링
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/preview", response_class=HTMLResponse)
async def product_preview(html: str):
    try:
        decoded_html = base64.b64decode(html).decode("utf-8")
        return f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <title>상품 상세 페이지</title>
            <style>
                body {{
                    font-family: '맑은 고딕', sans-serif;
                    padding: 20px;
                    max-width: 800px;
                    margin: auto;
                    line-height: 1.5;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 20px auto;
                }}
            </style>
        </head>
        <body>
            {decoded_html}
        </body>
        </html>
        """
    except Exception as e:
        return HTMLResponse(content=f"<h1>오류 발생</h1><p>{e}</p>", status_code=400)


# ✅ FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011)
