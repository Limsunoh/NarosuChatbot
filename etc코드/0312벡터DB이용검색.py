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
from typing import Union
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer


executor = ThreadPoolExecutor()

# ✅ 환경 변수 로드+
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
REDIS_URL = "redis://localhost:6379/0"
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
MANYCHAT_API_KEY = os.getenv('MANYCHAT_API_KEY')

print(f"🔍 로드된 VERIFY_TOKEN: {VERIFY_TOKEN}")
print(f"🔍 로드된 PAGE_ACCESS_TOKEN: {PAGE_ACCESS_TOKEN}")

# ✅ BERT 임베딩 모델 로드 (FAISS에서 쿼리 임베딩 생성)
BERT_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
embedding_model = SentenceTransformer(BERT_MODEL_NAME)

# ✅ FAISS 인덱스 파일 & 여러 개의 엑셀 데이터 파일
FAISS_INDEX_PATH = "faiss_index_03M.faiss"
excel_files = [
    "db/file1.xlsx",
    "db/file2.xlsx",
    "db/file3.xlsx",
    "db/file4.xlsx"
]

# ✅ FAISS 인덱스 로드
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"✅ FAISS 인덱스 로드 완료 ({index.ntotal} 개의 벡터 포함)")
else:
    raise FileNotFoundError(f"❌ FAISS 인덱스 파일이 존재하지 않습니다: {FAISS_INDEX_PATH}")

# ✅ 엑셀 데이터 로드 및 각 파일의 데이터 범위 저장
excel_dataframes = []
data_ranges = []  # 각 엑셀 파일의 데이터 범위를 저장

start_idx = 0
for excel_file in excel_files:
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
        excel_dataframes.append(df)
        data_ranges.append((start_idx, start_idx + len(df), excel_file))  # (시작 인덱스, 끝 인덱스, 파일명)
        start_idx += len(df)
        print(f"✅ 엑셀 데이터 로드 완료 ({len(df)} 개의 상품 정보) - {excel_file}")
    else:
        print(f"⚠️ 경고: 엑셀 파일 없음 - {excel_file}")

# ✅ FAISS 검색 함수 (검색 후 어느 엑셀 파일의 데이터인지 찾음)
def search_similar_documents(query, top_k=5):
    """
    (1) 검색어를 벡터로 변환 후 FAISS에서 유사한 벡터 검색
    (2) 검색된 인덱스를 기반으로 어느 엑셀 파일의 데이터인지 찾음
    """
    query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # ✅ FAISS에서 유사한 벡터 검색
    distances, indices = index.search(query_embedding, top_k)

    results = []
    
    for dist, idx in zip(distances[0], indices[0]):
        # ✅ 검색된 인덱스(idx)가 어느 엑셀 파일의 데이터인지 확인
        for start_idx, end_idx, file_name in data_ranges:
            if start_idx <= idx < end_idx:
                df_index = idx - start_idx  # ✅ 해당 엑셀 파일에서의 인덱스 변환
                df = pd.read_excel(file_name)  # ✅ 해당 엑셀 파일 로드
                product_info = df.iloc[df_index].to_dict()  # ✅ 상품 정보 가져오기
                
                results.append({
                    "source": file_name,
                    "index": df_index,
                    "distance": dist,
                    "product_info": product_info
                })
                break

    return sorted(results, key=lambda x: x["distance"])  # ✅ 거리 순 정렬

# ✅ CMD에서 검색 실행
if __name__ == "__main__":
    while True:
        query = input("\n🔍 검색어를 입력하세요 (종료: exit): ")
        if query.lower() == "exit":
            print("🚪 프로그램 종료")
            break

        results = search_similar_documents(query)

        if results:
            print("\n📌 검색 결과:")
            for i, result in enumerate(results):
                print(f"{i+1}. [파일: {result['source']}, 인덱스: {result['index']}, 거리: {result['distance']:.4f}]")
                print("    📄 제품 정보:")

                # ✅ 주요 제품 정보 출력
                product_info = result["product_info"]
                print(f"    🏷️ 제목: {product_info.get('상품코드', 'N/A')}")
                print(f"    💰 가격: {product_info.get('오너클랜판매가', 'N/A')} 원")
                print(f"    📦 원산지: {product_info.get('원산지', 'N/A')}")
                print(f"    📝 설명: {product_info.get('원본상품명', 'N/A')}")
                print(f"    🖼️ 이미지 링크: {product_info.get('이미지중', 'N/A')}")

                print("-" * 80)
        else:
            print("❌ 검색 실패")