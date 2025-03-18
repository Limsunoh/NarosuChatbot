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
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # 진행 바 추가
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


# ✅ FAISS 인덱스 파일 경로 설정
FAISS_INDEX_PATH = "faiss_index_03M.faiss"  # 저장된 FAISS 인덱스 경로

BERT_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# ✅ 여러 개의 엑셀 파일 리스트
excel_files = [
    "db/file1.xlsx",
    "db/file2.xlsx",
    "db/file3.xlsx",
    "db/file4.xlsx"
]

# ✅ 엑셀 데이터 로드 및 변환 함수
# ✅ 엑셀 데이터 로드 및 변환 함수
def load_excel_to_texts(file_path):
    """엑셀 파일을 텍스트 리스트로 변환"""
    try:
        data = pd.read_excel(file_path)
        data.columns = data.columns.str.strip()
        texts = [" | ".join([f"{col}: {row[col]}" for col in data.columns]) for _, row in data.iterrows()]
        return texts, data
    except Exception as e:
        raise Exception(f"❌ 엑셀 로드 오류: {e}")

# ✅ FAISS 인덱스가 존재하면 로드, 없으면 생성
if os.path.exists(FAISS_INDEX_PATH):
    print(f"✅ 기존 FAISS 인덱스 발견: {FAISS_INDEX_PATH}, 임베딩 생략")
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    print("🚀 FAISS 인덱스가 존재하지 않음, 새로 생성 중...")

    # ✅ 여러 개의 엑셀 파일을 읽어서 벡터 변환
    all_texts = []
    for file in tqdm(excel_files, desc="📂 엑셀 파일 로드 중"):
        texts, _ = load_excel_to_texts(file)
        all_texts.extend(texts)

    print(f"🔍 총 {len(all_texts)}개의 문장을 임베딩합니다.")

    # ✅ BERT 임베딩 모델 로드
    embedding_model = SentenceTransformer(BERT_MODEL_NAME)

    # ✅ 배치 단위로 BERT 임베딩 생성
    all_embeddings = []
    BATCH_SIZE = 200
    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="🚀 BERT 임베딩 진행 중"):
        batch_texts = all_texts[i:i+BATCH_SIZE]
        batch_embeddings = embedding_model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.extend(batch_embeddings)

    # ✅ FAISS 벡터 변환
    embeddings = np.array(all_embeddings, dtype=np.float32)

    # ✅ FAISS 인덱스 생성 (IndexIVFFlat 사용)
    d = embeddings.shape[1]
    nlist = 200
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    print("🔧 FAISS 인덱스 학습 중...")
    index.train(embeddings)
    
    print("📌 FAISS 인덱스 데이터 추가 중...")
    index.add(embeddings)

    # ✅ FAISS 인덱스 저장 (기존 경로 유지)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ FAISS 인덱스 저장 완료: {FAISS_INDEX_PATH}")