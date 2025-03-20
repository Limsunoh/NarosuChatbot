from dotenv import load_dotenv
import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np
from datetime import datetime

# ✅ 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# ✅ 오늘 날짜로 FAISS 인덱스 파일 경로 생성
faiss_file_path = f"faiss_index_{datetime.now().strftime('%Y%m%d')}.faiss"

# ✅ 엑셀 데이터 로드 및 변환
def load_excel_to_texts(file_path):
    try:
        data = pd.read_excel(file_path)
        texts = [
            " | ".join([f"{col}: {row[col]}" for col in data.columns])
            for _, row in data.iterrows()
        ]
        return texts
    except Exception as e:
        print(f"엑셀 파일 로드 오류: {e}")
        raise

# ✅ FAISS 인덱스 저장 (GPU → CPU 변환 후 저장)
def save_faiss_index(index, file_path):
    cpu_index = faiss.index_gpu_to_cpu(index)  # GPU 인덱스를 CPU로 변환
    faiss.write_index(cpu_index, file_path)
    print(f"✅ FAISS 인덱스가 {file_path}에 저장되었습니다!")

# ✅ OpenAI 임베딩 및 FAISS 인덱스 생성 및 저장
def create_and_save_faiss_index(file_path):
    texts = load_excel_to_texts(file_path)
    임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
    
    # ✅ 임베딩 벡터 생성
    embeddings = 임베딩.embed_documents(texts)
    
    # ✅ FAISS 인덱스 생성 및 벡터 추가 (GPU 사용)
    index = faiss.IndexFlatL2(1536)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(np.array(embeddings, dtype=np.float32))
    
    # ✅ FAISS 인덱스 저장
    save_faiss_index(index, faiss_file_path)

# ✅ 실행: 엑셀 데이터로부터 FAISS 인덱스 생성 및 저장
create_and_save_faiss_index("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")
