from dotenv import load_dotenv
import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
import pickle
from datetime import datetime

# ✅ 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# ✅ 오늘 날짜로 임베딩 벡터 파일 경로 생성
embedding_file_path = f"openai_embeddings_{datetime.now().strftime('%Y%m%d')}.pkl"

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

# ✅ OpenAI 임베딩 생성 및 벡터 저장
def create_and_save_openai_embeddings(file_path):
    texts = load_excel_to_texts(file_path)
    임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
    
    # ✅ OpenAI 임베딩 생성
    embeddings = 임베딩.embed_documents(texts)
    
    # ✅ 임베딩 벡터를 로컬에 저장 (.pkl)
    with open(embedding_file_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"✅ OpenAI 임베딩 벡터가 {embedding_file_path}에 저장되었습니다!")

# ✅ 실행: 엑셀 데이터로부터 임베딩 벡터 생성 및 저장
create_and_save_openai_embeddings("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")
