from dotenv import load_dotenv
import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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

# ✅ FAISS 인덱스 로드
def load_faiss_index(file_path):
    try:
        index = faiss.read_index(file_path)
        print(f"✅ FAISS 인덱스가 {file_path}에서 성공적으로 로딩되었습니다!")
        return index
    except Exception as e:
        print(f"❌ FAISS 인덱스 로딩 실패: {e}")
        return None

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

# ✅ 메인 실행 로직: 기존 인덱스 확인 후 로딩 또는 새로 생성
if os.path.exists(faiss_file_path):
    print(f"📦 기존 FAISS 인덱스 파일 발견: {faiss_file_path}")
    index = load_faiss_index(faiss_file_path)
else:
    print("⚙️ FAISS 인덱스 파일이 존재하지 않습니다. 임베딩을 생성합니다.")
    create_and_save_faiss_index("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")

# ✅ 벡터 수 및 차원 수 출력
if index:  
    print(f"✅ 저장된 벡터 수: {index.ntotal}")
    print(f"✅ 벡터 차원 수: {index.d}")
else:
    print("❌ FAISS 인덱스 로딩에 실패하였습니다.")

#########============임베딩 저장 후 로드============#########

# ✅ 검색 쿼리 실행 및 LLM 응답 생성 (CMD 입력 지원)
def search_and_generate_response(file_path):
    query = input("💬 상품 검색 문장을 입력하세요: ")
    index = load_faiss_index(file_path)
    data = pd.read_excel("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")

    if index is None:
        print("❌ FAISS 인덱스를 로드할 수 없습니다.")
        return

    임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
    query_embedding = 임베딩.embed_documents([query])

    # ✅ FAISS 검색 수행
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k=5)
    print(f"🔍 검색된 인덱스: {I}")

    # ✅ 검색 결과를 상품코드, 원본상품명, 오너클랜판매가, 배송비, 이미지중, 원산지로 출력
    results = []
    for idx in I[0]:
        result_row = data.iloc[idx]
        result_info = {
            "상품코드": result_row["상품코드"],
            "원본상품명": result_row["원본상품명"],
            "오너클랜판매가": result_row["오너클랜판매가"],
            "배송비": result_row["배송비"],
            "이미지중": result_row["이미지중"],
            "원산지": result_row["원산지"]
        }
        results.append(result_info)

    # ✅ LLM을 사용하여 검색 결과를 설명
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", api_key=API_KEY)
    response = llm.invoke(f"사용자가 요청한 '{query}'에 대한 상위 상품은: {I}")
    print(f"✅ LLM 응답: {response}")
    print(f"✅ 검색된 상품 정보: {results}")

# ✅ 사용자 질문에 대한 검색 및 응답 생성 (CMD 입력 지원)
search_and_generate_response(faiss_file_path)
