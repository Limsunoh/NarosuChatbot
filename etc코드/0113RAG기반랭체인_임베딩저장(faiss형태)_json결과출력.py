from dotenv import load_dotenv
import os
import pandas as pd
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import faiss
import numpy as np
from datetime import datetime

# ✅ 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# ✅ 오늘 날짜로 FAISS 인덱스 파일 경로 생성
faiss_file_path = f"faiss_index_{datetime.now().strftime('%Y%m%d')}.faiss"

# ✅ 엑셀 데이터 로드 및 변환 (공백 제거)
def load_excel_to_texts(file_path):
    try:
        data = pd.read_excel(file_path)
        data.columns = data.columns.str.strip()  # 공백 제거
        texts = [
            " | ".join([f"{col}: {row[col]}" for col in data.columns])
            for _, row in data.iterrows()
        ]
        return texts, data
    except Exception as e:
        print(f"엑셀 파일 로드 오류: {e}")
        raise

# ✅ FAISS 인덱스 저장 (CPU 버전)
def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)
    print(f"✅ FAISS 인덱스가 {file_path}에 저장되었습니다!")

# ✅ FAISS 인덱스 로드
def load_faiss_index(file_path):
    try:
        index = faiss.read_index(file_path)
        print(f"✅ FAISS 인덱스가 {file_path}에서 로딩되었습니다!")
        return index
    except Exception as e:
        print(f"❌ FAISS 인덱스 로딩 실패: {e}")
        return None

# ✅ OpenAI 임베딩 및 FAISS 인덱스 생성 및 저장
def create_and_save_faiss_index(file_path):
    texts, _ = load_excel_to_texts(file_path)
    임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
    
    # ✅ 임베딩 벡터 생성
    embeddings = 임베딩.embed_documents(texts)
    
    # ✅ FAISS 인덱스 생성 (CPU 버전)
    index = faiss.IndexFlatL2(1536)
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
    index = load_faiss_index(faiss_file_path)

# ✅ 벡터 수 및 차원 수 출력
if index:
    print(f"✅ 저장된 벡터 수: {index.ntotal}")
    print(f"✅ 벡터 차원 수: {index.d}")
else:
    print("❌ FAISS 인덱스 로딩에 실패하였습니다.")

#########============임베딩 저장 후 로드 및 JSON 변환============#########

# ✅ JSON 직렬화를 위한 int 변환 함수
def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    return obj

# ✅ 검색 쿼리 실행 및 JSON 결과 반환
def search_and_generate_response(file_path):
    query = input("💬 상품 검색 문장을 입력하세요: ")
    index = load_faiss_index(file_path)
    _, data = load_excel_to_texts("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")

    if index is None:
        print("❌ FAISS 인덱스를 로드할 수 없습니다.")
        return json.dumps({"error": "FAISS 인덱스를 로드할 수 없습니다."}, ensure_ascii=False)

    # ✅ OpenAI 임베딩 생성
    임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
    query_embedding = 임베딩.embed_query(query)

    # ✅ FAISS 검색 수행
    D, I = index.search(np.array([query_embedding], dtype=np.float32), k=5)

    # ✅ 검색 결과를 JSON 형식으로 변환 (int64 → int 변환 적용)
    results = []
    for idx in I[0]:
        if idx >= len(data):
            continue
        result_row = data.iloc[idx]
        result_info = {
            "상품코드": str(result_row["상품코드"]),  # 문자열 변환 적용
            "원본상품명": result_row["원본상품명"],
            "오너클랜판매가": convert_to_serializable(result_row["오너클랜판매가"]),
            "배송비": convert_to_serializable(result_row["배송비"]),
            "이미지중": result_row["이미지중"],
            "원산지": result_row["원산지"]
        }
        results.append(result_info)

    # ✅ LLM을 사용하여 JSON 기반 설명 생성
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", api_key=API_KEY)
    response = llm.invoke(f"사용자가 요청한 '{query}'에 대한 상위 5개의 검색 결과는 다음과 같습니다: {json.dumps(results, ensure_ascii=False)}")

    # ✅ 최종 JSON 직렬화로 결과 반환 (int64 변환 포함)
    final_output = {
        "query": query,
        "results": results,
        "llm_response": str(response)
    }

    # ✅ JSON 형식으로 결과 출력
    print(json.dumps(final_output, ensure_ascii=False, indent=4, default=convert_to_serializable))

# ✅ 사용자 질문에 대한 검색 및 JSON 응답 생성
search_and_generate_response(faiss_file_path)
