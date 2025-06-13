from openai import OpenAI
from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema,
    DataType, Collection
)
from dotenv import load_dotenv
import pandas as pd
import re
import tiktoken        
import numpy as np
import time
import os

# ── 설정 ─────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
EXCEL_PATH  = "ownerclan_주간인기상품_0613_2.xlsx"
COLLECTION  = "ownerclan_weekly_0428"
MODEL       = "text-embedding-3-small"
MAX_TOKENS  = 300_000
CHUNK_SIZE  = 5000

# ─────────────────────────────────────────────────────────────

# 1) OpenAI 클라이언트 & Milvus 연결
client = OpenAI(api_key=API_KEY)
connections.connect(alias="default", host="114.110.135.96", port="19530")

# 2) 엑셀 로드
df = pd.read_excel(EXCEL_PATH)


# 3) 한글→영문 필드명 매핑
column_map = {
    "상품코드":        "product_code",    # 메타데이터만
    "카테고리코드":    "category_code",   # 메타데이터만
    "카테고리명":      "category_name",
    "마켓상품명":      "market_product_name",
    "마켓실제판매가":  "market_price",
    "배송비":          "shipping_fee",
    "배송유형":        "shipping_type",
    "최대구매수량":    "max_quantity",
    "조합형옵션":      "composite_options",
    "이미지중":        "image_url",        # 메타데이터만
    "제작/수입사":     "manufacturer",
    "모델명":          "model_name",
    "원산지":          "origin",
    "키워드":          "keywords",
    "본문상세설명":    "description",       # 메타데이터만
    "반품배송비":      "return_shipping_fee",
    "독립형":          "independent_option",  # 신규
    "조합형":          "composite_flag"       # 신규
}
numeric_fields = {
    "market_price",
    "shipping_fee",
    "max_quantity",
    "return_shipping_fee",
}

# 4) 각 열별 최대 문자열 길이 계산 & 출력
print("컬럼명(영문)         | 컬럼명(한글)     | 최대 길이")
for kor, eng in column_map.items():
    # strip() 추가하면 양쪽 공백·개행 제거
    max_len = df[kor].astype(str).str.strip().str.len().max()
    print(f"{eng:20s} | {kor:12s} | {max_len}")

# 5) Milvus 컬렉션 준비 (존재하면 삭제 → 재생성)
if utility.has_collection(COLLECTION):
    utility.drop_collection(COLLECTION)

fields = [  
    FieldSchema("id",                  DataType.INT64,        is_primary=True, auto_id=True),
    FieldSchema("emb",                 DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema("text",                DataType.VARCHAR,      max_length=65535),
    FieldSchema("product_code",        DataType.VARCHAR,      max_length=50),
    FieldSchema("category_code",       DataType.VARCHAR,      max_length=20),
    FieldSchema("category_name",       DataType.VARCHAR,      max_length=256),
    FieldSchema("market_product_name", DataType.VARCHAR,      max_length=512),
    FieldSchema("market_price",        DataType.INT64),
    FieldSchema("shipping_fee",        DataType.INT64),
    FieldSchema("shipping_type",       DataType.VARCHAR,      max_length=16),
    FieldSchema("max_quantity",        DataType.INT64),
    FieldSchema("composite_options",   DataType.VARCHAR,      max_length=16384),
    FieldSchema("image_url",           DataType.VARCHAR,      max_length=2048),
    FieldSchema("manufacturer",        DataType.VARCHAR,      max_length=128),
    FieldSchema("model_name",          DataType.VARCHAR,      max_length=256),
    FieldSchema("origin",              DataType.VARCHAR,      max_length=100),
    FieldSchema("keywords",            DataType.VARCHAR,      max_length=1024),
    FieldSchema("description",         DataType.VARCHAR,      max_length=65535),
    FieldSchema("return_shipping_fee", DataType.INT64),
    FieldSchema("independent_option",   DataType.VARCHAR,      max_length=4096),   # 텍스트로
    FieldSchema("composite_flag",       DataType.VARCHAR,      max_length=16384),   # 텍스트로
]
schema = CollectionSchema(fields, description="Weekly Top 50k Products")
collection = Collection(name=COLLECTION, schema=schema)

# # 제외할 필드 목록에 상품코드, 카테고리코드, 이미지중 추가
# exclude_fields = {
#     "상품코드",            # product_code – 사용자 검색에 의미 없음
#     "카테고리코드",        # category_code – 시스템 코드
#     "이미지중",            # image_url – 벡터화 불가능
#     "모델명",              # model_name – 대부분 NaN 또는 제품번호
#     "최대구매수량",        # max_quantity – 검색에는 영향 없음
#     "본문상세설명"         # description – HTML+노이즈 덩어리
# }

# def make_text(row):
#     parts = []
#     for kor, eng in column_map.items():
#         if kor in exclude_fields:
#             continue
#         parts.append(f"{eng}:{row[kor]}")
#     return " || ".join(parts)

# texts = df.apply(make_text, axis=1).tolist()

# 8) 임베딩용 텍스트 생성 (카테고리명, 상품명, 키워드, 조합형옵션만)
def make_text(row):
    return " || ".join([
        f"cat:{row['카테고리명']}",
        f"name:{row['마켓상품명']}",
        f"kw:{row['키워드']}",
        f"opts:{row['조합형옵션']}"
    ])

texts = df.apply(make_text, axis=1).tolist()

# 5-1) 임베딩용 텍스트 확인용 테이블 생성
df_embed_text = df.copy()
df_embed_text["embedding_text"] = texts

# 임베딩 텍스트 열만 보기 (원하는 만큼 행 제한도 가능)
print("\n✅ [임베딩 텍스트 미리보기]")
print(df_embed_text[["embedding_text"]].head(10).to_string(index=False))

# 6) 토큰 기준 배치 분할
def split_batches(texts, max_tokens=MAX_TOKENS):
    enc = tiktoken.get_encoding("cl100k_base")
    batches, batch, tokens = [], [], 0
    for t in texts:
        tk = len(enc.encode(t))
        if tokens + tk > max_tokens:
            batches.append(batch)
            batch, tokens = [], 0
        batch.append(t)
        tokens += tk
    if batch:
        batches.append(batch)
    return batches


batches = split_batches(texts)
embeddings = []
for idx, batch in enumerate(batches, 1):
    start = time.time()
    resp = client.embeddings.create(input=batch, model=MODEL)
    embeddings.extend([d.embedding for d in resp.data])
    print(f"배치 {idx}/{len(batches)} 완료: {time.time()-start:.1f}s")
embeddings = np.array(embeddings)


# 8) Milvus에 삽입 (최초 1회만)
if collection.num_entities == 0:
    n = len(texts)
    total_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    for idx, start in enumerate(range(0, n, CHUNK_SIZE), 1):
        end = min(start + CHUNK_SIZE, n)
        chunk_emb = embeddings[start:end].tolist()
        chunk_txt = texts[start:end]
        chunk_meta = [
            (df[kor].astype(int).tolist() if eng in numeric_fields
             else df[kor].astype(str).tolist())[start:end]
            for kor, eng in column_map.items()
        ]

        collection.insert([chunk_emb, chunk_txt] + chunk_meta)
        collection.flush()
        print(f"▶️ Milvus 삽입 완료: 청크 {idx}/{total_chunks} (rows {start}–{end-1})")
else:
    print("▶️ Milvus에 이미 데이터가 있어 삽입 스킵")

# 9) 인덱스 생성 & 로드 (한 번만)
print("▶️ 인덱스 생성 중…")
collection.create_index(
    field_name="emb",
    index_params={"index_type":"IVF_FLAT","metric_type":"L2","params":{"nlist":128}}
)
print("▶️ 인덱스 생성 완료, 컬렉션 로드 중…")
collection.load()
print("▶️ 컬렉션 로드 완료")

# 10) 검색 함수
def semantic_search_with_price(query, top_k=5):
    m = re.search(r'(\d+)\s*원', query)
    price = int(m.group(1)) if m else None

    q_emb = client.embeddings.create(input=[query], model=MODEL).data[0].embedding
    expr = f"market_price <= {price}" if price else None

    results = collection.search(
        data=[q_emb],
        anns_field="emb",
        param={"metric_type":"L2","params":{"nprobe":30}},   #nprobe=20~30 → recall 90% 이상 목표
        limit=top_k,
        expr=expr,
        output_fields=list(column_map.values()) + ["text"]
    )
    print(f"\n🔎 Query: {query}" + (f" (≤{price}원)" if price else ""))
    for hit in results[0]:
        print(f" • {hit.entity.get('market_product_name')} — "
              f"{hit.entity.get('market_price')}원 (dist={hit.distance:.3f})")

# 11) 예시 실행
if __name__ == "__main__":
    semantic_search_with_price("캠핑용품 추천해줘", 5)
    semantic_search_with_price("여름 티셔츠 중 3000원 이하", 5)
    semantic_search_with_price("큰 가방 추천해줘줘", 5)
    semantic_search_with_price("여름용 얇은 강아지 옷 추천받고 싶어요", 5)
    semantic_search_with_price("섬유유연제", 5)
    semantic_search_with_price("남성티셔츠", 5)
    semantic_search_with_price("겨울용품 추천해줘", 5)