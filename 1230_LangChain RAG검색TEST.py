import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# **1. 환경 변수 로드**
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI API Key가 설정되지 않았습니다.")
os.environ["OPENAI_API_KEY"] = api_key

# **2. LangChain RAG 기반 임베딩 초기화**
def initialize_embeddings(documents):
    """
    문서를 벡터화하고 FAISS 벡터 저장소를 생성합니다.
    """
    embeddings = OpenAIEmbeddings()  # OpenAI Embeddings 사용
    vectorstore = FAISS.from_texts(documents, embeddings)  # 벡터 저장소 생성
    return vectorstore

# **3. TF-IDF 기반 검색**
def tfidf_search(query, documents):
    """
    TF-IDF를 사용하여 문서 간 유사도를 계산합니다.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-3:][::-1]
    return [documents[i] for i in top_indices]

# **4. 신경망 기반 검색**
def neural_search(query, documents):
    """
    SentenceTransformer를 사용하여 문서 간 의미적 유사도를 계산합니다.
    """
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')  # 다국어 모델
    doc_embeddings = model.encode(documents)
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
    top_indices = similarities.argsort()[-3:][::-1]
    return [documents[i] for i in top_indices]

# **5. RAG 기반 답변 생성**
def generate_rag_response(query, vectorstore):
    """
    LangChain을 사용하여 검색 및 응답을 생성합니다.
    """
    retriever = vectorstore.as_retriever()  # 벡터 저장소에서 검색 가능하도록 설정
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-4", temperature=0), retriever=retriever)
    response = qa_chain.invoke({"query": query})
    return response

# **6. 테스트 데이터**
documents = [
    "이 문서는 머신러닝에 관한 내용입니다.",
    "딥러닝 기법에 대해 배우고 있습니다.",
    "이 문서는 신경망의 사용에 대해 설명합니다.",
    "정보 검색은 인공지능에서 중요한 주제입니다.",
    "자연어 처리는 매력적인 분야입니다.",
    "이 텍스트는 시맨틱 검색과 임베딩에 대한 내용입니다.",
]

# **7. 메인 실행**
if __name__ == "__main__":
    # 사용자 입력 받기
    query = input("문장을 입력하세요: ")

    # LangChain Embedding
    print("\n[LangChain RAG 기반 Embedding 검색]")
    vectorstore = initialize_embeddings(documents)  # 문서 임베딩
    rag_response = generate_rag_response(query, vectorstore)  # 검색 및 생성
    print(f"RAG 응답: {rag_response}")

    # TF-IDF 검색
    print("\n[TF-IDF 기반 검색]")
    tfidf_results = tfidf_search(query, documents)
    for result in tfidf_results:
        print(f"- {result}")

    # 신경망 기반 검색
    print("\n[신경망 기반 검색]")
    neural_results = neural_search(query, documents)
    for result in neural_results:
        print(f"- {result}")
