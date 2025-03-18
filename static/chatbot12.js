import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import Document

# .env 파일 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 샘플 데이터 (예제 문서)
documents = [
    Document(page_content="Apple은 iPhone을 만드는 회사입니다.", metadata={"출처": "기술"}),
    Document(page_content="Microsoft는 Windows 운영 체제를 개발합니다.", metadata={"출처": "기술"}),
    Document(page_content="바나나는 칼륨의 좋은 공급원입니다.", metadata={"출처": "음식"}),
    Document(page_content="오렌지는 비타민 C가 풍부한 감귤류 과일입니다.", metadata={"출처": "음식"})
]

# 1. 텍스트 데이터를 임베딩하여 벡터화
임베딩 = OpenAIEmbeddings(openai_api_key=openai_api_key)
벡터스토어 = FAISS.from_documents(documents, 임베딩)

# 2. 검색 및 생성 체인 구성
검색기 = 벡터스토어.as_retriever()
질의응답_체인 = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini", openai_api_key=openai_api_key),
    retriever=검색기,
    return_source_documents=True  # 검색된 문서 정보 반환
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    질문 = data.get("question", "")
    if not 질문:
        return jsonify({"error": "질문이 필요합니다."}), 400

    # 질의응답 처리
    결과 = 질의응답_체인.invoke({"query": 질문})
    답변 = 결과["result"]
    출처 = [{"출처": 문서.metadata["출처"], "내용": 문서.page_content} for 문서 in 결과["source_documents"]]

    return jsonify({"answer": 답변, "sources": 출처})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
