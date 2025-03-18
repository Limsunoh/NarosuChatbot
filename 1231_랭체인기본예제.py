import os
import pandas as pd
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

# 업로드된 엑셀 파일을 데이터로 로드
def load_excel_to_documents(file_path):
    try:
        data = pd.read_excel(file_path)
        documents = [
            Document(
                page_content=" | ".join([f"{col}: {row[col]}" for col in data.columns]),
                metadata={col: row[col] for col in data.columns}
            )
            for _, row in data.iterrows()
        ]
        return documents
    except Exception as e:
        print(f"엑셀 파일 로드 오류: {e}")
        raise

# 엑셀 파일로부터 문서 생성
documents = load_excel_to_documents("db/infoitems2.xlsx")

# 1. 텍스트 데이터를 임베딩하여 벡터화
try:
    임베딩 = OpenAIEmbeddings(openai_api_key=openai_api_key)
    벡터스토어 = FAISS.from_documents(documents, 임베딩)
except Exception as e:
    print(f"임베딩 모델 초기화 오류: {e}")
    raise

# 2. LLM(ChatOpenAI) 초기화
try:
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini", openai_api_key=openai_api_key)
except Exception as e:
    print(f"LLM 초기화 오류: {e}")
    raise

# 3. 검색 및 생성 체인 구성
try:
    검색기 = 벡터스토어.as_retriever()
    질의응답_체인 = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=검색기,
        return_source_documents=False  # 검색된 문서 정보 반환 안 함
    )
except Exception as e:
    print(f"질의응답 체인 초기화 오류: {e}")
    raise

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
    try:
        print(f"질문 수신: {질문}")  # 디버깅용 로그 추가

        # 질문 임베딩 확인
        질문_임베딩 = 임베딩.embed_query(질문)
        print(f"질문 임베딩 벡터: {질문_임베딩}")

        # 핵심 단어 추출 (간단한 명사 추출 예제)
        핵심_단어 = [word for word in 질문.split() if len(word) > 1]
        print(f"추출된 핵심 단어: {핵심_단어}")

        결과 = 질의응답_체인.invoke({"query": 질문})
        답변 = 결과["result"]
        print(f"생성된 답변: {답변}")  # 디버깅용 로그 추가

        return jsonify({"answer": 답변, "embedding": 질문_임베딩, "keywords": 핵심_단어})
    except Exception as e:
        print(f"질의응답 처리 오류: {e}")
        return jsonify({"error": "질의응답 처리 중 오류가 발생했습니다."}), 500

# 테스트용 LLM 직접 호출 엔드포인트 추가
@app.route('/test_llm', methods=['GET'])
def test_llm():
    try:
        test_message = "랭스미스에 대해 알려주세요."
        response = llm(test_message)
        print(f"LLM 테스트 응답: {response}")
        return jsonify({"test_message": test_message, "response": response})
    except Exception as e:
        print(f"LLM 테스트 오류: {e}")
        return jsonify({"error": "LLM 테스트 중 오류가 발생했습니다."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
