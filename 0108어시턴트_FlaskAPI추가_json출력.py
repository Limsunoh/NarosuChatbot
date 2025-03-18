from dotenv import load_dotenv
import os
from openai import OpenAI
import time
from flask import Flask, request, jsonify, render_template

# 환경 변수 로드
load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
assistant_id = os.environ['ASSISTANT_ID']

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=API_KEY)
app = Flask(__name__, template_folder="templates")

# thread id를 하나로 고정
thread = client.beta.threads.create()
thread_id = thread.id

# HTML 페이지 서빙 라우트
@app.route('/')
def index():
    return render_template('index.html')

# Flask 엔드포인트 (Postman 및 HTML 통신 지원)
@app.route('/chatbot', methods=['POST'])
def chat_with_postman():
    global thread_id
    data = request.get_json()
    user_message = data.get("message")

    if not thread_id:
        thread = client.beta.threads.create()
        thread_id = thread.id

    # 응답 시간 측정 시작
    start_time = time.time()

    # OpenAI에 메시지 전송
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )

    # Assistant 실행
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

    while run.status != "completed":
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )

    # 응답 시간 측정 종료
    end_time = time.time()
    response_time = end_time - start_time
    print(f"API 응답 시간: {response_time:.2f}초")

    # 마지막 응답 반환
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    last_message = messages.data[0].content[0].text.value
    return jsonify({"response": last_message, "thread_id": thread_id})

if __name__ == "__main__":
    app.run(port=5050)
