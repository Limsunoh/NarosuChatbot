from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from chatbot_langchain_helper import load_excel_database, search_products, extract_keywords, generate_gpt_response
import os

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# 데이터 로드
db_products = load_excel_database("db\db_products.xlsx")
db_products.columns = db_products.columns.str.strip().str.lower()

# 루트 페이지 반환
@app.route("/", methods=["GET"])
def index():
    return render_template("bot.html")  # templates 폴더 내 bot.html 반환

# 정적 파일 직접 서빙
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# 채팅 엔드포인트
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    search_results, unmatched_reason = search_products(db_products, user_input)
    keywords = extract_keywords(user_input, num_keywords=10)

    # ChatGPT 역할 정의 및 응답 생성
    role = "고객 지원 상담사"  # 역할 예시
    gpt_response = generate_gpt_response(keywords, role)

    if search_results.empty:
        return jsonify({
            "status": "fail",
            "message": "관련 상품을 찾을 수 없습니다.",
            "gpt_response": gpt_response,
            "extracted_keywords": keywords,
            "unmatched_conditions": unmatched_reason
        })

    products = [
        {col: row[col] for col in db_products.columns if col != 'match_score'}
        for _, row in search_results.iterrows()
    ]

    response = {
        "status": "success",
        "message": "추천 상품을 찾았습니다.",
        "gpt_response": gpt_response,
        "extracted_keywords": keywords,
        "products": products
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="localhost", port=5500, debug=True)
