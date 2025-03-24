from dotenv import load_dotenv
import os
from langchain_community.llms import OpenAI
# .env 파일에서 환경변수 로드
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
# 확인
print(f":열쇠와_잠긴_자물쇠: OpenAI API Key: {':흰색_확인_표시: 존재함' if API_KEY else ':x: 없음'}")
# LLM 인스턴스 생성
llm = OpenAI(openai_api_key=API_KEY, temperature=0.5)
# 테스트용 사용자 입력
user_input = "저렴하고 배송 빠른 국산 사과를 추천해줘"
# 키워드 추출용 프롬프트
prompt = f"""
다음 문장에서 핵심 키워드만 콤마(,)로 나열해줘.
문장: "{user_input}"
"""
# LLM 호출
try:
    response = llm.invoke(prompt)
    print("\n:돋보기: 추출된 키워드:")
    print(response)
except Exception as e:
    print(f"\n:x: 오류 발생: {e}")