from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import gradio as gr
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']

# OpenAI 모델 초기화
llm = ChatOpenAI(temperature=1.0, model="gpt-4o-mini", openai_api_key=API_KEY)

def response(message, history, additional_input_info):
    """사용자 입력을 받아 OpenAI Assistant 응답을 반환하는 함수"""
    history_langchain_format = []
    if additional_input_info:
        history_langchain_format.append(SystemMessage(content=additional_input_info))
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format)
    return gpt_response.content

# ✅ Gradio 최신 버전에서 `retry_btn`, `undo_btn`, `clear_btn` 제거 완료
gr.ChatInterface(
    fn=response,
    textbox=gr.Textbox(placeholder="말을 걸어주세요..", container=False, scale=7),
    chatbot=gr.Chatbot(height=1000, type="messages"),  # ✅ 최신 메시지 형식 적용
    title="어떤 챗봇을 원하심미까?",
    description="물어보면 답하는 챗봇입니다.",
    theme="soft",
    examples=[["안녕하세요"], ["요즘 날씨가 어때요?"], ["점심메뉴 추천해주세요"]],
    additional_inputs=[
        gr.Textbox("", label="System Prompt를 입력해주세요", placeholder="I'm a friendly chatbot.")
    ]
).launch()
