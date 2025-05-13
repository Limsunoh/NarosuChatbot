from dotenv import load_dotenv
import os

load_dotenv()


key = os.getenv("MANYCHAT_API_KEY")
print("repr:", repr(key))  # → \x3a 나오면 인코딩 또는 파일 내용 이상
print("raw:", key)         # → : 이 나와야 정상
with open(".env", "rb") as f:
    content = f.read()
    print(content)
print("equal to ':'?", ':' in key)
print("ord check:", [hex(ord(c)) for c in key])

key2 = os.getenv("LANGCHAIN_ENDPOINT")
print("repr:", repr(key2))  # → \x3a 나오면 인코딩 또는 파일 내용 이상   
print("raw:", key2)         # → : 이 나와야 정상
with open(".env", "rb") as f:
    content = f.read()
    print(content)
print("equal to ':'?", ':' in key)
print("ord check:", [hex(ord(c)) for c in key])