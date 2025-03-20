import pickle
import pandas as pd

# .pkl 파일 열기
file_path = 'openai_embeddings_20250110.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 데이터 구조 및 첫 데이터 확인
print(f"데이터 타입: {type(data)}")

if isinstance(data, list):
    print(f"리스트 길이: {len(data)}")
    print("첫 번째 데이터 예시:", data[0])

    # 리스트가 벡터 데이터(중첩 리스트)인지 확인
    if isinstance(data[0], (list, tuple)):
        # 벡터 데이터를 DataFrame으로 변환
        df = pd.DataFrame(data)
        print("DataFrame으로 변환 완료!")
        print(df.head())
    else:
        print("데이터를 DataFrame으로 변환할 수 없습니다.")
        
elif isinstance(data, dict):
    print(f"딕셔너리 키: {list(data.keys())}")
    print("첫 번째 데이터 예시:", next(iter(data.values())))
else:
    print("데이터를 DataFrame으로 변환할 수 없습니다.")
