import pandas as pd
from fastapi import HTTPException

# ✅ 엑셀 데이터 로드 및 변환 (본문상세설명 컬럼 제외하고 임베딩용 텍스트 생성)
def load_excel_to_texts(file_path):
    try:
        data = pd.read_excel(file_path)
        data.columns = data.columns.str.strip()

        # '본문상세설명' 컬럼은 임베딩 대상에서 제외
        if '본문상세설명' in data.columns:
            embedding_df = data.drop(columns=['본문상세설명'])
        else:
            embedding_df = data

        texts = [" | ".join([f"{col}: {row[col]}" for col in embedding_df.columns]) for _, row in embedding_df.iterrows()]

        print(f"✅ 총 {len(texts)}개의 텍스트가 생성되었습니다.")
        print("🔍 예시 출력 (1줄):")
        print(texts[0])

        return texts, data  # texts는 임베딩용, data는 전체 컬럼 포함

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"엑셀 파일 로드 오류: {str(e)}")


# 테스트용 코드 (직접 실행할 수 있음)
if __name__ == "__main__":
    file_path = "db/ownerclan_주간인기상품_5만개.xlsx"  # 실제 파일 경로에 맞게 조정
    load_excel_to_texts(file_path)