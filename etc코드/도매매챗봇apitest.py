import json
import requests

# Domeggook API 호출 함수
def search_products():
    """
    사용자로부터 검색어를 입력받아 Domeggook API를 호출하고 결과를 반환합니다.
    """
    # Setting URL
    url = 'https://domeggook.com/ssl/api/'

    # 사용자로부터 검색어 입력 받기
    search_keyword = input("검색 키워드를 입력하세요: ").strip()

    # Setting Request Parameters
    param = dict()
    param['ver'] = '4.1'               # API 버전
    param['mode'] = 'getItemList'      # API 모드
    param['aid'] = 'b60537cd4d5a4917a817348ca310bb74'  # API 키
    param['market'] = 'supply'           # 시장 유형
    param['om'] = 'json'               # 출력 형식
    param['kw'] = search_keyword       # 검색 키워드

    # Sending Request
    try:
        res = requests.get(url, params=param)
        res.raise_for_status()  # HTTP 에러 발생 시 예외 처리

        # Parsing Response
        data = json.loads(res.content)
        return data, search_keyword
    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 오류 발생: {e}")
        return None, None

# 결과를 파일로 저장
def save_to_txt(data, search_keyword):
    """
    검색 결과를 텍스트 파일로 저장합니다.
    """
    file_name = f"{search_keyword}_검색결과.txt"
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            # JSON 데이터를 보기 좋게 텍스트로 변환
            formatted_data = json.dumps(data, indent=4, ensure_ascii=False)
            f.write(formatted_data)
        print(f"검색 결과가 '{file_name}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")

# 메인 실행
if __name__ == "__main__":
    # 검색 함수 호출
    result, search_keyword = search_products()

    # 결과 출력 및 파일 저장
    if result:
        print(json.dumps(result, indent=4, ensure_ascii=False))  # JSON 데이터를 보기 좋게 출력
        save_to_txt(result, search_keyword)  # 결과를 파일로 저장
    else:
        print("API 요청 실패 또는 결과 없음.")
