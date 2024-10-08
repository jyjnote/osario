import requests

# 녹음된 파일 경로
filename = r"C:\Users\hopio\Documents\project\A_Way_Trip\recoding_tmp\test.m4a"

# 파일 전송 및 텍스트 변환 요청
try:
    url = "http://3.34.227.229:5000/transcribe/"
    with open(filename, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    # API로부터의 응답을 확인
    if response.status_code == 200:
        # 응답이 성공적이면 텍스트 출력
        print("Transcribed Text:", response.json().get("text"))
    else:
        # 오류 발생 시 상태 코드와 에러 메시지 출력
        print(f"Error: {response.status_code}, {response.text}")
except Exception as e:
    print(f"An error occurred while sending the file: {e}")