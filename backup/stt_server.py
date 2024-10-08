from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import whisper
import os
import funs_vangogh
import httpx

app = FastAPI()

# Whisper 모델을 서버 시작 시 미리 로드합니다.
print("Loading Whisper model...")
model = whisper.load_model("large")
print("Whisper model loaded.")

class TextResponse(BaseModel):
    text: str

class TextRequest(BaseModel):
    text: str

class returnModelResponse(BaseModel):
    text: str
    returnAi: str
    path: str


@app.post("/onlytext", response_model=returnModelResponse)
async def onlyText(request: TextRequest):
    text = request.text

    return_ai = funs_vangogh.run(text)

    # TTS 서버로 텍스트를 전송하여 음성을 생성하도록 요청합니다.
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post("http://localhost:5003/generate/", json={"text": return_ai})

    if response.status_code == 200:
        wav_file = response.json().get("file_path")
        return returnModelResponse(
            text=text,
            returnAi=return_ai,
            path=wav_file
        )
    else:
        return returnModelResponse(
            text="Error generating audio.",
            returnAi=None,
            path=None
        )


@app.post("/transcribe", response_model=returnModelResponse)
async def transcribe(file: UploadFile = File(...)):
    file_location = "temp.wav"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Whisper 모델을 사용하여 텍스트로 변환합니다.
    result = model.transcribe(file_location, language="ko")
    text = result.get("text", "")

    os.remove(file_location)  # 임시 파일 삭제

    if not text:
        return TextResponse(text="No text transcribed.")

    # funs_vangogh의 run 함수를 호출하여 텍스트에 대한 응답을 생성합니다.
    return_ai = funs_vangogh.run(text)

    # TTS 서버로 텍스트를 전송하여 음성을 생성하도록 요청합니다.
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post("http://localhost:5003/generate/", json={"text": return_ai})

    if response.status_code == 200:
        wav_file = response.json().get("file_path")
        return returnModelResponse(
                text=text,
                returnAi=return_ai,
                path=wav_file
                            )
    else:
        return returnModelResponse(
                text="Error generating audio.",
                returnAi=None,
                path=None
                )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
