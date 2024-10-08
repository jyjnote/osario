from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import tempfile
import os

app = FastAPI()

# XTTS 모델을 서버 시작 시 미리 로드합니다.
xtts_checkpoint = "finetune_models/ready/model.pth"
xtts_config = "finetune_models/ready/config.json"
xtts_vocab = "finetune_models/ready/vocab.json"
xtts_speaker = "finetune_models/ready/speakers_xtts.pth"

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        raise ValueError("You need to provide the XTTS checkpoint path, XTTS config path, and XTTS vocab path fields.")

    config = XttsConfig()
    config.load_json(xtts_config)
    model = Xtts.init_from_config(config)
    print("Loading XTTS model!")
    model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, speaker_file_path=xtts_speaker, use_deepspeed=False)

    if torch.cuda.is_available():
        model.cuda()

    print("Model Loaded!")
    return model

model = load_model(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker)

class TextToSpeechRequest(BaseModel):
    text: str

@app.post("/generate/")
async def generate_audio(request: TextToSpeechRequest):
    text = request.text

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # TTS를 통해 음성을 생성합니다.
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tts_output_path = run_tts(
            model,
            lang="ko",
            tts_text=text,
            speaker_audio_file="finetune_models/ready/reference.wav",
            temperature=0.75,
            length_penalty=1.0,
            repetition_penalty=5.0,
            top_k=50,
            top_p=0.85,
            sentence_split=True,
            use_config=False
        )

        # Save the audio file path
        adjusted_audio_path = adjust_pitch_and_speed(tts_output_path, octaves=0.0, speed_factor=1.0)
        return {"file_path": adjusted_audio_path}
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

def run_tts(model, lang, tts_text, speaker_audio_file, temperature, length_penalty, repetition_penalty, top_k, top_p, sentence_split, use_config):
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=model.config.gpt_cond_len, max_ref_length=model.config.max_r>

    if use_config:
        out = model.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=model.config.temperature,
            length_penalty=model.config.length_penalty,
            repetition_penalty=model.config.repetition_penalty,
            top_k=model.config.top_k,
            top_p=model.config.top_p,
            enable_text_splitting=True
        )
    else:
        out = model.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting=sentence_split
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return out_path

def adjust_pitch_and_speed(audio_path, octaves, speed_factor):
    from pydub import AudioSegment

    sound = AudioSegment.from_file(audio_path)
    new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    sound = sound.set_frame_rate(24000)

    # Adjust speed by changing frame rate
    new_sample_rate = int(sound.frame_rate * speed_factor)
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    sound = sound.set_frame_rate(24000)

    output_path = audio_path.replace('.wav', '_adjusted.wav')
    sound.export(output_path, format="wav")
    return output_path

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)
