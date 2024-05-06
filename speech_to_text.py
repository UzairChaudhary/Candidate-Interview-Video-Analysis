import whisper

model = whisper.load_model("medium")
result = model.transcribe("uploads/recorded_video.wav",fp16=False)
print(result["text"])