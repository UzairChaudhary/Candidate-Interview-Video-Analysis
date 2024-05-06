# api="sk-proj-Ew15pEyLZ7XOd6uC46b6T3BlbkFJXcRQDB9mQdd1gdWBqDIj"
# from openai import OpenAI
# client = OpenAI()

# audio_file= open("uploads/recorded_video.wav", "rb")
# transcription = client.audio.transcriptions.create(
#   model="whisper-1", 
#   file=audio_file
# )
# print(transcription.text)
import whisper

model = whisper.load_model("medium")
result = model.transcribe("uploads/recorded_video.wav",fp16=False)
print(result["text"])