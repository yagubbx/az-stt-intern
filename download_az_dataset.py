#model yukleme hissesi
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="nijatzeynalov/wav2vec2-large-mms-1b-azerbaijani-common_voice15.0"
)

result = asr("audio.wav")
print(result["text"])