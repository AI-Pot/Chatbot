# NOTE: this example requires PyAudio because it uses the Microphone class
# !pip install SpeechRecognition
# !pip install pyaudio
# pyaudio가 다운 안된다면 sudo apt-get install portaudio19-dev 실행




#### 카카오 API 받기
import requests
import json
import ast

REST_API_KEY = ""
url = 'https://kakaoi-newtone-openapi.kakao.com/v1/recognize'
# Transfer-Encoding: chunked# 길게 입력받고 싶을 때
header = {"Content-Type": "application/octet-stream", 
        "Authorization": "KakaoAK " + REST_API_KEY}


#### 마이크 입력
import speech_recognition as sr

# obtain audio from the microphone
# recognizer 객체 할당
r = sr.Recognizer()
# 마이크 객체 설정
with sr.Microphone(sample_rate = 16000) as source:
    print("Say something!")
    # 소리가 날때까지 대기
    # 말하면 음성이 audio에 저장
    audio = r.listen(source)
    print(audio)

# 음성파일 저장
# # write audio to a WAV file
# with open("microphone-results.wav", "wb") as f:
#     f.write(audio.get_wav_data())

#### 요청하기
#  audio.get_raw_data()# 이 데이터를 request를 날릴때 사용
# false = False
res = requests.post(url, headers = header, data = audio.get_raw_data())
print(type(res))
res_text = res.text
result_json_string = res_text[res_text.index('{"type":"finalResult"'):res_text.rindex('}')+1]
print(result_json_string)
result = json.loads(result_json_string)
question = result['value']
print(question)

