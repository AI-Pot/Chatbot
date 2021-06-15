import requests
import json
import speech_recognition as sr
import ast

url = "https://kakaoi-newtone-openapi.kakao.com/v1/recognize"
rest_key = 'f086b91dd42878e96e93ebee3d0195fe'

header = {"Content-Type" : "application/octet-stream",
          "Authorization" : "KakaoAK " + rest_key
          }

r = sr.Recognizer()
with sr.Microphone(sample_rate=16000) as source:
    print('say something')
    audio = r.listen(source)
    # print(audio)

res = requests.post(url, headers=header, data=audio.get_raw_data())
# print(type(res))

res_text = res.text
result_json_string = res_text[res_text.index('{"type":"finalResult"'):res_text.rindex('}')+1]
# print(result_json_string)

result = json.loads(result_json_string)
question = result['value']
print(question)
