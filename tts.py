# apt install mpg123
import subprocess
import os
from echo_client import result

url = "https://kakaoi-newtone-openapi.kakao.com/v1/synthesize"

#data = """
#<speak>
#    <voice name="WOMAN_READ_CALM"> 그냥 가서 일해.</voice>
#    <voice name="MAN_READ_CALM"> 하하하하하</voice>
#    <voice name="WOMAN_DIALOG_BRIGHT"> 내 이름은 곽정은.</voice>
#    <voice name="MAN_DIALOG_BRIGHT"> 크크크크</voice>
#</speak>
#"""

text = result

data = """
<speak>
      <voice name="WOMAN_READ_CALM"> %s</voice>
</speak> 
""" % text


res = subprocess.Popen(['curl', '-v', '-X', 'POST', url,
                        '-H', "Content-Type: application/xml",
                        '-H', "Authorization: f086b91dd42878e96e93ebee3d0195fe",
                        '-d', data], 
                       stdout = subprocess.PIPE, stderr = subprocess.PIPE)

output, err = res.communicate()

# 파일 생성
f = open('/home/pi/AIPot/audio_test.wav', 'wb')
f.write(output)
f.close()

file = "/home/pi/AIPot/audio_test.wav"
os.system('cvlc --play-and-exit /home/pi/AIPot/audio_test.wav')
# 파일 삭제

os.remove("/home/pi/AIPot/audio_test.wav")

# print(err)
