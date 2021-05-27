curl -v -X POST "https://kakaoi-newtone-openapi.kakao.com/v1/synthesize" \
-H "Content-Type: application/xml" \
-H "Authorization: KakaoAK f086b91dd42878e96e93ebee3d0195fe" \
-d '<speak> 그는 그렇게 말했습니다. 
<voice name="MAN_DIALOG_BRIGHT">잘 지냈어? 나도 잘 지냈어.</voice> 
<voice name="WOMAN_DIALOG_BRIGHT" speechStyle="SS_ALT_FAST_1">금요일이 좋아요.</voice> </speak>' > result.mp3
