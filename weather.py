import requests
import json
import datetime


# sentence = "분아 날씨 알려줘"
# key = "날씨"



def weather(sentence):
    # Classification
    # 일반 대화인지 정보인지 판단
    # 정보라면:
    # (공공데이터 기상데이터 API)
    
    # 네이버 api(tts API) 활용

    vilage_weather_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService/getUltraSrtNcst?"

    service_key = "Your API key"


    today = datetime.datetime.today()
    base_date = today.strftime("%Y%m%d") # "20210601" == 기준 날짜
    base_time = today.strftime("%H"+"00") # time

    nx = "67"
    ny = "101"

    payload = "serviceKey=" + service_key + "&" +\
        "dataType=json" + "&" +\
        "base_date=" + base_date + "&" +\
        "base_time=" + base_time + "&" +\
        "nx=" + nx + "&" +\
        "ny=" + ny

    #갑 요청
    res = requests.get(vilage_weather_url + payload)
    # print(res)
    # print(res.text)

    items = res.json().get('response').get('body').get('items')
    #{'item': [{'baseDate': '20210601',
    #   'baseTime': '1900',
    #   'category': 'POP',
    #   'fcstDate': '20210601',
    #   'fcstTime': '1900',
    #   'fcstValue': '0',
    #   'nx': 60,
    #   'ny': 128},
    #  {'baseDate': '20210601',
    #   'baseTime': '0500',
    #   'category': 'PTY',
    #   'fcstDate': '20210601',
    #   'fcstTime': '1900',
    #   'fcstValue': '0',
    #   'nx': 60,
    #   'ny': 128},
    #      'ny': 128},
    #     {'baseDate': '20210601'


    data = dict()
    data['date'] = base_date

    weather_data = dict()
    for item in items['item']:
        # 기온
        if item['category'] == 'T1H':
            weather_data['tmp'] = item['obsrValue']
        
        # 기상상태
        if item['category'] == 'PTY':
            
            weather_code = item['obsrValue']
            
            if weather_code == '1':
                weather_state = '비'
            elif weather_code == '2':
                weather_state = '비/눈'
            elif weather_code == '3':
                weather_state = '눈'
            elif weather_code == '4':
                weather_state = '소나기'
            else:
                weather_state = '맑음'
            
            weather_data['code'] = weather_code
            weather_data['state'] = weather_state

    data['weather'] = weather_data
    data['weather']
    # {'code': '0', 'state': '없음', 'tmp': '9'} # 9도 / 기상 이상 없음





    # print(items)

    # if key in sentence:
    #     result = "현재 온도는 {}도, 날씨는 {}입니다.".format(data['weather']['tmp'], data['weather']['state'])
    #     print(result)

    result = "현재 온도는 {}도, 날씨는 {}입니다.".format(data['weather']['tmp'], data['weather']['state'])

    return result
# weather(sentence)

