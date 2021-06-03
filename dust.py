import requests
import json
import datetime

# sentence = "분아 날씨 알려줘"
# key = "날씨"

def dust(sentence):    
    dust_url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty?"

    service_key = "Your service key"
    version = "1.3"
    item_code_pm10 = "pm10Value"
    item_code_pm25 = "pm25Value"

    sidoName = "대전"

    payload = "serviceKey=" + service_key + "&" +\
        "returnType=xml" + "&" +\
        "ver=" + version + "&" +\
        "sidoName=" + sidoName + "&" +\
        "itemCode="

    # pm10 pm2.5 수치 가져오기
    pm10_res = requests.get(dust_url + payload + item_code_pm10)
    pm25_res = requests.get(dust_url + payload + item_code_pm25)   
    # print(pm10_res)# Response [200]
    # print(pm25_res)# Response [200]


    # xml 파싱하기
    import xml.etree.ElementTree as elemTree
    # print(pm10_res.text)
    # print(pm25_res.text)

# print(pm10_res.text)
#     <?xml version="1.0" encoding="UTF-8"?>
# <response>
#   <header>
#     <resultCode>00</resultCode>
#     <resultMsg>NORMAL_CODE</resultMsg>
#   </header>
#   <body>
#     <items>
#       <item>
#         <so2Grade>1</so2Grade>
#         <coFlag/>
#         <khaiValue>92</khaiValue>
#         <so2Value>0.003</so2Value>
#         <coValue>0.4</coValue>
#         <pm10Flag/>
#         <o3Grade>2</o3Grade>
#         <pm10Value>38</pm10Value>
#         <khaiGrade>2</khaiGrade>
#         <sidoName>대전</sidoName>
#         <no2Flag/>
#         <no2Grade>1</no2Grade>
#         <o3Flag/>
#         <so2Flag/>
#         <dataTime>2021-05-31 16:00</dataTime>
#         <coGrade>1</coGrade>
#         <no2Value>0.009</no2Value>
#         <stationName>정림동</stationName>
#         <pm10Grade>1</pm10Grade>
#         <o3Value>0.080</o3Value>
#       </item>

    pm10_tree = elemTree.fromstring(pm10_res.text)
    pm25_tree = elemTree.fromstring(pm25_res.text)
    # print(pm10_tree)# <Element 'response' at 0x7f6f8c16bbf0>
    # print(pm25_tree)# <Element 'response' at 0x7f6f8c16bbf0>
    dust_data = dict()
    for tree in [pm10_tree, pm25_tree]:

        item = tree.find("body").find("items").find("item")
        code = item.findtext("itemCode")
        value1 = int(item.findtext("pm10Value"))
        value2 = int(item.findtext("pm25Value"))
        dust_data["PM10"] = {'value' : value1}
        dust_data["PM2.5"] = {'value' : value2}

    # 결과 값
    dust_data
    # print(dust_data)# {'PM10': {'value': 94}, 'PM2.5': {'value': 71}}




    # PM10 미세먼지 30 80 150
    pm10_value = dust_data.get('PM10').get('value')
    if pm10_value <= 30:
        pm10_state = "좋음"
    elif pm10_value <= 80:
        pm10_state = "보통"
    elif pm10_value <= 150:
        pm10_state = "나쁨"
    else:
        pm10_state = "매우나쁨"

    pm25_value = dust_data.get('PM2.5').get('value')
    # PM2.5 초미세먼지 15 35 75
    if pm25_value <= 15:
        pm25_state = "좋음"
    elif pm25_value <= 35:
        pm25_state = "보통"
    elif pm25_value <= 75:
        pm25_state = "나쁨"
    else:
        pm25_state = "매우나쁨"

    # 미세먼지가 나쁜 상태인지(1)/아닌지(0)
    if pm10_value > 80 or  pm25_value > 75:
        dust_code = "1"
    else:
        dust_code = "0"

    dust_data.get('PM10')['state'] = pm10_state
    dust_data.get('PM2.5')['state'] = pm25_state
    dust_data['code'] = dust_code

    data = {}
    data['dust'] = dust_data
    # print(data['dust'])
    # {
    # 'PM10': {'value': 94, 'state': '나쁨'},
    # 'PM2.5': {'value': 71, 'state': '나쁨'}
    # }



    # # 날씨 정보
    # # data
    # {
    #     'weather': {
    #         'code': '0', 'state': '없음', 'tmp': '9'
    #     },
    #     'date': '20200214',
    #     'dust': {
    #         'PM10': {'value': 94, 'state': '나쁨'},
    #         'PM2.5': {'value': 71, 'state': '나쁨'},
    #         'code': '1'
    #     }
    # }
    result = "오늘의 미세먼지는 {}이며, 초미세먼지는 {}입니다.".format(data['dust']['PM10']['state'], data['dust']['PM2.5']['state'])
    # print(result)
    
    return result
# dust(sentence)