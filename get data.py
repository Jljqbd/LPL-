import requests
import json
import numpy as np
import pandas as pd
import openpyxl
def write(filename, excel_data):
   wb = openpyxl.load_workbook(filename)
   ws = wb['Sheet1']
   for x in excel_data:
      ws.append(x)
   wb.save(filename)
match_id = "65612" # 第一场比赛id
response_url = "58617"
url_1 = "https://www.wanplus.com/ajax/matchdetail/"+ match_id + "?_gtk=2083301385"
   #url_2 = "https://www.wanplus.com/ajax/matchdetail/"+ match_id2 + "?_gtk=2083301385"
headers={
    
   'Accept': 'application/json, text/javascript, */*; q=0.01', 
   'Accept-Encoding': 'gzip, deflate, br',
   'Accept-Language': 'zh-CN',
   'Cache-Control': 'max-age=0',
   'Cookie': 'isShown=1; CNZZDATA1275078652=1283438632-1586077904-%7C1586480302; wanplus_csrf=_csrf_tk_2016192521; wp_info=ssid=s6167248945; Hm_lpvt_f69cb5ec253c6012b2aa449fb925c1c2=1586482565; gameType=2; wanplus_token=592861eee94fb1d4715ed99f21a452ab; wp_pvid=1037503182; wanplus_storage=lf4m67eka3o; UM_distinctid=17149e13af7ea-0d8dd2eb38dccf-71415a3b-100200-17149e13af82a0; Hm_lvt_f69cb5ec253c6012b2aa449fb925c1c2=1586098008,1586231131,1586479603,1586481689; wanplus_sid=0cd1006fb6ba84fee9f5c544442ab265',
   'Host': 'www.wanplus.com',
   'Referer': 'https://www.wanplus.com/schedule/'+ response_url +'.html',
   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.18362',
   'X-CSRF-Token': '2083301385',
   'X-Requested-With': 'XMLHttpRequest',
}
'''
   cookie固定，爬取不同页面只需要更改
   referer：即网页地址，
   url：请求url， 
   可能需要给更改x-csrf-token
'''
data = requests.get(url_1,headers = headers)
json_data = json.loads(data.text)
'''
   data = requests.get(url_2,headers = headers)
   json_data2 = json.loads(data.text)
'''
hero_player_dict={}
'''
   拿到数据首先建立一个英雄：使用者的键值对字典（olaf：LNG.Xx）
   json_data2['data']['plStats']['damage']['blue']['teamAlians'] 蓝色方队伍名称
   json_data2['data']['plStats']['damage']['red']['teamAlians']
   json_data2['data']['plStats']['damage']['blue']['players']['playername'] 使用者名称与上面队伍名称结合
   json_data2['data']['plStats']['damage']['blue']['players']['cpherokey'] 使用英雄
   使用的位置顺序（上单，打野，中单，adc，辅助）
'''
for i in ['blue','red']:
   for j in range(5):
      teamname = json_data['data']['plStats']['damage'][i]['teamAlias']
      play_data = json_data['data']['plStats']['damage'][i]['players'][j]
      hero_player_dict[play_data['cpherokey']] = teamname + "." + play_data['playername']
'''
   mathch_id 比赛id
   json_data1['data']['info']['matchorder'] 这个bo3/bo5的第几场
'''
match_number = json_data['data']['info']['matchorder']
'''
   json_data1['data']['blue']
   json_data1['data']['eventLine'][0到__len__-1]['list']['0到__len__-1']['killerId'] 击杀者英雄（使用上面创建的字典来获取击杀者的名称与队伍）
   json_data1['data']['eventLine'][0到__len__-1]['list']['0到__len__-1']['victimId'] 被击杀者英雄（同上）
   json_data1['data']['eventLine'][0到__len__-1]['list']['0到__len__-1']['type'] 事件类型
   json_data1['data']['eventLine'][0到__len__-1]['list']['0到__len__-1']['monsterType'] 具体事件概述
   json_data1['data']['eventLine'][0到__len__-1]['list']['0到__len__-1']['position']['x']/['y'] 发生的地点
   json_data1['data']['eventLine'][0到__len__-1]['list']['0到__len__-1']['assistingParticipantIds'] 助攻英雄（与上面的字典来获取助攻者的名称与队伍）
   json_data1['data']['eventLine'][0到__len__-1]['list']['0到__len__-1']['time'] 发生时间
'''
excel_data_list = []
for i in range(len(json_data['data']['eventLine'])): # 那个时间段的
   for j in range(len(json_data['data']['eventLine'][i]['list'])): #这个时间段的哪个事件
      event_data = json_data['data']['eventLine'][i]['list']
      eventtime = event_data[j]['time']
      eventername = hero_player_dict[event_data[j]['killerId']] if event_data[j]['killerId'] != None else ' '
      eventtype = event_data[j]['type'] # 事件类型
      eventmonstertype =  event_data[j]['monsterType'] if 'monsterType' in event_data[j] else ' '# 事件的子事件类型
      eventposition =  event_data[j]['position']['x'] + ","  +  event_data[j]['position']['y']
      eventParter = ""
      for k in range(len(event_data[j]['assistingParticipantIds'])):
         eventParter = eventParter + hero_player_dict[ event_data[j]['assistingParticipantIds'][k] ] + ";"
      match_data = [match_id, match_number, eventtime, eventername, eventtype, eventmonstertype, eventposition, eventParter]
      # 比赛id;第几场;事件时间;事件发生方;事件类型;子事件类型;发生地点;助攻者
      excel_data_list.append(match_data)
write('event.xlsx', excel_data_list)
print('写入完成')
