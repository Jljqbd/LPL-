import requests
import json
import numpy as np
import pandas as pd
import openpyxl
import random
from retrying import retry
def write(filename,excel_data):
    wb = openpyxl.load_workbook(filename)
    ws = wb['Sheet1']
    for x in excel_data:
        ws.append(x)
    wb.save(filename)
USER_AGENTS = [
 "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
 "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
 "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
 "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
 "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
 "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
 "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
 "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
 "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
 "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
 "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
 "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
 "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
 "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
 "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
 "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
]
#response_url ??? ?????????????????????????????????match??????id?????????id???f12??????
class download_match_data():
    def __init__(self, session, atd, i):
        
        global USER_AGENTS
        match_id = str(int(atd.match_id[i]))
        response_url = str(int(atd.response_url[i]))
        self.session = atd.session
        self.response_url = response_url
        self.match_id = match_id
        self.url = "https://www.wanplus.com/ajax/matchdetail/"+ match_id + "?_gtk=1210991487"
        self.atd = atd
        random_agent = USER_AGENTS[0]
        #
        headers = {
            'authority': 'www.wanplus.com',
            'method': 'GET',
            'path': '/ajax/matchdetail/'+ self.match_id +'?_gtk=1210991487',
            'scheme': 'https',
            'accept': 'application/json, text/javascript, */*; q=0.01',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh,en-US;q=0.9,en;q=0.8',
            'referer': 'https://www.wanplus.com/schedule/'+ response_url +'.html?matchid=' + self.match_id,
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': random_agent,
            'x-csrf-token': '1210991487',
            'x-requested-with': 'XMLHttpRequest',
        }
        self.headers = headers
    @retry    
    def download(self):
        data = self.session.get(self.url, headers = self.headers, timeout = 10000)

        json_data = json.loads(data.text)
        '''
        data = requests.get(url_2,headers = headers)
        json_data2 = json.loads(data.text)
        '''
        hero_player_dict={}
        '''
        ?????????????????????????????????????????????????????????????????????olaf???LNG.Xx???
        json_data2['data']['plStats']['damage']['blue']['teamAlias'] ?????????????????????
        json_data2['data']['plStats']['damage']['red']['teamAlias']
        json_data2['data']['plStats']['damage']['blue']['players']['playername'] ??????????????????????????????????????????
        json_data2['data']['plStats']['damage']['blue']['players']['cpherokey'] ????????????
        ???????????????????????????????????????????????????adc????????????
        '''
        hero_index_blue = {}
        hero_index_red = {}
        for i in ['blue','red']:
            teamname = json_data['data']['plStats']['damage'][i]['teamAlias']
            for j in range(5):
                play_data = json_data['data']['plStats']['damage'][i]['players'][j]
                hero_player_dict[play_data['cpherokey']] = teamname + "." + play_data['playername']
                if i =='blue':
                    hero_index_blue[play_data['cpherokey']] = j # ????????????:??????????????????????????????????????????
                else:
                    hero_index_red[play_data['cpherokey']] = j
            if i =='blue':
                blue_use_hero = [ i for i in hero_player_dict.keys()]
            if i =='red':
                red_use_hero = list( set([i for i in hero_player_dict.keys()])^set(blue_use_hero) )
            
        # winner_name = json_data['data']['red']['teamalias'] if json_data['data']['red']['teamid']==json_data['data']['info']['winner'] else json_data['data']['blue']['teamalias']
        if json_data['data']['red']['teamid']==json_data['data']['info']['winner']:
            winner_name = json_data['data']['red']['teamalias']
            fail_name = json_data['data']['blue']['teamalias']
            win_color = 'red'
            fail_color = 'blue'
        else:
            winner_name = json_data['data']['blue']['teamalias']
            fail_name = json_data['data']['red']['teamalias']
            win_color = 'blue'
            fail_color = 'red'
        winner_index = self.atd.teamsname.index(winner_name) # ?????????????????????
        fail_index = self.atd.teamsname.index(fail_name) # ?????????????????????
        
        self.atd.team_win_fail_num[0][winner_index] += 1 #??????????????????+1
        self.atd.team_win_fail_num[1][fail_index] += 1 #??????????????????+1

        copy_team_Hc = self.atd.Hc[winner_index] # ????????????????????????????????????????????????
        copy_team_Jc = self.atd.Jc[winner_index] # ????????????????????????????????????????????????
        # copy_dx = self.dxdy[0][winner_index] 
        # copy_dy = self.dxdy[1][winner_index]  # ??????????????????????????????
        # print("???????????????dx???"+str(copy_dx)+" dy:"+str(copy_dy) )
        '''                             ^dx                             ^dy
        mathch_id ??????id
        json_data1['data']['info']['matchorder'] ??????bo3/bo5????????????
        '''
        match_number = json_data['data']['info']['matchorder']
        '''
        json_data1['data']['blue']
        json_data1['data']['eventLine'][0???__len__-1]['list']['0???__len__-1']['killerId'] ????????????????????????????????????????????????????????????????????????????????????
        json_data1['data']['eventLine'][0???__len__-1]['list']['0???__len__-1']['victimId'] ??????????????????????????????
        json_data1['data']['eventLine'][0???__len__-1]['list']['0???__len__-1']['type'] ????????????
        json_data1['data']['eventLine'][0???__len__-1]['list']['0???__len__-1']['monsterType'] ??????????????????
        json_data1['data']['eventLine'][0???__len__-1]['list']['0???__len__-1']['position']['x']/['y'] ???????????????
        json_data1['data']['eventLine'][0???__len__-1]['list']['0???__len__-1']['assistingParticipantIds'] ????????????????????????????????????????????????????????????????????????
        json_data1['data']['eventLine'][0???__len__-1]['list']['0???__len__-1']['time'] ????????????
        '''
        # ======================================
        # ???????????????
        red_event_number = 0 #r/b????????????
        blue_event_number = 0
        red_team_name = json_data['data']['plStats']['damage']['red']['teamAlias']
        blue_team_name = json_data['data']['plStats']['damage']['blue']['teamAlias']
        red_Centroid_x = 0
        red_Centroid_y = 0
        red_index =  self.atd.teamsname.index(red_team_name)
        blue_Centroid_x = 0
        blue_Centroid_y = 0
        blue_index =  self.atd.teamsname.index(blue_team_name)
        excel_data_list = []
        #==============================================================
        '''
        (0, 0)                  (0, 100)
        |---------------------|
        |              ?????????  |
        |                     |
        |                     |
        |                     |
        |                     |
        |?????????                |
        |---------------------|
        (100, 0)             (100, 100)
        '''
        last_win_x = 0 #??????????????????x
        last_win_y = 0 #??????????????????y
        last_fail_x = 0
        last_fail_y = 0

        win_event_time = 0 # ?????????????????????????????????
        fail_event_time = 0 # ?????????????????????????????????
        for i in range(len(json_data['data']['eventLine'])): # ??????????????????
            for j in range(len(json_data['data']['eventLine'][i]['list'])): #?????????????????????????????????
                event_data = json_data['data']['eventLine'][i]['list']
                eventtime = event_data[j]['time']
                num_event_seconds = float(eventtime.split(':')[0])*60 + float(eventtime.split(':')[1]) # ??????????????????????????????
                if event_data[j]['color'] == win_color:
                    win_event_time += 1
                if event_data[j]['color'] == fail_color:
                    fail_event_time += 1
                if win_event_time == 25:
                    self.atd.win_fail_n_time[0][winner_index] += num_event_seconds
                    #print(eventtime+" ->"+str(num_event_seconds))
                if fail_event_time == 25:
                    self.atd.win_fail_n_time[1][fail_index] += num_event_seconds
                    #print(eventtime+" ->"+str(num_event_seconds))
                eventername = hero_player_dict[event_data[j]['killerId']] if event_data[j]['killerId'] != None else ' '
                eventtype = event_data[j]['type'] # ????????????
                eventmonstertype =  event_data[j]['monsterType'] if 'monsterType' in event_data[j] else ' '# ????????????????????????
                eventposition =  event_data[j]['position']['x'] + ","  +  event_data[j]['position']['y']
                eventParter = ""
                eventHeros = ""
                ass_len = len(event_data[j]['assistingParticipantIds']) if 'assistingParticipantIds' in event_data[j].keys() else 0
                # print("161:"+str(self.win_dxdy[0][9]))
                for k in range(ass_len):
                    eventParter = eventParter + hero_player_dict[ event_data[j]['assistingParticipantIds'][k] ] + ";"
                    eventHeros = eventHeros + event_data[j]['assistingParticipantIds'][k] + ";"
                if event_data[j]['killerId'] != None:
                    eventHeros += event_data[j]['killerId']
                match_data = [self.match_id, match_number, eventtime, eventername, eventtype, eventmonstertype, eventposition, eventParter, eventHeros]
                # ??????id;?????????;????????????;???????????????;????????????;???????????????;????????????;?????????
                excel_data_list.append(match_data)
                #
                #print("174:"+str(self.win_dxdy[0][9]))
                pos_x = float(event_data[j]['position']['x'])
                pos_y = float(event_data[j]['position']['y'])
                if event_data[j]['color'] == 'red': # ?????????????????????????????????????????????
                    pos_x, pos_y = pos_y, pos_x
                if event_data[j]['color'] == win_color:
                    # print("dxdy and win_dxdy add: dx:"+str((float(event_data[j]['position']['x']) - last_win_x)) +" dy:"+ str((float(event_data[j]['position']['y']) - last_win_y)))
                    self.atd.dxdy[0][winner_index] += ( pos_x - last_win_x )
                    #print("???"+str(self.win_dxdy[0][winner_index]))
                    self.atd.win_dxdy[0][winner_index] += ( pos_x - last_win_x )
                    #print("???"+str(self.win_dxdy[0][winner_index]))
                    self.atd.dxdy[1][winner_index] += ( pos_y - last_fail_x )
                    self.atd.win_dxdy[1][winner_index] += ( pos_y - last_fail_y )
                    if eventtype == 'CHAMPION_KILL' and event_data[j]['killerId'] not in self.atd.remove_hero:
                        self.atd.remove_dxdy[0][winner_index] += ( pos_x - last_win_x )
                        self.atd.remove_dxdy[1][winner_index] += ( pos_y - last_fail_x )
                    last_win_x = pos_x
                    last_win_y = pos_y
                if event_data[j]['color'] == fail_color:
                    #print("dxdy and fail add:"+str((float(event_data[j]['position']['x']) - last_win_x)) + " dy:" + str((float(event_data[j]['position']['y']) - last_win_y)))
                    self.atd.dxdy[0][fail_index] += ( pos_x - last_fail_x )
                    self.atd.fail_dxdy[0][fail_index] += ( pos_x - last_fail_x )
                    self.atd.dxdy[1][fail_index] += ( pos_y - last_fail_y )
                    self.atd.fail_dxdy[1][fail_index] += ( pos_y - last_fail_y )
                    if eventtype == 'CHAMPION_KILL' and event_data[j]['killerId'] not in self.atd.remove_hero:
                        self.atd.remove_dxdy[0][fail_index] += ( pos_x - last_fail_x )
                        self.atd.remove_dxdy[1][fail_index] += ( pos_y - last_fail_y )
                    last_fail_x = pos_x
                    last_fail_y = pos_y

                #type:CHAMPION_KILL
                if event_data[j]['type']!='CHAMPION_KILL' or eventername ==' ': #??????????????????????????????????????????
                    if eventername!=' ': #?????????????????????????????????
                       self.atd.team_event[self.atd.teamsname.index(eventername.split('.')[0])] += 1 # ???????????????+1 
                    continue
                index_1 = self.atd.teamsname.index(eventername.split('.')[0]) # ????????????????????????????????????
                
                self.atd.team_kill_event[index_1] += 1
                if eventername.split('.')[0]==red_team_name:
                    red_Centroid_x += pos_x
                    red_Centroid_y += pos_y
                    red_event_number += 1
                if eventername.split('.')[0]==blue_team_name:
                    blue_Centroid_x += pos_x
                    blue_Centroid_y += pos_y
                    blue_event_number += 1

                if event_data[j]['killerId'] in hero_index_blue.keys():
                    player_index = hero_index_blue
                    use_hero = blue_use_hero
                    if event_data[j]['killerId'] not in self.atd.remove_hero:#???????????????????????????????????????????????????+1
                        self.atd.team_remove_kill_event[blue_index] += 1
                if event_data[j]['killerId'] in hero_index_red.keys():
                    player_index = hero_index_red
                    use_hero = red_use_hero 
                    if event_data[j]['killerId'] not in self.atd.remove_hero:#???????????????????????????????????????????????????+1
                        self.atd.team_remove_kill_event[red_index] += 1
                event_hero = event_data[j]['assistingParticipantIds']
                event_hero.append(event_data[j]['killerId']) # ???event_hero????????????????????????????????????????????????+1 
                diff_set = list( set(use_hero) - set(self.atd.team_use_hero[index_1]) )
                if diff_set: # ??????diff_set?????????Hc?????????????????????
                    self.atd.team_use_hero[index_1] += diff_set
                    row_col = len(self.atd.team_use_hero[index_1])
                    Hc_row_col = self.atd.Hc[index_1].shape[0]
                    if Hc_row_col!=row_col:
                        new_array = np.zeros((row_col, row_col))
                        new_array[:Hc_row_col,:Hc_row_col] = self.atd.Hc[ index_1 ]
                        new_list = []
                        for i1 in range(len(self.atd.Hc)):
                            if i1 == index_1:
                                new_list.append(new_array)
                            else:
                                new_list.append(self.atd.Hc[i1])
                        self.atd.Hc = np.array(new_list) 
                    # self.Hc[index_1] = new_array # ?????????????????? #?????????????????????
                for event_hero_x in range(len(event_hero)):
                    for event_hero_y in range(len(event_hero)):
                        if event_hero_x >= event_hero_y:
                            continue
                        self.atd.Jc[ index_1 ][ event_hero_x ][ event_hero_y ] += 1
                        self.atd.Jc[ index_1 ][ event_hero_y ][ event_hero_x ] += 1
                        if event_data[j]['killerId'] not in self.atd.remove_hero:
                            self.atd.remove_Jc[ index_1 ][ event_hero_x ][ event_hero_y ] += 1
                            self.atd.remove_Jc[ index_1 ][ event_hero_y ][ event_hero_x ] += 1
                        index_2 = self.atd.team_use_hero[index_1].index(event_hero[event_hero_x])
                        index_3 = self.atd.team_use_hero[index_1].index(event_hero[event_hero_y])
                        self.atd.Hc[ index_1 ][ index_2 ][ index_3 ] += 1
                        self.atd.Hc[ index_1 ][ index_3 ][ index_2 ] += 1
        r_c_x = red_Centroid_x/red_event_number if red_event_number != 0 else 0
        r_c_y = red_Centroid_y/red_event_number if red_event_number != 0 else 0
        b_c_x = blue_Centroid_y/blue_event_number if blue_event_number != 0 else 0 # ??????????????????????????????????????????
        b_c_y = blue_Centroid_x/blue_event_number if blue_event_number != 0 else 0 # ??????????????????????????????????????????

        new_win_Hc_body = np.zeros(self.atd.Hc[winner_index].shape)
        new_copy_team_Hc = np.zeros(self.atd.Hc[winner_index].shape)
        if self.atd.win_Hc[winner_index].shape[0] !=0 and self.atd.win_Hc[winner_index].shape[1] !=0:
            old_win_Hc_shape = self.atd.win_Hc[winner_index].shape
            old_copy_team_Hc_shape = copy_team_Hc.shape
            # ??????????????????????????????
            new_win_Hc_body[:old_win_Hc_shape[0], :old_win_Hc_shape[1]] = self.atd.win_Hc[winner_index] # ?????????win_Hc???????????????win_Hc
            new_copy_team_Hc[:old_copy_team_Hc_shape[0], :old_copy_team_Hc_shape[1]] = copy_team_Hc # ???????????????Hc
        new_win_Hc = [[] for i in range(len(self.atd.teamsname))]
        new_win_Hc[:winner_index] = self.atd.win_Hc[:winner_index]
        new_win_Hc[winner_index] = new_win_Hc_body + self.atd.Hc[winner_index] - new_copy_team_Hc
        if winner_index != len(self.atd.win_Hc) - 1: # ?????????winner??????????????????
            new_win_Hc[winner_index+1:] = self.atd.win_Hc[winner_index+1:]
        # ??????Hc?????? ??????????????????????????????Hc - ??????????????????????????????Hc
        self.atd.win_Hc = new_win_Hc
        self.atd.win_Jc[winner_index] = self.atd.Jc[winner_index] - copy_team_Jc


        self.atd.Centroid_x[ red_index ].append( r_c_x )
        self.atd.Centroid_y[ red_index ].append( r_c_y )
        self.atd.Centroid_x[ blue_index ].append( b_c_x )
        self.atd.Centroid_y[ blue_index ].append( b_c_y )
        if self.atd.isread ==1:
            write('event.xlsx', excel_data_list)
            print(self.match_id +'????????????\n')

