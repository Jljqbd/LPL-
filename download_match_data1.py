import requests
import json
import numpy as np
import pandas as pd
import openpyxl
import random
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
#response_url 为 你搜集这个数据的地址，match——id为比赛id在f12里找
class download_match_data():
    def __init__(self, response_url, match_id, teamsname, Jc, team_use_hero, Hc, Centroid_x, \
        Centroid_y, team_event, team_kill_event, dxdy, \
            win_Hc, win_Jc, win_fail_n_time, win_dxdy, fail_dxdy, team_win_fail_num, session, isread):
        global USER_AGENTS
        match_id = str(int(match_id))
        response_url = str(int(response_url))
        self.session = session
        self.response_url = response_url
        self.match_id = match_id
        self.url = "https://www.wanplus.com/ajax/matchdetail/"+ match_id + "?_gtk=1210991487"
        self.teamsname = teamsname #常量
        self.Jc = Jc # 17 * 5 * 5记录每个队伍每个位置之间的联动 shape不会改变
        self.team_use_hero = team_use_hero # shape改变
        self.Hc = Hc # 每个队伍每个英雄协作矩阵
        self.Centroid_x = Centroid_x
        self.Centroid_y = Centroid_y
        self.team_event = team_event
        self.team_kill_event = team_kill_event
        self.isread = isread
        self.dxdy = dxdy
        self.win_Hc = win_Hc
        self.win_Jc = win_Jc
        self.win_fail_n_time = win_fail_n_time
        self.win_dxdy = win_dxdy
        self.fail_dxdy = fail_dxdy
        self.team_win_fail_num = team_win_fail_num
        self.Df = np.zeros(len(teamsname)) # Direction forwward( 推进方向 )
        random_agent = USER_AGENTS[random.randint(0, len(USER_AGENTS)-1)]
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
    def download(self):
        data = self.session.get(self.url, headers = self.headers)
        json_data = json.loads(data.text)
        '''
        data = requests.get(url_2,headers = headers)
        json_data2 = json.loads(data.text)
        '''
        hero_player_dict={}
        '''
        拿到数据首先建立一个英雄：使用者的键值对字典（olaf：LNG.Xx）
        json_data2['data']['plStats']['damage']['blue']['teamAlias'] 蓝色方队伍名称
        json_data2['data']['plStats']['damage']['red']['teamAlias']
        json_data2['data']['plStats']['damage']['blue']['players']['playername'] 使用者名称与上面队伍名称结合
        json_data2['data']['plStats']['damage']['blue']['players']['cpherokey'] 使用英雄
        使用的位置顺序（上单，打野，中单，adc，辅助）
        '''
        hero_index_blue = {}
        hero_index_red = {}
        for i in ['blue','red']:
            teamname = json_data['data']['plStats']['damage'][i]['teamAlias']
            for j in range(5):
                play_data = json_data['data']['plStats']['damage'][i]['players'][j]
                hero_player_dict[play_data['cpherokey']] = teamname + "." + play_data['playername']
                if i =='blue':
                    hero_index_blue[play_data['cpherokey']] = j # 创建字典:英雄与矩阵中位置的键值对映射
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
        winner_index = self.teamsname.index(winner_name) # 胜利者队伍编号
        fail_index = self.teamsname.index(fail_name) # 失败者队伍编号
        
        self.team_win_fail_num[0][winner_index] += 1 #成功队伍胜场+1
        self.team_win_fail_num[1][fail_index] += 1 #失败队伍败场+1

        copy_team_Hc = self.Hc[winner_index] # 作为统计后的对比镜像，使用英雄的
        copy_team_Jc = self.Jc[winner_index] # 作为统计后的对比镜像，各个位置的
        # copy_dx = self.dxdy[0][winner_index] 
        # copy_dy = self.dxdy[1][winner_index]  # 作为统计后的对比镜像
        # print("比赛开始前dx："+str(copy_dx)+" dy:"+str(copy_dy) )
        '''                             ^dx                             ^dy
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
        red_event_number = 0 #r/b事件总数
        blue_event_number = 0
        red_team_name = json_data['data']['plStats']['damage']['red']['teamAlias']
        blue_team_name = json_data['data']['plStats']['damage']['blue']['teamAlias']
        red_Centroid_x = 0
        red_Centroid_y = 0
        red_index =  self.teamsname.index(red_team_name)
        blue_Centroid_x = 0
        blue_Centroid_y = 0
        blue_index =  self.teamsname.index(blue_team_name)
        excel_data_list = []
        self.Df[red_index] = -1
        self.Df[blue_index] = 1
        '''
        (0, 0)                  (0, 100)
        |---------------------|
        |              红色方  |
        |                     |
        |                     |
        |                     |
        |                     |
        |蓝色方                |
        |---------------------|
        (100, 0)             (100, 100)
        '''
        last_win_x = 0 #上一次事件的x
        last_win_y = 0 #上一次事件的y
        last_fail_x = 0
        last_fail_y = 0

        win_event_time = 0 # 胜利队伍事件发生的次数
        fail_event_time = 0 # 失败队伍事件发生的次数
        for i in range(len(json_data['data']['eventLine'])): # 那个时间段的
            for j in range(len(json_data['data']['eventLine'][i]['list'])): #这个时间段的这一个事件
                event_data = json_data['data']['eventLine'][i]['list']
                eventtime = event_data[j]['time']
                num_event_seconds = float(eventtime.split(':')[0])*60 + float(eventtime.split(':')[1]) # 事件发生事件转化为秒
                if event_data[j]['color'] == win_color:
                    win_event_time += 1
                if event_data[j]['color'] == fail_color:
                    fail_event_time += 1
                if win_event_time == 25:
                    self.win_fail_n_time[0][winner_index] += num_event_seconds
                    #print(eventtime+" ->"+str(num_event_seconds))
                if fail_event_time == 25:
                    self.win_fail_n_time[1][fail_index] += num_event_seconds
                    #print(eventtime+" ->"+str(num_event_seconds))
                eventername = hero_player_dict[event_data[j]['killerId']] if event_data[j]['killerId'] != None else ' '
                eventtype = event_data[j]['type'] # 事件类型
                eventmonstertype =  event_data[j]['monsterType'] if 'monsterType' in event_data[j] else ' '# 事件的子事件类型
                eventposition =  event_data[j]['position']['x'] + ","  +  event_data[j]['position']['y']
                eventParter = ""
                ass_len = len(event_data[j]['assistingParticipantIds']) if 'assistingParticipantIds' in event_data[j].keys() else 0
                # print("161:"+str(self.win_dxdy[0][9]))
                for k in range(ass_len):
                    eventParter = eventParter + hero_player_dict[ event_data[j]['assistingParticipantIds'][k] ] + ";"
                match_data = [self.match_id, match_number, eventtime, eventername, eventtype, eventmonstertype, eventposition, eventParter]
                # 比赛id;第几场;事件时间;事件发生方;事件类型;子事件类型;发生地点;助攻者
                excel_data_list.append(match_data)
                #
                #print("174:"+str(self.win_dxdy[0][9]))
                pos_x = float(event_data[j]['position']['x'])
                pos_y = float(event_data[j]['position']['y'])
                if event_data[j]['color'] == win_color:
                    # print("dxdy and win_dxdy add: dx:"+str((float(event_data[j]['position']['x']) - last_win_x)) +" dy:"+ str((float(event_data[j]['position']['y']) - last_win_y)))
                    self.dxdy[0][winner_index] += self.Df[winner_index] * ( pos_x - last_win_x )
                    #print("前"+str(self.win_dxdy[0][winner_index]))
                    self.win_dxdy[0][winner_index] += self.Df[winner_index] * ( pos_x - last_win_x )
                    #print("后"+str(self.win_dxdy[0][winner_index]))
                    self.dxdy[1][winner_index] += -1 * self.Df[winner_index] * ( pos_y - last_fail_x )
                    self.win_dxdy[1][winner_index] += -1 * self.Df[winner_index] * ( pos_y - last_fail_y )
                    last_win_x = float(event_data[j]['position']['x'])
                    last_win_y = float(event_data[j]['position']['y'])
                if event_data[j]['color'] == fail_color:
                    #print("dxdy and fail add:"+str((float(event_data[j]['position']['x']) - last_win_x)) + " dy:" + str((float(event_data[j]['position']['y']) - last_win_y)))
                    self.dxdy[0][fail_index] += self.Df[fail_index] * ( pos_x - last_fail_x )
                    self.fail_dxdy[0][fail_index] += self.Df[fail_index] * ( pos_x - last_fail_x )
                    self.dxdy[1][fail_index] += -1 * self.Df[fail_index] * ( pos_y - last_fail_y )
                    self.fail_dxdy[1][fail_index] += -1 * self.Df[fail_index] * ( pos_y - last_fail_y )
                    last_fail_x = float(event_data[j]['position']['x'])
                    last_fail_y = float(event_data[j]['position']['y'])
                ########################################
                #print("i:"+str(i)+" j:"+str(j)+" 190:"+str(self.win_dxdy[0][9]))

                #type:CHAMPION_KILL
                if event_data[j]['type']!='CHAMPION_KILL' or eventername ==' ': #如果不是击杀事件就跳过不统计
                    if eventername!=' ': #如果事件有队伍名称的话
                       self.team_event[self.teamsname.index(eventername.split('.')[0])] += 1 # 队伍事件数+1 
                    continue
                index_1 = self.teamsname.index(eventername.split('.')[0]) # 这一事件主体队伍第一维度
                # 有问题
                
                self.team_kill_event[index_1] += 1
                if eventername.split('.')[0]==red_team_name:
                    red_Centroid_x += float(event_data[j]['position']['x'])
                    red_Centroid_y += float(event_data[j]['position']['y'])
                    red_event_number += 1
                if eventername.split('.')[0]==blue_team_name:
                    blue_Centroid_x += float(event_data[j]['position']['x'])
                    blue_Centroid_y += float(event_data[j]['position']['y'])
                    blue_event_number += 1

                if event_data[j]['killerId'] in hero_index_blue.keys():
                    player_index = hero_index_blue
                    use_hero = blue_use_hero
                if event_data[j]['killerId'] in hero_index_red.keys():
                    player_index = hero_index_red
                    use_hero = red_use_hero
                event_hero = event_data[j]['assistingParticipantIds']
                event_hero.append(event_data[j]['killerId']) # 对event_hero列表里出现的英雄的相应矩阵位置值+1 
                diff_set = list( set(use_hero) - set(self.team_use_hero[index_1]) )
                if diff_set: # 如果diff_set为空则Hc大小不需要改变
                    self.team_use_hero[index_1] += diff_set
                    row_col = len(self.team_use_hero[index_1])
                    Hc_row_col = self.Hc[index_1].shape[0]
                    if Hc_row_col!=row_col:
                        new_array = np.zeros((row_col, row_col))
                        new_array[:Hc_row_col,:Hc_row_col] = self.Hc[ index_1 ]
                        new_list = []
                        for i1 in range(len(self.Hc)):
                            if i1 == index_1:
                                new_list.append(new_array)
                            else:
                                new_list.append(self.Hc[i1])
                        self.Hc = np.array(new_list) 
                    # self.Hc[index_1] = new_array # 添加新的行列 #错误形状不符合
                for event_hero_x in range(len(event_hero)):
                    for event_hero_y in range(len(event_hero)):
                        if event_hero_x >= event_hero_y:
                            continue
                        self.Jc[ index_1 ][ event_hero_x ][ event_hero_y ] += 1
                        self.Jc[ index_1 ][ event_hero_y ][ event_hero_x ] += 1
                        index_2 = self.team_use_hero[index_1].index(event_hero[event_hero_x])
                        index_3 = self.team_use_hero[index_1].index(event_hero[event_hero_y])
                        self.Hc[ index_1 ][ index_2 ][ index_3 ] += 1
                        self.Hc[ index_1 ][ index_3 ][ index_2 ] += 1
        r_c_x = red_Centroid_x/red_event_number if red_event_number != 0 else 0
        r_c_y = red_Centroid_y/red_event_number if red_event_number != 0 else 0
        b_c_x = blue_Centroid_y/blue_event_number if blue_event_number != 0 else 0 # 蓝色方坐标改为红色方坐标视角
        b_c_y = blue_Centroid_x/blue_event_number if blue_event_number != 0 else 0 # 蓝色放坐标改为红色方坐标视角

        new_win_Hc_body = np.zeros(self.Hc[winner_index].shape)
        new_copy_team_Hc = np.zeros(self.Hc[winner_index].shape)
        if self.win_Hc[winner_index].shape[0] !=0 and self.win_Hc[winner_index].shape[1] !=0:
            old_win_Hc_shape = self.win_Hc[winner_index].shape
            old_copy_team_Hc_shape = copy_team_Hc.shape
            # 老数据复制到新矩阵中
            new_win_Hc_body[:old_win_Hc_shape[0], :old_win_Hc_shape[1]] = self.win_Hc[winner_index] # 在更新win_Hc之前，存老win_Hc
            new_copy_team_Hc[:old_copy_team_Hc_shape[0], :old_copy_team_Hc_shape[1]] = copy_team_Hc # 操作之前的Hc
        new_win_Hc = [[] for i in range(len(self.teamsname))]
        new_win_Hc[:winner_index] = self.win_Hc[:winner_index]
        new_win_Hc[winner_index] = new_win_Hc_body + self.Hc[winner_index] - new_copy_team_Hc
        if winner_index != len(self.win_Hc) - 1: # 不能让winner访问下标溢出
            new_win_Hc[winner_index+1:] = self.win_Hc[winner_index+1:]
        # 胜利Hc矩阵 这场比赛后这支队伍的Hc - 这场比赛前这只队伍的Hc
        self.win_Hc = new_win_Hc
        self.win_Jc[winner_index] = self.Jc[winner_index] - copy_team_Jc


        self.Centroid_x[ red_index ].append( r_c_x )
        self.Centroid_y[ red_index ].append( r_c_y )
        self.Centroid_x[ blue_index ].append( b_c_x )
        self.Centroid_y[ blue_index ].append( b_c_y )
        if self.isread ==1:
            write('event.xlsx', excel_data_list)
            print(self.match_id +'写入完成\n')

