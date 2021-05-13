import requests
import json
import numpy as np
import pandas as pd
import openpyxl
import random


class ALLTeamData():
    def __init__(self, session, remove_hero, response_url_list, match_id_list, teamsname, isread):
        '''
        session : 保存爬取状态
        remove_hero : 移除的频繁集英雄
        response_url_list: 构造header用的
        match_id_list : matchid用于构造url
        teamname : 队伍名称
        isread : 是否在download_matchid_data中将每次爬取的数据写入excel
        '''
        team_len = len(teamsname)
        dx = np.zeros(team_len)
        dy = np.zeros(team_len)
        remove_dx = np.zeros(team_len)
        remove_dy = np.zeros(team_len)
        team_win_num = np.zeros(len(teamsname))
        team_fail_num = np.zeros(len(teamsname)) # 大小相同
        team_len = len(teamsname)
        self.session = session # 爬虫sessio,保存爬取状态
        self.response_url = response_url_list # 用于在header中的refer部分
        self.match_id = match_id_list # 构造请求url
        self.teamsname = teamsname #队伍名称
        self.Jc = np.zeros((team_len,5,5)) # Job collaboration  17 * 5 * 5记录每个队伍每个位置之间的联动 shape不会改变
        # 5 * 5 : 上单，打野，中路，adc，辅助
        self.team_use_hero = [ [] for i in range(team_len)] # shape会随着每次运行改变
        self.Hc = np.array([ [] for i in range(team_len)]) # 每个队伍每个英雄协作矩阵 Hero collaboration
        self.Centroid_x = [ [] for i in range(team_len) ] # 所有队伍整个赛季的平均x
        self.Centroid_y = [ [] for i in range(team_len) ] # 所有队伍整个赛季的平均y
        self.team_event = np.zeros(team_len) # 队伍事件数
        self.team_kill_event = np.zeros(team_len) #队伍击杀事件数
        self.team_remove_kill_event = np.zeros(team_len) # 去除掉自定义的频繁集英雄的击杀事件数
        self.isread = isread # 是否写入excel表
        self.dxdy = [dx, dy] # sum(thisevent(x) - lastevent(x)) and sum(thisevent(y) - lastevent(y))
        self.remove_dxdy = [remove_dx, remove_dy] # 去除掉自定义的频繁集英雄的dxdy
        self.win_Hc = np.array([ [] for i in range(team_len)]) # 胜利场次的英雄的协作矩阵
        self.win_Jc = np.zeros((team_len,5,5)) # 胜利场次的5*5位置的协作矩阵
        self.win_fail_n_time = [np.zeros(len(teamsname)), np.zeros(len(teamsname))] # list长度为2,list[0] 是所有成功的对局的所用总时间
        self.win_dxdy = [np.zeros(team_len), np.zeros(team_len)]  # 所有队伍成功场次dxdy
        self.fail_dxdy = [np.zeros(team_len), np.zeros(team_len)] # 所有队伍失败场次dxdy
        self.remove_hero = remove_hero # 自定义的频繁集英雄集合
        self.team_win_fail_num = [team_win_num, team_fail_num] # 所有队伍的成功与失败场次,list[0]成功,list[1]失败的
        self.remove_Jc = np.zeros((team_len,5,5)) # 移除频繁集英雄后JC,Hc不用直接删除对应的行和列即可
        self.team_remove_event = np.zeros(team_len) # 每支队伍去除缺省英雄后的事件数字
        self.Centroid_remove_x = [ [] for i in range(team_len) ] #去掉节点后的平均x
        self.Centroid_remove_y = [ [] for i in range(team_len) ] #去除节点后的平均y
        # 将三种中心性与平均x,y,成功场次,事件数,击杀数,成功游戏中对局时间,dy/dx的相关系数