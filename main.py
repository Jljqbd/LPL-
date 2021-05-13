import numpy as np
import xlrd
import sys
sys.path.append('E:/VSC_project/VSC py/LPL/aripor/')
from download_match_data import *
from Network_Indicators import *
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import requests
from lxml import etree
team_P = 1 # 研究的队伍
session = requests.session()
'''
match_id = "59550" # 第一场比赛id
response_url = "54681"
'''
##赋值的时候不能简便赋值，列表变量之间赋值的是传址，即a=[1,2,3] b=a，则a变化则b也随之变化
'''
def read_match_data(path):
    book = xlrd.open_workbook(path,encoding_override="utf-8")
    sheet = book.sheet_by_name(u'Sheet2')#通过名称获取
    row = sheet.nrows
    matchdata =[]
    for i in range(row):
        matchdata.append(sheet.row_values(i))
    matchdata = np.array(matchdata)
    response_url_list = matchdata[:,0]
    match_id_list = matchdata[:,1]
    return response_url_list, match_id_list
'''
def read_match_data_2():
    global session
    headers = {
        'authority': 'www.wanplus.com',
        'method': 'GET',
        'path': '/lol/video/pro?=817',
        'scheme': 'https',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'zh,en-US;q=0.9,en;q=0.8',
        'cache-control':'max-age=0',
        'cookie':'wanplus_token=19121e79eb8e27c5b1d3343913d7842b; wanplus_storage=lf4m67eka3o; wanplus_sid=ac3d612f62c41aa2dbf94cc8df23a4cb; UM_distinctid=172f13cea921ae-09188750dc4708-79657967-100200-172f13cea9324f; wp_pvid=290161117; isShown=1; gameType=2; wanplus_csrf=_csrf_tk_1210991487; CNZZDATA1275078652=1682856499-1593184647-%7C1593249135; wp_info=ssid=s8405666614; Hm_lvt_f69cb5ec253c6012b2aa449fb925c1c2=1593185201,1593253838; Hm_lpvt_f69cb5ec253c6012b2aa449fb925c1c2=1593253838',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36 Edg/83.0.478.56',
    }
    url = "https://www.wanplus.com/lol/video/pro?eid=817"
    r = session.get(url, headers = headers)
    webstr = r.text
    html = etree.HTML(webstr)
    html_data = html.xpath("/html/body/div/div/div/div/div/div/ul/li/a/@href")
    response_url_list = []
    match_id_list = []
    for line in html_data:
        response_url_list.append(line[10:15])
        match_id_list.append(line.split("matchid=")[-1])
    return response_url_list, match_id_list
'''
match_id_list = ["65528","65529","65535"]
response_url_list = ["58610","58610","58610"]
'''

path = "E:\\VSC_project\\VSC py\\LPL\\match_data.xls"
response_url_list, match_id_list = read_match_data_2()

teamsname = ['FPX','RNG','TES','EDG','BLG','LNG','WE','SN','JDG','DMO','IG','VG','RW','LGD','V5','OMG']
team_len = len(teamsname)
team_event = np.zeros(team_len) # 每支队伍的事件数
team_kill_event = np.zeros(team_len) #每支队伍的击杀次数
dx = np.zeros(team_len)
dy = np.zeros(team_len)
dxdy = [dx, dy]
Centroid_x = [ [] for i in range(team_len) ]
Centroid_y = [ [] for i in range(team_len) ]
team_use_hero =[ [] for i in range(team_len)] # 每支队伍使用过的英雄 作为Hc矩阵的辅助变量
Hc = [ [] for i in range(team_len)] # Hero collaboration
Hc = np.array(Hc)
Jc = np.zeros((team_len,5,5)) # Job collaboration
# 5 * 5 : 上单，打野，中路，adc，辅助

team_win_num = np.zeros(len(teamsname))
team_fail_num = np.zeros(len(teamsname)) # 大小相同
team_win_fail_num = [team_win_num, team_fail_num]
win_Hc = np.array([ [] for i in range(team_len)])
win_Jc = np.zeros((team_len,5,5))
win_fail_n_time = [np.zeros(len(teamsname)), np.zeros(len(teamsname))] # 
win_dxdy = [np.zeros(team_len), np.zeros(team_len)] # 大小与dxdy相等,且  失败dxdy+成功dxdy=dxdy 因为dxdy中存储的是总值
fail_dxdy =[np.zeros(team_len), np.zeros(team_len)]
# len(match_id_list) 为这个赛季的比赛场次
isread = 0
for i in range(len(match_id_list)):
    if i ==0:
        dmd = download_match_data(response_url_list[i], match_id_list[i], teamsname, Jc, \
            team_use_hero, Hc, Centroid_x, Centroid_y, team_event, team_kill_event, dxdy, \
                win_Hc, win_Jc, win_fail_n_time, win_dxdy, fail_dxdy,  team_win_fail_num, session, isread)
    else:
        if i%40==0:
            session = requests.session()
            read_match_data_2()
            dmd.session = session
        dmd = download_match_data(response_url_list[i], match_id_list[i], dmd.teamsname, dmd.Jc, \
            dmd.team_use_hero, dmd.Hc, dmd.Centroid_x, dmd.Centroid_y, dmd.team_event, \
                dmd.team_kill_event, dmd.dxdy, dmd.win_Hc, dmd.win_Jc, dmd.win_fail_n_time, \
                    dmd.win_dxdy, dmd.fail_dxdy, dmd.team_win_fail_num, dmd.session, dmd.isread)
    dmd.download()
    print(str(((i+1)/(len(match_id_list)+1))*100) + ' %')
# 变量名简化    
team_use_hero = dmd.team_use_hero
Hc = dmd.Hc
Jc = dmd.Jc
win_Jc = dmd.win_Jc
win_Hc = dmd.win_Hc
fail_Jc = Jc - win_Jc
fail_Hc = [[] for i in range(len(teamsname))]
for i in range(len(teamsname)):
    fail_Hc[i] = np.zeros(Hc[i].shape)
copy_win_Hc = fail_Hc
# 有可能有队伍最后一次比赛为负，且用到了win_Hc中没有用到的英雄,所以可能会导致Hc和win_Hc大小不匹配
for i in range(len(Hc)):
    if Hc[i].shape[0]!=win_Hc[i].shape[0]: #如果这支队伍有上述情况，win_Hc进行扩充
        new_win_Hc = np.zeros(Hc[i].shape)
        new_win_Hc[:win_Hc[i].shape[0], :win_Hc[i].shape[1]] = win_Hc[i]
        copy_win_Hc[i] = new_win_Hc
        continue
    copy_win_Hc[i] = win_Hc[i]
win_Hc = copy_win_Hc
fail_Hc = Hc - win_Hc
Centroid_x = dmd.Centroid_x
Centroid_y = dmd.Centroid_y

win_dx = dmd.win_dxdy[0]
win_dy = dmd.win_dxdy[1]
fail_dx = dmd.fail_dxdy[0]
fail_dy = dmd.fail_dxdy[1]

fail_dxdy = fail_dx / fail_dy
win_dxdy = win_dx / win_dy
win_n_time = win_fail_n_time[0] // 60 + (win_fail_n_time[0] % 60) / 100
fail_n_time = win_fail_n_time[1] // 60 + (win_fail_n_time[1] % 60) / 100

team_win_num = dmd.team_win_fail_num[0]
team_fail_num = dmd.team_win_fail_num[1]
#
team_mean_x = np.zeros(team_len)
team_mean_y = np.zeros(team_len)

create_shortpath_graph(team_use_hero[team_P], Hc[team_P], 1)
plt.figure()
write_excel_data = []
team_data, avg_data = num_event(dmd.team_event, team_P, teamsname, "Number of events", 421, "A") #第team_P个队伍和所有队伍事件数
write_excel_data.append(["Number of events", team_data, avg_data])
team_data, avg_data = num_event(dmd.team_kill_event, team_P, teamsname, "Number of kill events",422, "B") # 第team_P个队伍和所有队伍的击杀数
write_excel_data.append(["Number of kill events", team_data, avg_data])
for i in range(team_len):
    team_mean_x[i] = np.array(Centroid_x[i]).mean() # Centroid_x[i] 保存的是第i个队伍这个赛季每场比赛的x,所以取得的是这个队伍这个赛季的x_mean
    team_mean_y[i] = np.array(Centroid_y[i]).mean() # Centroid_y[i] 同上

team_data, avg_data = num_event(team_mean_x, team_P, teamsname, "<X>", 423, "C")
write_excel_data.append(["<X>", team_data, avg_data])
team_data, avg_data = num_event(team_mean_y, team_P, teamsname, "<Y>", 424, "D")
write_excel_data.append(["<Y>", team_data, avg_data])
DxDy = dmd.dxdy[0] / dmd.dxdy[1]
team_data, avg_data = num_event(dmd.dxdy[0], team_P, teamsname, "Dx", 425, "E")
write_excel_data.append(["Dx", team_data, avg_data])
team_data, avg_data = num_event(dmd.dxdy[1], team_P, teamsname, "Dy", 426, "F")
write_excel_data.append(["Dy", team_data, avg_data])
team_data, avg_data = num_event(DxDy, team_P, teamsname, "Dx / Dy", 414, "G") # 0 < DxDy < 1 说明此队伍侧重于上半区
write_excel_data.append(["Dx / Dy", team_data, avg_data])
plt.show()
team_C = []
team_hero_short_path = []
team_eig = []
team_AC = []
for i in range(team_len):
    team_C.append(C(Jc[i])) # job list
    # team_hero_short_path.append(avg_short_path_lenth(Hc[i])) # hero list
    team_eig.append(graph_max_eig(Hc[i])) # hero list
    team_hero_short_path.append(avg_short_path_length(create_shortpath_graph(team_use_hero[i], Hc[i])))
    team_AC.append(Ac(Hc[i]))
plt.figure()
team_data, avg_data = num_event(team_C, team_P, teamsname, "Clustering coefficient", 221, "A")
write_excel_data.append(["Clustering coefficient", team_data, avg_data])
team_data, avg_data = num_event(team_hero_short_path, team_P, teamsname, "Shortest path", 222,"B")
write_excel_data.append(["Shortest path", team_data, avg_data])
team_data, avg_data = num_event(team_AC, team_P, teamsname, "Algebraic connectivity", 223, "C")
write_excel_data.append(["Algebraic connectivity", team_data, avg_data])
team_data, avg_data = num_event(team_eig, team_P, teamsname, "Max eig", 224, "D")
write_excel_data.append(["Max eig", team_data, avg_data])
plt.show()
#write_excel(write_excel_data, 1) #
# 1) 胜利和失败时到达40个事件的平均时间
# 2) 胜利和失败时<dX>/<Dy>
plt.figure()
scatter_plot(fail_dxdy, win_dxdy, team_P, "defeat <DX>/<DY>", "win <DX>/<DY>", 121, "A")
scatter_plot(fail_n_time / team_fail_num, win_n_time / team_win_num, team_P, " Time to defeat 25 events", "Time to win 25 events", 122, "B", 1, [1.05, 0, 3, 0])
plt.show()
# 1) 胜利和失败时的聚类系数(C)
win_C = []
fail_C =[]
# 2) 胜利和失败时的最短路(avg_short_path_length)
win_short_path = []
fail_short_path = []
# 3) 胜利和失败时的最大特征值(graph_max_eig)
win_max_eig = []
fail_max_eig = []
# 4) 胜利和失败的代数连通性(Ac)
win_Ac = []
fail_Ac = []
# 5) 胜利和失败的度中心性(degree)
win_degree = []
fail_degree = []
# 6) 胜利和失败的紧密中心性(closeness)
win_closeness = []
fail_closeness = []
# 7) 胜利和失败的间接中心性(betweenness)
win_betweenness = []
fail_betweenness = []
# 8) 胜利和失败时的特征向量中心度(eigenvector)
win_eigenvector = []
fail_eigenvector = []
for i in range(team_len):
    hero_win_shortpath_graph = create_shortpath_graph(team_use_hero[i], win_Hc[i])
    hero_win_node_graph = create_G(team_use_hero[i], win_Hc[i])
    hero_fail_shortpath_graph = create_shortpath_graph(team_use_hero[i], fail_Hc[i])
    hero_fail_node_graph = create_G(team_use_hero[i], fail_Hc[i])
    win_C.append(C(win_Hc[i]))
    fail_C.append(C(fail_Hc[i]))
    win_short_path.append(avg_short_path_length(hero_win_shortpath_graph))
    fail_short_path.append(avg_short_path_length(hero_fail_shortpath_graph))
    win_max_eig.append(graph_max_eig(win_Hc[i]))
    fail_max_eig.append(graph_max_eig(fail_Hc[i]))
    win_Ac.append(Ac(win_Hc[i]))
    fail_Ac.append(Ac(fail_Hc[i]))
    win_degree.append(degree(hero_win_node_graph))
    fail_degree.append(degree(hero_fail_node_graph))
    win_closeness.append(closeness(hero_win_shortpath_graph))
    fail_closeness.append(closeness(hero_fail_shortpath_graph))
    win_betweenness.append(betweenness(hero_win_node_graph))
    fail_betweenness.append(betweenness(hero_fail_node_graph))
    win_eigenvector.append(eigenvector(hero_win_node_graph))
    fail_eigenvector.append(eigenvector(hero_fail_node_graph))
plt.figure()
scatter_plot(fail_C, win_C, team_P, "defeat clustering coefcient C,", "win clustering coefcient C,", 241, "A")
scatter_plot(fail_short_path, win_short_path, team_P, "defeat average shortest-path d", "win average shortest-path d", 242, "B")
scatter_plot(fail_max_eig, win_max_eig, team_P, "defeat largest eigenvalue", "win largest eigenvalue", 243, "C")
scatter_plot(fail_Ac, win_Ac, team_P, "defeat algebraic connectivity", "win algebraic connectivity", 244, "D")
scatter_plot(fail_degree, win_degree, team_P, "defeat degree centrality", "win degree centrality", 245, "E")
scatter_plot(fail_closeness, win_closeness, team_P, "defeat closeness centrality", "win closeness centrality", 246, "F")
scatter_plot(fail_betweenness, win_betweenness, team_P, "defeat betweenness centrality", "win betweenness centrality", 247, "G")
scatter_plot(fail_eigenvector, win_eigenvector, team_P, "defeat eigenvector centrality", "win eigenvector centrality", 248, "H", 1, [1.05, 1, 3, 0])
plt.show()
print('finish')