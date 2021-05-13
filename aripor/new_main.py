import numpy as np
import xlrd
import networkx as nx
import sys
sys.path.append('E:/VSC_project/VSC py/LPL/aripor/')
from download_match_data import *
from Network_Indicators import *
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import requests
from lxml import etree
from getdata import *
from ALLTeamData import *
from pandas.core.frame import DataFrame
from pandas import *
import time
team_P = 1 # 研究的队伍
session = requests.session()
'''
match_id = "59550" # 第一场比赛id
response_url = "54681"
'''
remove_hero = ['Sylas', 'Kaisa'] # 缺省的英雄, 这里以塞拉斯为例
remove_hero_3 = [
                ['Kaisa', 'Nautilus'],
                ['Gragas', 'Kaisa'],
                ['Varus', 'TahmKench'],
                ]
remove_hero_4 = [
                ['Olaf', 'Karma', 'Yuumi', 'Sivir'],
                ['Corki', 'Leona', 'Kalista', 'JarvanIV'],
                ['Xayah', 'Karma', 'RekSai', 'Morgana'],
                ['Olaf', 'Karma', 'Sivir', 'Morgana'],
                 ]
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
isread = 0
atd = ALLTeamData(session, remove_hero, response_url_list, match_id_list, teamsname, isread)
for i in range(len(match_id_list)):
    if i ==0:
        dmd = download_match_data(session, atd, i)
    else:
        if i%40==0:
            print('\r','在第'+ str(i) +'次刷新', end='')
            session = requests.session()
            read_match_data_2()
            dmd.atd.session = session
        dmd = download_match_data(dmd.session, dmd.atd, i)
    dmd.download()
    # print(str(((i+1)/(len(match_id_list)+1))*100) + ' %')
    print('\r', str(((i+1)/(len(match_id_list)+1))*100) + ' %', end='')
    time.sleep(0.5)
# 变量名简化
team_use_hero = dmd.atd.team_use_hero
Hc = dmd.atd.Hc
Jc = dmd.atd.Jc
win_Jc = dmd.atd.win_Jc
win_Hc = dmd.atd.win_Hc
fail_Jc = Jc - win_Jc
#=========================================
# 以下这段代码是为了构造fail_Hc
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
#==========================================


# ===============================================|
# 以下指标没有用到,但是信息都在dmd.atd存储(从这开始)
'''
Centroid_x = dmd.atd.Centroid_x # 平均x
Centroid_y = dmd.atd.Centroid_y # 平均y

win_dx = dmd.atd.win_dxdy[0] # 
win_dy = dmd.atd.win_dxdy[1]
fail_dx = dmd.atd.fail_dxdy[0]
fail_dy = dmd.atd.fail_dxdy[1]

fail_dxdy = fail_dx / fail_dy
win_dxdy = win_dx / win_dy
win_n_time = dmd.atd.win_fail_n_time[0] // 60 + (dmd.atd.win_fail_n_time[0] % 60) / 100 # 时间从s转化为min
fail_n_time = dmd.atd.win_fail_n_time[1] // 60 + (dmd.atd.win_fail_n_time[1] % 60) / 100

team_win_num = dmd.atd.team_win_fail_num[0]
team_fail_num = dmd.atd.team_win_fail_num[1]
'''
#=================================(从这结束)
DxDy = dmd.atd.dxdy[0] / dmd.atd.dxdy[1]
remove_dx = dmd.atd.remove_dxdy[0]
remove_dy = dmd.atd.remove_dxdy[1]
remove_DxDy = remove_dx / remove_dy
#
team_remove_kill_event = dmd.atd.team_remove_kill_event

Hc_team_P = Hc[team_P].copy()
# 协作网络图
create_shortpath_graph(team_use_hero[team_P], Hc[team_P].copy(), 1)
fig = plt.figure()
# 缺省节点前后的击杀数
fig = network_bar(team_remove_kill_event, dmd.atd.team_kill_event, teamsname, 111, 'Number of kill events', fig)
plt.xticks(rotation=270)
plt.show()

# 矩阵可视化

cm = plt.cm.get_cmap('RdYlBu')
sc = plt.matshow(np.array(Hc[team_P]), cmap=cm)
plt.colorbar(sc)
plt.show()

# 可视化结束

# 画度分布曲线:
degree_data = Degree_distribution(Hc[team_P])
Function_fitting(degree_data)
# 度分布曲线输出结束

# 频数分布图
graph = create_G(team_use_hero[team_P], Hc[team_P].copy())
# 特征向量中心性分布
data =  nx.eigenvector_centrality(graph)
#bins = np.arange(0,1.2*max(data.values()), 0.01)
plt.figure()
probability_distribution(data=list(data.values()), bins_interval = 0.007, margin=0.05, x_title='Eigenvector Centrality')
dicw_excel(data, '1')
plt.show()

# 特征向量中心性分布结束--------------
# 度中心性分布
plt.figure()
data = nx.degree_centrality(graph)
probability_distribution(data=list(data.values()), bins_interval = 0.007, margin=0.05, y_title = 'Degree Centrality')
dicw_excel(data, '2')
plt.show()

# 度中心性分布结束--------------------
# 紧密中心性分布
plt.figure()
data = nx.closeness_centrality(graph)
probability_distribution(data=list(data.values()), bins_interval = 0.007, margin=0.05, y_title = 'Closeness Centrality')
dicw_excel(data, '3')
plt.show()

# 紧密中心性分布结束------------------
# 间接中心性分布
plt.figure()
data = nx.betweenness_centrality(graph)
probability_distribution(data=list(data.values()), bins_interval = 0.007, margin=0.05, y_title = 'Betweenness Centrality')
plt.show()

# 间接中心性分布结束
# 频数分布图结束
# 画指标的空间分布图(开始)

# 度中心性
ind(graph, nx.degree_centrality, "Degree Centrality")
print('表三度中心性Cd排序结果[由大到小]\n')
print(sorted(nx.degree_centrality(graph).items(),key=lambda x:x[1], reverse=True))
# 介数中心性(betweenness)
ind(graph, nx.betweenness_centrality, "Betweenness Centrality")
print('表三介数中心性排序结果\n')
print(sorted(nx.betweenness_centrality(graph).items(),key=lambda x:x[1], reverse=True))
# 接近中心性(closeness centrality)
ind(graph, nx.closeness_centrality, "Closeness Centrality")
print('表三接近中心性排序结果\n')
print(sorted(nx.closeness_centrality(graph).items(),key=lambda x:x[1], reverse=True))
# 画指标的空间分布图(结束)

# 相关性分析
# 将三种中心性与平均x,y,成功场次,时间数,击杀数,成功游戏中对局时间,dy/dx的相关系数
# 由于atd.Centroid_x和y中存放的是每支队伍的每场比赛的数据,我们需要计算一支队伍一个赛季的表现
Centroid_team_x = []
Centroid_team_y = []
for i in atd.Centroid_x:
    Centroid_team_x.append(np.array(i).mean())
for i in atd.Centroid_y:
    Centroid_team_y.append(np.array(i).mean())
all_team_graph = [ create_G(team_use_hero[i], Hc[i].copy()) for i in range(team_len)]
eigenvector_data = [ eigenvector(i) for i in all_team_graph]
betweenness_data = [ betweenness(i) for i in all_team_graph]
closeness_data = [ closeness(i) for i in all_team_graph]
degree_data = [ degree(i) for i in all_team_graph]
Ac_data = [Ac(i) for i in Hc]
graph_max_eig_data = [graph_max_eig(i) for i in Hc]
short_path_data = [ avg_short_path_length(create_shortpath_graph(team_use_hero[i], Hc[i].copy())) for i in range(team_len)]
C_data = [ C(i) for i in Hc]
pandata = {
    #'Centroid_x': Centroid_team_x,
    #'Centroid_y': Centroid_team_y,
    'Winning_game':atd.team_win_fail_num[0],
    #'Winning_game_Time':atd.win_fail_n_time[0],
    'Kill_event': atd.team_kill_event,
    #'All_event': atd.team_event,
    #'Dx':atd.dxdy[0],
    #'Dy':atd.dxdy[1],
    #'C':C_data,
    'Max_eig':graph_max_eig_data,
    'Short_path':short_path_data,
    #'Ac':Ac_data,
    'Degree':degree_data,
    'Closeness':closeness_data,
    #'Betweenness':betweenness_data,
    #'Eigenvector':eigenvector_data,
}
data=DataFrame(pandata)#将字典转换成为数据框
# 相关性分析结束
corr_data = data.corr() #可以使用参数 'pearson', 'kendall', 'spearman'
print("表5(注意选出几列即可不用全写上去)")
print(corr_data)
# 小世界网络指数
ws = nx.watts_strogatz_graph(50, 5, 0.5)
A=np.array(nx.adjacency_matrix(ws).todense()) # 对应的邻接矩阵
L_rand = nx.average_shortest_path_length(ws, weight="weight")
C_rand = C(A)
# L_avg = pandata['Short_path'][team_P]
zo = ad2zo(Hc_team_P) #0-1矩阵
zgraph = zo2graph(team_use_hero[team_P], zo) #0-1图
L_avg = avg_short_path_length(zgraph)
# C_avg = pandata['C'][team_P]
C_avg = C(zo)
s = (C_avg / C_rand) / (L_avg / L_rand)
print("表一协作网络平均路径长度:"+str(L_avg)+"聚集系数:"+str(C_avg) +"\n")
print("表一随机网络平均路径长度:"+str(L_rand)+"聚集系数:"+str(C_rand)+ "\n")
print('小世界网络指数s为:'+str(s)+'\n小世界聚类系数为:'+str(C_avg)+"\n小世界最短路径为:"+str(L_avg)+"\n")
# 缺省前后的dx
'''
fig = plt.figure()
fig = network_bar(remove_dx, dmd.atd.dxdy[0], teamsname, 111, 'Dx', fig)
plt.xticks(rotation=270)
plt.show()

# 缺省前后的dy
fig = plt.figure()
fig = network_bar(remove_dy, dmd.atd.dxdy[1], teamsname, 111, 'Dy', fig)
plt.xticks(rotation=270)
plt.show()

# 缺省前后的dxdy
fig = plt.figure()
fig = network_bar(remove_DxDy, DxDy, teamsname, 111, 'Dx / Dy', fig)
plt.xticks(rotation=270)
plt.show()
'''
remove_Jc = dmd.atd.remove_Jc
remove_Hc = del_row_col(Hc.copy(), team_use_hero.copy(), remove_hero)
remove_win_Hc = del_row_col(win_Hc.copy(), team_use_hero.copy(), remove_hero)
remove_fail_Hc = del_row_col(fail_Hc.copy(), team_use_hero.copy(), remove_hero)
team_C = []
team_remove3_C = []
team_remove4_C = []
team_hero_short_path = []
team_remove3_hero_short_path = []
team_remove4_hero_short_path = []
team_eig = []
team_remove3_eig = []
team_remove4_eig = []
team_AC = []
team_remove3_AC = []
team_remove4_AC = []
remove_hero_Hc3 = [] # 每个元素为删除3个英雄后的Hc矩阵
remove_hero_Hc4 = [] # 每个元素为删除了4个英雄后的Hc矩阵

for i in range(3): # 默认每个为3组
    remove_hero_Hc3.append( del_row_col(Hc.copy(), team_use_hero.copy(), remove_hero_3[i]) )
    remove_hero_Hc4.append( del_row_col(Hc.copy(), team_use_hero.copy(), remove_hero_4[i]) )
for j in range(3): #缺省组合
    team_remove3_C.append([])
    team_remove4_C.append([])

    team_remove3_eig.append([])
    team_remove4_eig.append([])

    team_remove3_hero_short_path.append([])
    team_remove4_hero_short_path.append([])

    team_remove3_AC.append([])
    team_remove4_AC.append([])

    for i in range(team_len):
        if j==0:
            team_C.append(C(Hc[i]))
            team_eig.append(graph_max_eig(Hc[i])) # hero list
            team_hero_short_path.append(avg_short_path_length(create_shortpath_graph(team_use_hero[i], Hc[i])))
            team_AC.append(Ac(Hc[i]))

        team_remove3_C[j].append(C(remove_hero_Hc3[j][i])) #第i个队伍的第j种缺省组合
        team_remove4_C[j].append(C(remove_hero_Hc4[j][i]))

        
        team_remove3_eig[j].append(graph_max_eig(remove_hero_Hc3[j][i]))
        team_remove4_eig[j].append(graph_max_eig(remove_hero_Hc4[j][i]))

        team_remove3_hero_icopy = team_use_hero[i].copy()
        team_remove4_hero_icopy = team_use_hero[i].copy()
        for hero in remove_hero_3[j]: #去除第j种缺省模式种的英雄
            (hero not in team_remove3_hero_icopy) or team_remove3_hero_icopy.remove(hero)
            #team_remove_hero_icopy.remove(hero)
        for hero in remove_hero_4[j]:
            (hero not in team_remove4_hero_icopy) or team_remove4_hero_icopy.remove(hero)
        team_remove3_hero_short_path[j].append(avg_short_path_length(create_shortpath_graph(team_remove3_hero_icopy, remove_hero_Hc3[j][i])))
        team_remove4_hero_short_path[j].append(avg_short_path_length(create_shortpath_graph(team_remove4_hero_icopy, remove_hero_Hc4[j][i])))

        team_remove3_AC[j].append(Ac(remove_hero_Hc3[j][i]))
        team_remove4_AC[j].append(Ac(remove_hero_Hc4[j][i]))
fig = plt.figure()
# 缺省前后的Clustering coefficient
# fig = network_bar(team_remove_C, team_C, teamsname, 221, 'Clustering coefficient', fig)
fig = print_plot(plural2float_2(team_remove3_C), plural2float_2(team_remove4_C), team_C, teamsname, 221, 'Clustering coefficient', fig)

# 缺省前后的最短路径
# fig = network_bar(team_remove_hero_short_path, team_hero_short_path, teamsname, 222, 'Short Path', fig)
fig = print_plot(team_remove3_hero_short_path, team_remove4_hero_short_path,team_hero_short_path, teamsname, 222, 'Short Path', fig)

# 缺省前后的Algebraic connectivity
# fig = network_bar(team_remove_AC, team_AC, teamsname, 223, 'Algebraic connectivity', fig)
fig = print_plot(plural2float_2(team_remove3_AC), plural2float_2(team_remove4_AC), plural2float_1(team_AC), teamsname, 223, 'Algebraic connectivity', fig)
# 缺省前后的最大特征值
# fig = network_bar(team_remove_eig, team_eig, teamsname, 224, 'Max eig', fig)
fig = print_plot(plural2float_2(team_remove3_eig), plural2float_2(team_remove4_eig), team_eig, teamsname, 224, 'Max eig', fig)
plt.show()
exit(0)
if dmd.atd.isread == 0:
    print("其他数据没有写入excel表格,运行结束...")
    exit(0)
win_fail_data = [ [] for i in range(len(teamsname))]
for i in range(len(teamsname)):
    team_remove_hero_icopy = team_use_hero[i].copy()
    for hero in remove_hero:
        (hero not in team_remove_hero_icopy) or team_remove_hero_icopy.remove(hero)
    win_Hc_graph = create_G(team_use_hero[i], win_Hc[i])
    remove_win_Hc_graph = create_G(team_remove_hero_icopy, remove_win_Hc[i])
    fail_Hc_graph = create_G(team_use_hero[i], fail_Hc[i])
    remove_fail_Hc_graph = create_G(team_remove_hero_icopy, remove_fail_Hc[i])
    #0)teamname
    win_fail_data[i].append(teamsname[i])
    #1)C
    win_fail_data[i].append(C(win_Hc[i]))
    win_fail_data[i].append(C(remove_win_Hc[i]))
    win_fail_data[i].append(C(fail_Hc[i]))
    win_fail_data[i].append(C(remove_fail_Hc[i]))
    #2)avg_short_path_length
    win_fail_data[i].append(avg_short_path_length(create_shortpath_graph(team_use_hero[i], win_Hc[i])))
    win_fail_data[i].append(avg_short_path_length(create_shortpath_graph(team_remove_hero_icopy, remove_win_Hc[i])))
    win_fail_data[i].append(avg_short_path_length(create_shortpath_graph(team_use_hero[i], fail_Hc[i])))
    win_fail_data[i].append(avg_short_path_length(create_shortpath_graph(team_remove_hero_icopy, remove_fail_Hc[i])))
    #3)graph_max_eig
    win_fail_data[i].append(graph_max_eig(win_Hc[i]))
    win_fail_data[i].append(graph_max_eig(remove_win_Hc[i]))
    win_fail_data[i].append(graph_max_eig(fail_Hc[i]))
    win_fail_data[i].append(graph_max_eig(remove_fail_Hc[i]))
    #4)Ac
    win_fail_data[i].append(Ac(win_Hc[i]))
    win_fail_data[i].append(Ac(remove_win_Hc[i]))
    win_fail_data[i].append(Ac(fail_Hc[i]))
    win_fail_data[i].append(Ac(remove_fail_Hc[i]))
    #5)degree
    win_fail_data[i].append(degree(win_Hc_graph))
    win_fail_data[i].append(degree(remove_win_Hc_graph))
    win_fail_data[i].append(degree(fail_Hc_graph))
    win_fail_data[i].append(degree(remove_fail_Hc_graph))
    # 6)closeness
    win_fail_data[i].append(closeness(win_Hc_graph))
    win_fail_data[i].append(closeness(remove_win_Hc_graph))
    win_fail_data[i].append(closeness(fail_Hc_graph))
    win_fail_data[i].append(closeness(remove_fail_Hc_graph))
    # 7)betweenness
    win_fail_data[i].append(betweenness(win_Hc_graph))
    win_fail_data[i].append(betweenness(remove_win_Hc_graph))
    win_fail_data[i].append(betweenness(fail_Hc_graph))
    win_fail_data[i].append(betweenness(remove_fail_Hc_graph))
    # 8)eigenvector
    win_fail_data[i].append(eigenvector(win_Hc_graph))
    win_fail_data[i].append(eigenvector(remove_win_Hc_graph))
    win_fail_data[i].append(eigenvector(fail_Hc_graph))
    win_fail_data[i].append(eigenvector(remove_fail_Hc_graph))
Label_content = ['队伍名称',
    '原胜利聚类系数','移除后胜利聚类系数','原失败聚类系数','移除后失败聚类系数', \
    '原胜利最短路','移除后胜利最短路','原失败最短路','移除后失败最短路', \
    '原胜利最大特征值','移除后胜利最大特征值','原失败最大特征值','移除后失败最大特征值', \
    '原胜利代数连通性','移除后胜利代数连通性', '原失败代数连通性','移除后失败代数连通性', \
    '原胜利度中心性','移除后胜利度中心性','原失败度中心性','移除后失败度中心性', \
    '原胜利紧密中心性','移除后胜利紧密中心性','原失败紧密中心性','移除后失败紧密中心性', \
    '原胜利间接中心性', '移除后胜利间接中心性','原失败间接中心性','移除后失败间接中心性', \
    '原胜利特征向量中心度', '移除后胜利特征向量中心度', '原失败特征向量中心度', '移除后失败特征向量中心度'
    ]
write_excel(win_fail_data, 2, Label_content)
print("其他数据写入excel表格,运行结束...")