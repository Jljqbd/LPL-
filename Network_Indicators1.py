import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

teamsname = ['FPX','RNG','TES','EDG','BLG','LNG','WE','SN','JDG','DMO','IG','VG','RW','LGD','V5','OMG']

def get_max(array):
    max = 0
    for i in range(len(array)):
        for j in range(len(array)):
           if array[i][j] != float("inf") and array[i][j] > max:
               max = array[i][j]
    return max
def get_min(array):
    min = float("inf")
    for i in range(len(array)):
        for j in range(len(array)):
            if array[i][j] != 0 and  array[i][j] < min:
                min = array[i][j]
    return min
def num_event(event, team_id, teamname, event_type, pic_pos, pic_text):
    index = np.arange(2)
    avg_other = np.delete(event, team_id).mean()
    myteam = event[ team_id ]
    error = [np.std( event ), np.std( np.delete( event, team_id ) )]
    plt.subplot(pic_pos)
    plt.title(event_type)
    plt.bar(index, [avg_other, myteam], yerr = error, error_kw = {'ecolor' : '0.2', 'capsize' :6}, alpha=0.7, color = ['blue', 'red'])
    plt.xticks(index, ['mean', teamname[ team_id ]] )
    #plt.text( max([avg_other, myteam])*0.9, max([avg_other, myteam])*0.9, pic_text)
    #plt.legend(loc = 2)
    #plt.show()
    return myteam, avg_other
def create_shortpath_graph(node_name, node_weight, isput = 0):
    #node_weight = np.trunc(node_weight/20)
    #ave_weight = 1/(sum(sum(node_weight))/len(node_weight)**2)
    node_weight = 1/node_weight
    node_max_weight = get_max(node_weight)
    G = nx.Graph()
    
    if node_weight[-1][0] == float("inf"):
        node_weight[-1][0] = node_max_weight*10
    for i in range(len(node_name)-1):
        if node_weight[i][i+1]==float("inf"):
            node_weight[i][i+1] = node_max_weight*10
    
    for i in range(len(node_name)):
        for j in range(len(node_name)):
            if i!=j and node_weight[i][j]!=float("inf"):
                G.add_edge(node_name[i], node_name[j], weight = node_weight[i][j])
                G.add_node(node_name[i], desc=node_name[i])
                G.add_node(node_name[j], desc=node_name[j])
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.2]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.2]

    pos = nx.shell_layout(G)
    # nx.draw_networkx(G)
    if isput == 1:
        nx.draw_networkx_nodes(G, pos, node_size = 150, node_color='g' )
        node_labels = nx.get_node_attributes(G, 'desc')
        nx.draw_networkx_labels(G, pos, labels = node_labels)
        nx.draw_networkx_edges(G, pos, edgelist = esmall, width = 1.5, edge_color = 'b')
        nx.draw_networkx_edges(G, pos, edgelist = elarge, width = 1.5, alpha = 0.5, edge_color = 'b', style = 'dashed')
        #nx.draw(G, pos = nx.shell_layout(G), node_color = 'b', edge_color = 'r', with_labels = True, font_size =18, node_size =20)
        plt.show()
    return G
    # print('图像输出完成')
def create_G(node_name, node_weight):
    #ave_weight = 1/(sum(sum(node_weight))/len(node_weight)**2)
    node_min_weight = get_min(node_weight)
    # node_weight = 1/node_weight
    G = nx.Graph()
    # 保持基本的连通性
    
    if node_weight[-1][0] == 0:
        node_weight[-1][0] = node_min_weight*0.01
    for i in range(len(node_name)-1):
        if node_weight[i][i+1] == 0:
            node_weight[i][i+1] = node_min_weight*0.01 #保持基本连通性
    # add edge
    for i in range(len(node_name)):
        for j in range(len(node_name)):
            if i != j and node_weight[i][j] != 0:
                G.add_edge(node_name[i], node_name[j], weight = node_weight[i][j])
    return G
def avg_short_path_length(G):
    UG=G.to_undirected()
    return nx.average_shortest_path_length(UG, weight="weight")
def graph_max_eig(node_weight):
    a , b = np.linalg.eig(node_weight)
    return max(a)
def C(node_weight):
    n_len = len(node_weight)
    c_list = []
    for i in range(n_len):
        above = 0
        below = 0
        for j in range(n_len):
            for k in range(n_len):
                if i!=j and j!=k and i!=k:
                    above += node_weight[i][j]*node_weight[j][k]*node_weight[i][k]
                    below += node_weight[i][j]*node_weight[i][k]
                    if below == 0:
                        below = 0.001
        c_list.append(above/below)
    return np.mean(c_list)
def Ac(node_weight): # Algebraic connectivity
    for i in range(len(node_weight)):
        for j in range(len(node_weight)):
            if node_weight[i][j] == 0:
                node_weight[i][j] = 0.01
    diagonal = sum(node_weight)
    S = np.eye(len(diagonal))*diagonal
    L = S - node_weight
    e1 , e2 = np.linalg.eig(L)
    e1.sort()
    return e1[1] #第二小特征值
#这里网络散度其实作用不是很大,可以尝试,5个位置的散度
def scatter_plot(x_c, y_c, team_id, x_text, y_text, pic_pos, pic_text, isbox = 0, label_pos = []):
      #类似于先声明一张图片，这个figure后面所有的设置都是在这张图片上操作的
    x_max = max(x_c)*1.05
    y_max = max(y_c)*1.05
    x_min = min(x_c)*0.95
    y_min = min(y_c)*0.95
    x = [min(x_min, y_min), max(x_max, y_max)]
    y = [min(x_min, y_min), max(x_max, y_max)]
    plt.subplot(pic_pos)
    plt.plot(x,y)    #制图
    for i in range(len(x_c)):
        if i == team_id:
            plt.scatter(x_c[i], y_c[i], marker='h', edgecolors='c', s=100, label = str(i) + ": " + teamsname[i])
        else:
            plt.scatter(x_c[i], y_c[i], c='', marker='o', edgecolors='b', s=50, label = str(i) + ": " + teamsname[i])
    plt.xlabel(x_text)
    plt.ylabel(y_text)
    for i in range(len(x_c)):
        if i == team_id:
            plt.text(x_c[i], y_c[i], teamsname[i])
        else:
            plt.text(x_c[i], y_c[i], i)
    if isbox == 1:
        ax=plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])
        ax.legend(loc=label_pos[2], bbox_to_anchor=(label_pos[0], label_pos[1]), borderaxespad=label_pos[3])
    write_data = [[x_c[i], y_c[i]] for i in range(len(x_c))]
    write_excel(write_data, 2, [x_text, y_text])
    #plt.show()    #显示图片
# 特征向量中心性
def eigenvector(G):
    a =  nx.eigenvector_centrality(G)
    a_avg = dict_Avg(a)
    return a_avg
# 度中心性
def degree(G):
    a = nx.degree_centrality(G)
    a_avg = dict_Avg(a)
    return a_avg
# 紧密中心性
def closeness(G):
    a = nx.closeness_centrality(G)
    a_avg = dict_Avg(a)
    return a_avg
# 间接中心性
def betweenness(G):
    a = nx.betweenness_centrality(G)
    a_avg = dict_Avg(a)
    return a_avg
def dict_Avg( Dict ) :
    L = len( Dict )						#取字典中键值对的个数
    S = sum( Dict.values() )				#取字典中键对应值的总和
    A = S / L
    return A
def write_excel(data, writetype, type2data = ''):
    output = open('data.xls','a+',encoding='gbk')
    if writetype == 2:
        output.write("\n" + type2data[0]+'\t'+ type2data[1]+ '\n')
    for i in range(len(data)):
        for j in range(len(data[i])):
            output.write(str(data[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
            output.write('\t')   #相当于Tab一下，换一个单元格
        output.write('\n')       #写完一行立马换行
    output.close()