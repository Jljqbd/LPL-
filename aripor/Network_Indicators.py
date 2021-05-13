import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from scipy import optimize
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter
from numpy.random import randn
from matplotlib.font_manager import FontProperties

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
def Degree_distribution(node):
    '''
    本函数为了统计度分布即:度为1的点有几个,为2的点有几个
    返回的数组第i个元素为度为i的点有几个
    '''
    node = ad2zo(node)
    hero_num = len(node)
    h_sum = np.sum(node, axis=1)
    l_sum = np.sum(node, axis=0)
    dedis = np.zeros(int(hero_num)*2+1)
    for i in range(hero_num):
        dedis[int(h_sum[i]+l_sum[i])] += 1
    return dedis
def func(x, p): # 拟合函数的形式
    lamda, a, b = p
    return (a / x**(lamda))
def residuals(p, y ,x):
    return y - func(x, p)
def Function_fitting(node_list):
    '''
    给出数据点拟合曲线用的,有效点为(i,nodelist[i])
    '''
    x = []
    y1 = []
    node_len = len(node_list)
    for i in range(node_len):
        if node_list[i]!= 0:
            x.append(i)
            y1.append(node_list[i])
    m=[]
    x = np.array(x)
    y1 = np.array(y1)
    p0 = [1.8, 40, -1] #初始的拟合参数
    plsq = optimize.leastsq(residuals, p0, args=(y1, x)) 

    print (u"拟合参数", plsq[0] )# 实验数据拟合后的参数
    fig = plt.figure()
    plt.plot(x, y1, "o", label=u"Know data point")
    plt.plot(x, func(x, plsq[0]), label=u"Fit data")
    plt.legend(loc="best")
    plt.grid(True, linestyle='-.')
    plt.show()
def probability_distribution_extend(data, bins, margin=0.01, label='Distribution', x_title = '',y_title='Frequency distribution'):
    '''
    频度分布图
    # 自己给定区间，小于区间左端点和大于区间右端点的统一做处理，对于数据分布不均很的情况处理较友好
    # bins      自己设定的区间数值列表
    # margin    设定的左边和右边空留的大小
    # label     右上方显示的图例文字
    '''
    bins = sorted(bins)
    length = len(bins)
    intervals = np.zeros(length+1)
    for value in data:
        i = 0
        while i < length and value >= bins[i]:
            i += 1
        intervals[i] += 1
    print(intervals)
    intervals = intervals / float(len(data))
    plt.xlim(min(bins) - margin, max(bins) + margin)
    bins.insert(0, -999)
    plt.title("probability-distribution")
    plt.xlabel('Interval')
    plt.ylabel('Probability')
    plt.bar(bins, intervals, color=['b'], label=label)
    plt.legend()
def to_percent(y,position):
    return str(100*y)+"%"#这里可以用round（）函数设置取几位小数
def probability_distribution(data, bins_interval=0, margin=1, x_title = '', y_title='Frequency distribution'):
    data_min = max(data)
    for i in data:
        if i != 0 and i < data_min:
            data_min = i
    # data = [data_min / 2 if x == 0 else x for x in data] # 在不影响结果的情况下让图片好看一些
    bins = np.arange(0, max(data)  + 0.05, bins_interval)
    x_min = max(0, min(data) - margin)
    plt.xlim(x_min, max(data) + margin)
    plt.title("Probability-distribution")
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    # 频率分布normed=True，频次分布normed=False
    n, bins, patches = plt.hist(x=data, bins=bins, alpha=0.5, histtype='stepfilled',
         color='steelblue', edgecolor='none', weights=1)
    '''
    fomatter=FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)
    '''
    mu =np.mean(data) #计算均值
    sigma =np.std(data)
    plt.grid(True, linestyle='-.')
    y = norm.pdf(bins, mu, sigma) # 拟合一条最佳正态分布曲线y 
    plt.plot(bins, y, 'b') # 绘制y的曲线
    '''
    for x, y in zip(left, prob):
        # 字体上边文字
        # 频率分布数据 normed=True
        plt.text(x + bins_interval / 2, y + 0.003, '%.2f' % y, ha='center', va='top')
        # 频次分布数据 normed=False
        #plt.text(x + bins_interval / 2, y + 0.25, '%.2f' % y, ha='center', va='top')
    '''
def create_default_G(nodename, G, del_node_num, isnode = 0):
    '''
    功能创建自定义缺省的节点的图,与接下来与原图的指标作比较
    nodename:原图的节点名称
    G:原图
    del_node_num:想要缺省的节点,list类型
    isnode : 是否要返回节点类型的变量, 默认为图类型
    return ndG(not default G), dG(default G)
    '''
    nG = np.delete(G, del_node_num, axis = 1)
    nG = np.delete(nG, del_node_num, axis = 0)
    ndG = create_G(node_name, G) if isnode==0 else G #返回的第一个变量, 原图的G
    for i in sorted(index, reverse = True):
        del nodename[i]
    dG = create_G(node_name, nG) if isnode == 0 else nG #返回的第二个变量, 缺省自定义节点后的图
    return ndG, dG
def network_bar(fdata, ydata, x, pic_pos, iname, fig):
    '''
    fdata : 缺省后的数据
    ydata : 原来的数据
    x : 队伍名称是个list
    pic_pos : 图像位置
    iname: 指标名称(Indicator name)
    '''
    fdata = [ round(i,1) for i in fdata]
    ydata = [ round(i,1) for i in ydata]
    ax1 = fig.add_subplot(pic_pos)
    ax1.set_ylim([min( min(ydata), min(fdata) ), max( max(ydata), max(fdata) )])
    ax1.bar(x, ydata, alpha=0.7, color='k')
    ax1.set_ylabel(iname, fontsize='20')
    ax1.tick_params(labelsize=10)
    ax1.grid(True, linestyle='-.')
    for i, (_x, _y) in enumerate(zip(x, ydata)):
        plt.text(_x, _y, ydata[i], color='black', fontsize=10, ha='center', va='bottom')  # 将数值显示在图形上
    ax2 = ax1.twinx()  # 组合图必须加这个
    ax2.set_ylim([min( min(ydata), min(fdata) ), max( max(ydata), max(fdata) )]) 
    ax2.plot(x, fdata, 'r', ms=10, lw=3, marker='o') # 设置线粗细，节点样式
    #ax2.set_ylabel(u'Indicator after deleting the node', fontsize='20')
    sns.despine(left = True, bottom = True)   # 删除坐标轴，默认删除右上
    ax2.tick_params(labelsize = 10)
    for x, y in zip(x, fdata):   # # 添加数据标签
        plt.text(x, y-2.5, str(y), color = 'blue', ha='center', va='bottom', fontsize=10, rotation=0)
    return fig
    #plt.show()
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
    # 用于非最短路的图指标计算,边的权重为他们的协作次数
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
                G.add_node(node_name[i], desc=node_name[i])
                G.add_node(node_name[j], desc=node_name[j])
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
        below = 0
        above = 0
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
# 指标分布网络图(每个节点指标大小不同颜色不同)
def ind(G, fun, title): # Indicator network diagram
# 程序参考网址,实现点度不同,点的颜色不同的协作网络图
# https://networkx.github.io/documentation/latest/auto_examples/drawing/plot_random_geometric_graph.html#sphx-glr-auto-examples-drawing-plot-random-geometric-graph-py
    p = fun(G)
    # pos = nx.shell_layout(G)
    # pos = nx.random_layout(G)
    pos = nx.spring_layout(G)
    # pos = nx.circular_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    sc = nx.draw_networkx_nodes(
        G,
        pos,
        with_labels=True,
        nodelist=list(p.keys()),
        node_size=80,
        node_color=list(p.values()),
        # cmap=plt.cm.Reds_r, #对应于红色的颜色映射
    )
    # 如果觉得太满名字黑漆漆不好看,就删掉下面两行(开始)
    node_labels = nx.get_node_attributes(G, 'desc')
    nx.draw_networkx_labels(G, pos, labels = node_labels)
    # (结束)
    plt.colorbar(sc)
    #plt.xlim(-0.05, 1.05)
    #plt.ylim(-0.05, 1.05)
    plt.axis("off")
    plt.title(title)
    plt.show()
def ad2zo(A): # 邻接矩阵到0-1矩阵
    row = len(A)
    col = len(A[0])
    r_array = np.zeros((len(A),len(A)))
    for i in range(row):
        for j in range(row):
            r_array[i][j] = 1 if A[i][j]!=0 else 0
    return r_array
def zo2graph(node_name, node_weight):
    G = nx.Graph()
    # add edge
    for i in range(len(node_name)):
        for j in range(len(node_name)):
            if i != j and node_weight[i][j] != 0:
                G.add_edge(node_name[i], node_name[j], weight = node_weight[i][j])
                G.add_node(node_name[i], desc=node_name[i])
                G.add_node(node_name[j], desc=node_name[j])
    return G
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
    a = nx.closeness_centrality(G) # 从这里可一看到每个英雄的紧密中心性的值,是这个节点到每个节点的最短路径之和,它越小说明整个英雄与其他英雄之间的连接越紧密
    a_avg = dict_Avg(a) # 平均的每个节点的指标相加取平均
    return a_avg
# 间接中心性
def betweenness(G):
    a = nx.betweenness_centrality(G) # 此函数的节点值越高越好,越高说明越多的最短路经过了该节点
    a_avg = dict_Avg(a) # 平均间接中心性,越多说明整个图中每个节点在最短路中出现的频次越高,进而可能就能推出,每个选择出来的英雄就越大的发挥了其作用
    return a_avg
def dict_Avg( Dict ) :
    L = len( Dict )						#取字典中键值对的个数
    S = sum( Dict.values() )				#取字典中键对应值的总和
    A = S / L
    return A
def write_excel(data, writetype, type2data = ''):
    output = open('win_fail_data.xls','a+',encoding='gbk')
    strline = "\n"
    if writetype == 2:
        for s in type2data:
            strline += (s+"\t")
        strline += "\n"
        output.write(strline)
    for i in range(len(data)):
        for j in range(len(data[i])):
            output.write(str(data[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
            output.write('\t')   #相当于Tab一下，换一个单元格
        output.write('\n')       #写完一行立马换行
    output.close()
    print("\n写入成功, 共写入 "+str(len(data))+" 行数据....\n")
def dicw_excel(data, num):
    output = open('pin_' + num + '.xls','a+',encoding='gbk')
    for i in data.values():
        output.write(str(i)+'\n')
    output.close()
def del_row_col(array, use_hero, del_name):#严重bug他会删除原有的变量
    '''
    删除array的特定几行与几列
        input:
            array : numpy类型的矩阵
            use_hero : 这个队伍使用的所有英雄
            del_name : 这个队伍要删除的英雄
        output:
            array : 删除后的矩阵
    '''
    del_name_l = list(del_name) if type(del_name) != list else del_name
    for i in range(len(array)):#对于每个队伍
        del_index = []
        for hero in del_name_l:
            if hero in use_hero[i]:
                del_index.append(use_hero[i].index(hero))
        array[i] = np.delete(array[i], del_index, axis = 0) # 删除多行
        array[i] = np.delete(array[i], del_index, axis = 1) # 删除多列
    return array
def print_plot(fdata_3, fdata_4, ydata, x, pic_pos, iname, fig):
    '''
    fdata_3 : 缺省3英雄后的数据
    fdata_4 : 缺省4英雄后的数据
    ydata : 原来的数据
    x : 队伍名称是个list
    pic_pos : 图像位置
    iname: 指标名称(Indicator name)
    '''
    ax1 = fig.add_subplot(pic_pos)
    ax1.set_ylabel(iname, fontsize='20')
    ax1.tick_params(labelsize=10)
    ax1.grid(True, linestyle='-.')
    '''
    for i, (_x, _y) in enumerate(zip(x, fdata_3)):
        plt.text(_x, _y, fdata_3[i], color='black', fontsize=10, ha='center', va='bottom')  # 将数值显示在图形上
    for i, (_x, _y) in enumerate(zip(x, fdata_4)):
        plt.text(_x, _y, fdata_4[i], color='black', fontsize=10, ha='center', va='bottom')  # 将数值显示在图形上
    for i, (_x, _y) in enumerate(zip(x, ydata)):
        plt.text(_x, _y, ydata[i], color='black', fontsize=10, ha='center', va='bottom')  # 将数值显示在图形上     
    '''
   
    #缺省3个的
    line1, = plt.plot(x, fdata_3[0], lw=1, c='red', marker='s', ms=4, label='X1')  # 绘制x1
    line2, = plt.plot(x, fdata_3[1], lw=1, c='y', marker='o', ms=4, label='X2')  # 绘制x2
    line3, = plt.plot(x, fdata_3[2], lw=1, c='g', marker='^', ms=4, label='X3')  # 绘制x3
    #缺省4个的
    line4, = plt.plot(x, fdata_4[0], lw=1, c='c', marker='v', ms=4, label='Y1')  # 绘制y1
    line5, = plt.plot(x, fdata_4[1], lw=1, c='b', marker='+', ms=4, label='Y2')  # 绘制y2
    line6, = plt.plot(x, fdata_4[2], lw=1, c='m', marker='x', ms=4, label='Y3')  # 绘制y3

    line7, = plt.plot(x, ydata, lw=2, c = 'black', marker='*', ms=4, label ='Original')
    sns.despine(left = True, bottom = True)   # 删除坐标轴，默认删除右上
    if pic_pos==224:
        ax1.legend([line1,line2,line3,line4,line5,line6,line7],['X1','X2','X3','Y1','Y2','Y3','Original'], bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0) # 多加一个逗号 
    return fig
def plural2float_1(a):
    return [i.real for i in a]
def plural2float_2(a):
    b=[]
    for ai in a:
        b.append([i.real for i in ai])
    return b
            

