import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
#这里网络散度其实作用不是很大,可以尝试,5个位置的散度
def scatter_plot(x_c, y_c, team_id, x_text, y_text, pic_pos, pic_text, isbox = 0):
      #类似于先声明一张图片，这个figure后面所有的设置都是在这张图片上操作的
    x_max = max(x_c)+( max(x_c) - min(x_c))
    y_max = max(y_c)+( max(y_c) - min(y_c))
    x = [0,max(x_max, y_max)]
    y = [0,max(x_max, y_max)]
    plt.subplot(pic_pos)
    plt.plot(x,y)    #制图
    for i in range(len(x_c)):
        if i == team_id:
            plt.scatter(x_c[i], y_c[i], marker='h', edgecolors='c', s=100, label = i)
        else:
            plt.scatter(x_c[i], y_c[i], c='', marker='o', edgecolors='b', s=50, label = i)
    plt.xlabel(x_text)
    plt.ylabel(y_text)
    for i in range(len(x_c)):
        plt.text(x_c[i], y_c[i], i)
    if isbox == 1:
        ax=plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])
        #ax.legend(pic_legend, tuple([i for i in range(len(x_c))]))
        ax.legend(loc=3, bbox_to_anchor=(1.05, 0),borderaxespad=0)
    #plt.show()    #显示图片
# 特征向量中心性
x=[1.2, 2.3, 5.9, 7.1, 9.0, 6.6, 7.9, 8.1]
y=[4.4, 3.2, 5.5, 7.1, 6.8, 9.9, 2.1, 1.3]
p = 1
plt.figure()
scatter_plot(x, y, p, "test x", "test y", 121, "")
scatter_plot(x, y, p, "test x", "test y", 122, "", 1)
plt.show()