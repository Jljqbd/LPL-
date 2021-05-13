import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from numpy.random import randn
from matplotlib.font_manager import FontProperties
def to_percent(y,position):
    return str(100*y)+"%"#这里可以用round（）函数设置取几位小数
#font=FontProperties(fname='/Library/Fonts/Songti.ttc')#这里设置字体，可以显示中文
x=randn(1000)
plt.hist(x,bins=30,weights=[1./len(x)]*len(x))#这里weights是每一个数据的权重，这里设置是1，weights是和x等维的列表或者series
fomatter=FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(fomatter)
plt.title("zft")
plt.show()
