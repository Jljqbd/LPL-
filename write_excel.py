import xlrd
import openpyxl
data = [['张三','男','未婚',20],['李四','男','已婚',28],['小红','女','未婚',18],['小芳','女','已婚',25]]
def write_excel(data, writetype, type2data = ''):
    output = open('data.xls','w',encoding='gbk')
    if writetype == 2:
        output.write(type2data + '\n')
    for i in range(len(data)):
        for j in range(len(data[i])):
            output.write(str(data[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
            output.write('\t')   #相当于Tab一下，换一个单元格
    output.write('\n')       #写完一行立马换行
    output.close()