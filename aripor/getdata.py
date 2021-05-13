import numpy as np
import requests
import json
import time
import hashlib
import os
import datetime
import sys
sys.path.append('E:/VSC_project/VSC py/LPL/aripor/')
import fpGrowth as fp
import ExcelIO as exo
from lxml import etree
from xpinyin import Pinyin
match_hero = []
session = requests.session()
index_url = "https://www.wanplus.com/lol/video/pro?eid=817"
def getdata(url, headers):
    global cookie, cookie_jar, session
    r = session.get(url, headers = headers)
    webstr = r.text
    html = etree.HTML(webstr)
    html_data = html.xpath("/html/body/div/div/div/div/div/div/ul/li/a/@href")
    all_url = []
    for line in html_data:
        all_url.append("https://www.wanplus.com"+line)
    return all_url
def match_data(url):
    global match_hero, cookie, cookie_jar, session
    matchid = url.split("matchid=")[-1]
    headers = {
        'authority': 'www.wanplus.com',
        'method': 'GET',
        'path': '/ajax/matchdetail/'+ matchid +'?_gtk=1210991487',
        'scheme': 'https',
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'zh,en-US;q=0.9,en;q=0.8',
        'referer': url,
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36 Edg/83.0.478.56',
        'x-csrf-token': '1210991487',
        'x-requested-with': 'XMLHttpRequest',
        }
    requrl = 'http://' + headers['authority'] + headers['path']
    data = session.get(requrl, headers = headers)
    json_data = json.loads(data.text)
    bherolist = json_data['data']['plStats']['damage']['blue']['players']
    rherolist = json_data['data']['plStats']['damage']['red']['players']
    templist1 = []
    templist2 = []
    for i in range(len(bherolist)):
        templist1.append(bherolist[i]['cpherokey'])
        templist2.append(rherolist[i]['cpherokey'])
    match_hero.append(templist1)
    match_hero.append(templist2)
def event_hero_list():
    excel_op = exo.ExcelOp(file="E:/VSC_project/VSC py/LPL/event.xlsx")
    return excel_op.get_col_value(9)
def member_list(hero_list):
    new_list = []
    for i in hero_list:
        if i != None:
            new_list.append(i.split(";"))
    return new_list
def main():
    hero_data = member_list(event_hero_list())
    initSet = fp.createInitSet(hero_data)              #对数据集进行整理，相同集合进行合并。
    myFPtree, myHeaderTab = fp.createTree(initSet, 20)#创建FP树。
    freqItemList = []
    fp.mineTree(myFPtree, myHeaderTab, 20, set([]), freqItemList) #递归的从FP树中挖掘出频繁项集。
    print('以下是频繁子项:')
    print(freqItemList)
    print("频繁子项寻找完毕")
def main1():
    global cookie, cookie_jar, session
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
    urllist = getdata(index_url, headers)
    for url in urllist:
        match_data(url)
    initSet = fp.createInitSet(match_hero)              #对数据集进行整理，相同集合进行合并。
    myFPtree, myHeaderTab = fp.createTree(initSet, 110)#创建FP树。
    freqItemList = []
    fp.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItemList) #递归的从FP树中挖掘出频繁项集。
    print('以下是频繁子项:')
    print(freqItemList)
    print("频繁子项寻找完毕")
    return freqItemList
if __name__ == '__main__':
    main()