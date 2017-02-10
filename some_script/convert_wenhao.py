#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thu May  5 11:09:20 2016

@author: yangfan
"""
'''
经过测试这个办法是无效的。问号产生以后，只能用替换的方式去除。

产生的原因：从数据库导出以后就会有，可能是数据库使用的gbk编码导致的，表现为从Windows转换到mac，关键词匹配不上，解决办法就是去Linux电脑上去除特殊符号，因为这些符号在其他系统上面看不到。
'''

import xlrd
import xlwt
import time

paths=['/Users/zhangyu/Desktop/CT_keyword-part1.xls']


time1 = time.time()

count=0
content=[]

words=[]
cnt = []
flag = []

list_word = []
list_cnt= []
list_flag =[]

for path in paths:
    data=xlrd.open_workbook(path)
    table=data.sheet_by_index(0)
    words+=table.col_values(0)
    cnt+=table.col_values(1)
    flag+=table.col_values(2)

randomlist=range(len(words))

for folderi in randomlist:
    list_word.append(words[folderi])
    list_cnt.append(cnt[folderi])
    list_flag.append(flag[folderi])
        
#print(len(content))
#print(len(set(content)))
#content=sorted(content)


xlcount=0
xlcount_sheet2=0
guanjianci=xlwt.Workbook(encoding='utf-8')
sheet1=guanjianci.add_sheet('Sheet1')
sheet2=guanjianci.add_sheet('Sheet2')

colcount = 0
for col_name in [list_word,list_cnt,list_flag]:
    xlcount = 0
    for item in col_name:
        if xlcount <=60000:      
            sheet1.write(xlcount,colcount,item)
            xlcount+=1   
        if xlcount >60000:
            sheet2.write(xlcount_sheet2,colcount,item)
            xlcount_sheet2+=1 
    colcount+=1

time2 = time.time()
print('time:'+str(time2-time1))
guanjianci.save('/Users/zhangyu/Desktop/convert_WH.xls')       





