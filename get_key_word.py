#!/usr/bin/env python
# -*- coding: utf-8 -*-



import xlrd
import xlwt
import time

paths=['/Users/zhangyu/Desktop/new_all_ct.xls']
#分割诊断报告之前，需要将报告中的分隔符统一，武汉同济的报告使用 【中文分号】 


time1 = time.time()

count=0
content=[]
sp='；'
sp=sp.decode('utf-8')
word_dic={}
for path in paths:
    #print(path)
    data=xlrd.open_workbook(path)
    table=data.sheet_by_index(0)
    words=[]
    words_dic={}
    words+=table.col_values(3)
    for index in range(1,len(words)):
        #print(words[index])
        content+=words[index].split(sp)
        
#print(len(content))
#print(len(set(content)))
#content=sorted(content)

content1=set(content)
xlcount=0
xlcount_sheet2=0
guanjianci=xlwt.Workbook(encoding='utf-8')
sheet1=guanjianci.add_sheet('Sheet1')
sheet2=guanjianci.add_sheet('Sheet2')

for item in content1:
    print item,'of',content.count(item)
    print('------------------------------')  
    if xlcount <=60000:      
        sheet1.write(xlcount,0,item)
        sheet1.write(xlcount,1,content.count(item))
        xlcount+=1   
        #if xlcount >2000:
        #    break
    if xlcount >60000:
        sheet2.write(xlcount_sheet2,0,item)
        sheet2.write(xlcount_sheet2,1,content.count(item))
        xlcount_sheet2+=1 
time2 = time.time()
print('time:'+str(time2-time1))
guanjianci.save('/Users/zhangyu/Desktop/ct_guanjianci.xls')       




