#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import time

import numpy as np
import cPickle as pickle

import time

import xlrd
import string
import glob
import csv

import random
import xlwt
random.seed(1337)

if __name__=='__main__':
    print('start...')
    #ypathlist=['/Users/zhangyu/Desktop/同济医院资料/关键字/CT_keyword/all_ct.xls']
    ypathlist=['/Users/zhangyu/Desktop/14_15病房X光标记/15_bingfang.xls']
    format = 'excel'

    textcol=[] 
    valcol=[]

    indtext0=[] 
    indtext1=[]
    indtext2=[]

    risno_text = []       
    patient_id_text = []
    decrtextcol = []
    rawtextcol=[]

    list_risno_val = []   
    list_patient_id = []
    list_dec = []
    list_raw = [] 

    list_risno_val1 = []   
    list_patient_id1 = []
    list_dec1 = []
    list_raw1 = [] 

    #xlspath= '/Users/zhangyu/Desktop/同济医院资料/关键字/CT_keyword/CT_keyword-part1.xls'
    #xlspath2='/Users/zhangyu/Desktop/同济医院资料/关键字/CT_keyword/CT_keyword-part2.xls'
    xlspath= '/Users/zhangyu/Desktop/同济医院资料/关键字/CR_keyword/guanjianci_chest_zy_0720.xls'

    vartab=xlrd.open_workbook(xlspath)
    varlist=vartab.sheet_by_index(0)
    textcol+=varlist.col_values(0) 
    valcol+=varlist.col_values(2)  
    print(len(valcol))

    #vartab2=xlrd.open_workbook(xlspath2)
    #varlist2=vartab2.sheet_by_index(0)
    #textcol+=varlist2.col_values(3) 
    #valcol+=varlist2.col_values(2)  
    #print(len(valcol))

    for index in range(0,len(valcol)):
        if(valcol[index]==0):
            indtext0.append(textcol[index].strip())
            print('0:',textcol[index].strip())
        if(valcol[index]==1):#heart sick
            indtext1.append(textcol[index].strip())
            print('1:',textcol[index].strip())
        if(valcol[index]==2):#lung sick
            indtext2.append(textcol[index].strip())
            print('2:',textcol[index].strip())

    for ypath in ypathlist:
        if format == 'excel':
            wb=xlrd.open_workbook(ypath)
            sh=wb.sheet_by_index(0)

            risno_text+=sh.col_values(0)
            patient_id_text+=sh.col_values(1)
            decrtextcol += sh.col_values(2)
            rawtextcol+=sh.col_values(3)
        else:
            reader = csv.reader(open(ypath,"rU"))    
            risno_text = [row[0] for row in reader]
            reader = csv.reader(open(ypath,"rU"))    
            patient_id_text =   [row[1] for row in reader]
            reader = csv.reader(open(ypath,"rU"))    
            decrtextcol =         [row[2] for row in reader]
            reader = csv.reader(open(ypath,"rU"))    
            rawtextcol =         [row[3] for row in reader]

    #print('data:'+str(len(rawtextcol)))

    part='CHEST'
    position='PA' 


    time_start = time.time()
    randomlist=range(1,len(patient_id_text))
    random.shuffle(randomlist)
    for folderi in randomlist:
        rawtext=rawtextcol[folderi]
        if any(cont in rawtext for cont in indtext0) and not any(cont in rawtext for cont in indtext1) and not any(cont in rawtext for cont in indtext2):
            list_risno_val.append(risno_text[folderi])
            list_patient_id.append(patient_id_text[folderi])
            list_dec.append(decrtextcol[folderi])
            list_raw.append(rawtextcol[folderi]) 
        elif any(cont in rawtext for cont in indtext1) or any(cont in rawtext for cont in indtext2):
            list_risno_val1.append(risno_text[folderi])
            list_patient_id1.append(patient_id_text[folderi])
            list_dec1.append(decrtextcol[folderi])
            list_raw1.append(rawtextcol[folderi]) 

    ###health save
    np_risno = xlwt.Workbook(encoding='utf-8')
    sheet1=np_risno.add_sheet('Sheet1')
    sheet1.write(0,0,'RIS_NO')
    sheet1.write(0,1,'PATIENT_LOCAL_ID')
    sheet1.write(0,2,'DESCRIPTION')
    sheet1.write(0,3,'IMPRESSION')

    colcount = 0
    for content in [ list_risno_val, list_patient_id, list_dec, list_raw]:
        xlcount = 1
        for item in content:      
            sheet1.write(xlcount,colcount,item)
            xlcount+=1   
        colcount += 1
    np_risno.save('/Users/zhangyu/Desktop/15_bingfang_health.xls') 
    ###sick save
    np_risno1 = xlwt.Workbook(encoding='utf-8')
    sheet2=np_risno1.add_sheet('Sheet1')
    sheet2.write(0,0,'RIS_NO')
    sheet2.write(0,1,'PATIENT_LOCAL_ID')
    sheet2.write(0,2,'DESCRIPTION')
    sheet2.write(0,3,'IMPRESSION')

    colcount = 0
    for content in [ list_risno_val1, list_patient_id1, list_dec1, list_raw1]:
        xlcount = 1
        for item in content:      
            sheet2.write(xlcount,colcount,item)
            xlcount+=1   
        colcount += 1
    np_risno1.save('/Users/zhangyu/Desktop/15_bingfang_sick_x_ray.xls') 
        
    time_end = time.time()
    print('runing time is :'+str(time_end-time_start))   


