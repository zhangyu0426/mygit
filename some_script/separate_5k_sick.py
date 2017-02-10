#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import time
import Image
import time
import xlrd
import string
import glob
import dicom

import random
import xlwt
random.seed(1337)

excelpath = '/home/tuixiangbeijingtest0/workspace/xls/'
excelname = ['bingfang_16_noduleshadow_part1.xls','bingfang_16_noduleshadow_part2.xls','bingfang_16_noduleshadow_part3.xls']  

savepath = '/mnt/10.35.51.13/wuhan_x_ray/PACSS/jpg/16_nodule/16_bingfang_nodule/' #jpg路径
newsavepath =['/mnt/backup/PACSS/jpg/bingfang_16_noduleshadow_zy/'
			 ,'/mnt/backup/PACSS/jpg/bingfang_16_noduleshadow_whf/'
			 ,'/mnt/backup/PACSS/jpg/bingfang_16_noduleshadow_wsc/']

for i in range(len(excelname)):#循环excel名，每个名字对应的是一个文件夹
    ypath=excelpath+excelname[i]
    PATIENT_LOCAL_IDcol = []

    wb=xlrd.open_workbook(ypath)
    sh=wb.sheet_by_index(0)
    PATIENT_LOCAL_IDcol+=sh.col_values(1)[1:]

    randomlist=range(len(PATIENT_LOCAL_IDcol))
    random.shuffle(randomlist)

    for folderi in randomlist:
        folder=PATIENT_LOCAL_IDcol[folderi]
        if len(folder)==0:
            pass
        else:            
            cmdstr1=savepath+PATIENT_LOCAL_IDcol[folderi]+'.jpg'
            cpstr = 'cp '+cmdstr1+' '+newsavepath[i]#循环excel名，每个名字对应的是一个文件夹
            os.system(cpstr)



