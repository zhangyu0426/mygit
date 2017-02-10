#!/usr/bin/env python



from __future__ import print_function
import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle

import lasagne

import scipy.io as sio
import scipy.misc as misc
#import matplotlib.pyplot as plt
import time
#from sklearn.datasets import make_classification

#from sklearn.cross_validation import train_test_split

import xlrd
import string
import glob
import dicom
import scipy.io as sio
import scipy.misc as misc
import random
import xlwt
random.seed(1337)

from generic_utils import subchannel


print('start...')
ypathlist=['/home/tuixiangbeijingtest0/workspace/xls/10w_find_all_sick1.xls','/home/tuixiangbeijingtest0/workspace/xls/10w_find_all_sick2.xls']

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

xlspath='/home/tuixiangbeijingtest0/workspace/xls/10w_find_sick_key_word.xls'
vartab=xlrd.open_workbook(xlspath)
varlist=vartab.sheet_by_index(0)
textcol+=varlist.col_values(0) 
valcol+=varlist.col_values(2)  

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
    wb=xlrd.open_workbook(ypath)
    sh=wb.sheet_by_index(0)

    risno_text+=sh.col_values(0)
    patient_id_text+=sh.col_values(1)
    decrtextcol += sh.col_values(2)
    rawtextcol+=sh.col_values(3)

part='CHEST'
position='PA' 


time_start = time.time()
randomlist=range(1,len(patient_id_text))
random.shuffle(randomlist)
for folderi in randomlist:
    rawtext=rawtextcol[folderi]
    if any(cont in rawtext for cont in indtext1) or any(cont in rawtext for cont in indtext2):
        list_risno_val.append(risno_text[folderi])
        list_patient_id.append(patient_id_text[folderi])
        list_dec.append(decrtextcol[folderi])
        list_raw.append(rawtextcol[folderi]) 



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
np_risno.save('/home/tuixiangbeijingtest0/workspace/xls/all_sick_in_10w_CR.xls') 

    
time_end = time.time()
print('runing time is :'+str(time_end-time_start))

