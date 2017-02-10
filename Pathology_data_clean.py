#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import,print_function
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import time
import Image
import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
import matplotlib.pyplot as plt
import lasagne
import scipy.io as sio
import scipy.misc as misc
import scipy 
print (scipy.__version__)
import time
import xlrd
import string
import glob
import dicom
import scipy.io as sio
import scipy.misc as misc
import random
import xlwt
random.seed(1337)

excelpath = '/home/tx-eva/workspace/xls/'
excelname = ['非肿瘤_image.xls','鳞癌_image.xls','腺癌_image.xls','小细胞癌_image.xls']  #health and sick excel write here
savepath  = '/media/tx-eva/0f75bc71-c713-4ff8-a186-764bd526b22f/Pathology_Classify/'
folder_name = ['health','lin_cancer','xian_cancer','xiaoxibao_cancer']
pathology_path = '/media/tx-eva/0f75bc71-c713-4ff8-a186-764bd526b22f/Pathology/'
data_len = 1200   
def data_clean(newsize=[299,299],tvt=5,excelpath=excelpath,excelname=excelname):
    print('start.................................')

    for i in range(len(excelname)):
        ypath = excelpath+excelname[i]

        RIS_NOcol = []
        PATIENT_LOCAL_IDcol = []
        DESCRIPTIONcol=[]
        IMPRESSIONcol=[]
        UNC_PATHcol=[]
        FILENAMEcol = []
        FLAGcol = []

        wb=xlrd.open_workbook(ypath)
        sh=wb.sheet_by_index(0)
        RIS_NOcol+=sh.col_values(0)[1:]
        UNC_PATHcol+=sh.col_values(5)[1:]
        FILENAMEcol+=sh.col_values(6)[1:]

        count = 0
        time_start = time.time()
        randomlist=range(len(RIS_NOcol))
        #random.shuffle(randomlist)
        for folderi in randomlist:
            if count == data_len :
                break
            folder=FILENAMEcol[folderi]
            if len(folder)==0:
                pass
            else:            
                cmdstr1=pathology_path+str(UNC_PATHcol[folderi])[0:6]+'/'+FILENAMEcol[folderi]
                cmdstr1=cmdstr1.replace(' ','\ ')
                cmdcp = 'cp  '+cmdstr1+' '+savepath+folder_name[i]
                os.system(cmdcp)  
                count +=1 
                print(count)
                print(ypath)

        print('count:',count)
        time_end = time.time()
        print('runing time is :'+str(time_end-time_start))




if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        #print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print("you need modify excelname,val_excel,numpypath,numpyname")
    else:
        kwargs = {}
        #if len(sys.argv) > 1:
        #    kwargs['model'] = sys.argv[1]
        #if len(sys.argv) > 2:
        #    kwargs['mini_batch'] = int(sys.argv[2])
        #    kwargs['num_epochs'] = int(sys.argv[3])
        #    kwargs['dro']=float(sys.argv[4])
        #    kwargs['lr']=float(sys.argv[5])
        data_clean(**kwargs)


