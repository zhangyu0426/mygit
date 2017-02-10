#!/usr/bin/env python
# -*- coding: utf-8 -*-
#sudo mount -t cifs //isilon.com/PACSS /mnt/PACSS -o username=tjh 
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
excelname = ['非肿瘤_image.xls','鳞癌_image.xls','腺癌_image.xls','小细胞癌_image.xls']  #四类数据的表格
savepath  = '/media/tx-eva/0f75bc71-c713-4ff8-a186-764bd526b22f/Pathology_Classify_four_label/' #图片存放路径
val_excel = 'pathology_lung_four_label'                                              #验证集excel名称
numpypath = '/media/tx-eva/0f75bc71-c713-4ff8-a186-764bd526b22f/numpy/pathology_lung_four_label/' #numpy存放路径
numpyname = 'pathology_lung_four_label'                                                            #numpy名称
pathology_path = '/media/tx-eva/0f75bc71-c713-4ff8-a186-764bd526b22f/Pathology/'      #全部病理图片的路径截止16年年底
data_len = 1200
def data_clean(newsize=[299,299],tvt=5,excelpath=excelpath,excelname=excelname):
    print('start.................................')
    #读取excel中的数据
    ypathlist=[]
    for i in range(len(excelname)):
        ypathlist.append(excelpath+excelname[i])

    RIS_NOcol = []
    PATIENT_LOCAL_IDcol = []
    DESCRIPTIONcol=[]
    IMPRESSIONcol=[]
    UNC_PATHcol=[]
    FILENAMEcol = []
    FLAGcol = []
    
    #define for val
    list_RIS_NOcol = []
    list_PATIENT_LOCAL_IDcol = []
    list_DESCRIPTIONcol = []
    list_IMPRESSIONcol=[]
    list_UNC_PATHcol = []
    list_FILENAMEcol = [] 
    list_FLAGcol = []
    
    for ypath in ypathlist: #ypathlist is datasource
        wb=xlrd.open_workbook(ypath)
        sh=wb.sheet_by_index(0)

        RIS_NOcol+=sh.col_values(0)[1:]
        DESCRIPTIONcol+=sh.col_values(1)[1:]
        IMPRESSIONcol+=sh.col_values(2)[1:]
        UNC_PATHcol+=sh.col_values(5)[1:]
        FILENAMEcol+=sh.col_values(6)[1:]
        FLAGcol+=sh.col_values(7)[1:]

    count0=0
    count1=0
    count2=0
    count3=0
    count_train=0
    count_val=0

    X_train, Y_train, X_val, Y_val = [],[],[],[]
    time_start = time.time()
    randomlist=range(len(RIS_NOcol))
    #randomlist=range(50)
    random.shuffle(randomlist)
    for folderi in randomlist:
        if count0 == data_len and count1== data_len and count2== data_len and count3==data_len:
            break
        folder=FILENAMEcol[folderi]
        if len(folder)==0:
            pass
        else:            
            cmdstr1=pathology_path+str(UNC_PATHcol[folderi])[0:6]+'/'+FILENAMEcol[folderi]
            cmdstr1=cmdstr1.replace(' ','\ ')
            cmdcp = 'cp  '+cmdstr1+' '+savepath
            os.system(cmdcp)   
            if os.path.exists(cmdstr1): 
                tmp=Image.open(cmdstr1)
                pix=np.array(tmp)
                pix=misc.imresize(pix,(newsize[0],newsize[1]))
                pix=pix-np.mean(pix)
                #print(pix.shape)
                if FLAGcol[folderi]==0:
                    if count0 == data_len:
                        continue
                    indexremain0=count0%tvt 
                    if indexremain0==0:
                        count_val+=1
                        Y_val.append([0,0,0,0])
                        X_val.append(pix)
                        list_RIS_NOcol.append(RIS_NOcol[folderi])    # export excel & record test information
                        list_DESCRIPTIONcol.append(DESCRIPTIONcol[folderi])
                        list_IMPRESSIONcol.append(IMPRESSIONcol[folderi]) 
                        list_UNC_PATHcol.append(UNC_PATHcol[folderi]) 
                        list_FILENAMEcol.append(FILENAMEcol[folderi])
                        list_FLAGcol.append(0)
                    else:
                        count_train+=1
                        Y_train.append([0,0,0,0])
                        X_train.append(pix)
                    count0+=1
                    print('health:'+str(count0)+'/'+str(len(RIS_NOcol)))

                elif FLAGcol[folderi]==1:
                    if count1== data_len:
                        continue
                    indexremain1=count1%tvt 
                    if indexremain1==0:
                        count_val+=1
                        Y_val.append([0,1,0,0])
                        X_val.append(pix)
                        list_RIS_NOcol.append(RIS_NOcol[folderi])    # export excel & record test information
                        list_DESCRIPTIONcol.append(DESCRIPTIONcol[folderi])
                        list_IMPRESSIONcol.append(IMPRESSIONcol[folderi]) 
                        list_UNC_PATHcol.append(UNC_PATHcol[folderi]) 
                        list_FILENAMEcol.append(FILENAMEcol[folderi])
                        list_FLAGcol.append(1)
                    else:
                        count_train+=1
                        Y_train.append([0,1,0,0])
                        X_train.append(pix)
                    count1+=1
                    print('lin_ai:'+str(count1)+'/'+str(len(RIS_NOcol)))

                elif FLAGcol[folderi]==2:
                    if count2== data_len:
                        continue
                    indexremain2=count2%tvt 
                    if indexremain2==0:
                        count_val+=1
                        Y_val.append([0,0,1,0])  #numpy标签格式
                        X_val.append(pix)
                        list_RIS_NOcol.append(RIS_NOcol[folderi])    # export excel & record test information
                        list_DESCRIPTIONcol.append(DESCRIPTIONcol[folderi])
                        list_IMPRESSIONcol.append(IMPRESSIONcol[folderi]) 
                        list_UNC_PATHcol.append(UNC_PATHcol[folderi]) 
                        list_FILENAMEcol.append(FILENAMEcol[folderi])
                        list_FLAGcol.append(2) #excel标签格式
                    else:
                        count_train+=1
                        Y_train.append([0,0,1,0])
                        X_train.append(pix)
                    count2+=1
                    print('xian_ai:'+str(count2)+'/'+str(len(RIS_NOcol)))

                elif FLAGcol[folderi]==3:
                    if count3== data_len:
                        continue
                    indexremain3=count3%tvt 
                    if indexremain3==0:
                        count_val+=1
                        Y_val.append([0,0,0,1])  #numpy标签格式
                        X_val.append(pix)
                        list_RIS_NOcol.append(RIS_NOcol[folderi])    # export excel & record test information
                        list_DESCRIPTIONcol.append(DESCRIPTIONcol[folderi])
                        list_IMPRESSIONcol.append(IMPRESSIONcol[folderi]) 
                        list_UNC_PATHcol.append(UNC_PATHcol[folderi]) 
                        list_FILENAMEcol.append(FILENAMEcol[folderi])
                        list_FLAGcol.append(3) #excel标签格式
                    else:
                        count_train+=1
                        Y_train.append([0,0,0,1])
                        X_train.append(pix)
                    count3+=1
                    print('xiaoxibao_ai:'+str(count3)+'/'+str(len(RIS_NOcol)))

    print('count0:',count0)
    print('count1:',count1)
    print('count2:',count2)
    print('count3:',count3)

    X_train=np.array(X_train)
    X_val=np.array(X_val)

    X_train=np.swapaxes(X_train,1,3)
    X_val=np.swapaxes(X_val,1,3)

    print(X_train.shape)

    X_train=X_train.astype('float32')
    X_val=X_val.astype('float32')

    Y_train=np.array(Y_train)
    Y_val=np.array(Y_val)

    Y_train = Y_train.astype(np.uint8)
    Y_val = Y_val.astype(np.uint8)

    print('training dataset has shape', X_train.shape)
    print('validaation dataset has shape', X_val.shape)

    print('train:',X_train.max(),X_train.min(),np.mean(X_train))
    print('val:',X_val.max(),X_val.min(),np.mean(X_val))

    np_risno = xlwt.Workbook(encoding='utf-8')
    sheet1=np_risno.add_sheet('Sheet1')
    sheet1.write(0,0,'RIS_NO')
    sheet1.write(0,1,'DESCRIPTION')
    sheet1.write(0,2,'IMPRESSION')
    sheet1.write(0,3,'UNC_PATH')
    sheet1.write(0,4,'FILENAME')
    sheet1.write(0,5,'FLAG')
    
    colcount = 0
    for content in [list_RIS_NOcol, list_DESCRIPTIONcol, list_IMPRESSIONcol, list_UNC_PATHcol, list_FILENAMEcol,list_FLAGcol]:
        xlcount = 1
        for item in content:      
            sheet1.write(xlcount,colcount,item)
            xlcount+=1   
        colcount += 1
    np_risno.save(excelpath+val_excel+'_'+str(newsize[0])+'validation'+'.xls') 

    np.save(numpypath+numpyname+'_'+str(count_train)+'_'+str(newsize[0])+'_X_train.npy',X_train)
    np.save(numpypath+numpyname+'_'+str(count_val)+'_'+str(newsize[0])+'_X_val.npy',X_val)

    np.save(numpypath+numpyname+'_'+str(count_train)+'_'+str(newsize[0])+'_Y_train.npy',Y_train)
    np.save(numpypath+numpyname+'_'+str(count_val)+'_'+str(newsize[0])+'_Y_val.npy',Y_val)

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


