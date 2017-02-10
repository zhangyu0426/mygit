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
#sudo mount -t cifs //isilon.com/PACSS /mnt/PACSS -o username=tjh 
#需要调整的几个参数分别是xls文件的路径，excel文件名，下载图片保存位置，生成的验证集合excel的文件名，numpy保存位置，numpy保存名称
excelpath = '/home/tuixiangbeijingtest0/workspace/xls/'
excelname = ['sick_1k_sick_in_5w_bingfang_cr.xls','health_1k_health_in_5w_bingfang_cr.xls']  #health and sick excel write here
savepath  = '/media/disk_sda/srv/train_bingfang_1k_cr'
val_excel = 'bingfang_1k_cr_299_'
numpypath = '/media/disk_sda/srv/train_bingfang_1k_cr_numpy/'
numpyname = 'bingfang_1k_cr_299_'
data_len = 1000   



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
    
    list_RIS_NOcol1 = []
    list_PATIENT_LOCAL_IDcol1 = []
    list_DESCRIPTIONcol1 = []
    list_IMPRESSIONcol1=[]
    list_UNC_PATHcol1 = []
    list_FILENAMEcol1 = [] 
    list_FLAGcol1 = []

    for ypath in ypathlist: #ypathlist is datasource
        wb=xlrd.open_workbook(ypath)
        sh=wb.sheet_by_index(0)

        RIS_NOcol+=sh.col_values(0)[1:]
        PATIENT_LOCAL_IDcol+=sh.col_values(1)[1:]

        DESCRIPTIONcol+=sh.col_values(2)[1:]
        IMPRESSIONcol+=sh.col_values(3)[1:]
        UNC_PATHcol+=sh.col_values(4)[1:]
        FILENAMEcol+=sh.col_values(5)[1:]
        FLAGcol+=sh.col_values(6)[1:]

    part='CHEST'
    position='PA' 
    #print(sh.nrows)
    #print(len(PATIENT_LOCAL_IDcol))

    count0=0
    count1=0
    count_train=0
    count_val=0

    X_train, Y_train, X_val, Y_val = [],[],[],[]
    time_start = time.time()
    randomlist=range(len(PATIENT_LOCAL_IDcol))
    #randomlist=range(5)
    random.shuffle(randomlist)
    for folderi in randomlist:
        if count0 == data_len and count1== data_len :
            break
        folder=FILENAMEcol[folderi]
        if len(folder)==0:
            pass
        else:            
            cmdstr1=UNC_PATHcol[folderi]+FILENAMEcol[folderi]
            cmdstr1=cmdstr1.replace(' ','\ ')
            cpstr = 'cp --parents '+cmdstr1+' '+savepath
            os.system(cpstr)   #copy img to savepath
            cmdstr2='gdcmconv -w '+savepath+cmdstr1+'  '+savepath+cmdstr1+'hh' #gdcm2pnm
            os.system(cmdstr2)
            cmd_rm = 'rm '+savepath+cmdstr1
            os.system(cmd_rm)   # delete original dicom file or not
            newpath=savepath+UNC_PATHcol[folderi]+FILENAMEcol[folderi]+'hh'  #img_path
            if os.path.exists(newpath): 
                tmp=dicom.read_file(newpath)
                if hasattr(tmp,'BodyPartExamined') and tmp.BodyPartExamined==part and hasattr(tmp,'ViewPosition') and tmp.ViewPosition==position :              
                    pix=tmp.pixel_array
                    pix=misc.imresize(pix,(newsize[0],newsize[1]))
                    pix=pix-np.mean(pix)
                    #print(pix.shape)
                    #pix=subchannel(pix,7)
                    if FLAGcol[folderi]==0:
                        if count0 == data_len:
                            continue
                        indexremain0=count0%tvt 
                        if indexremain0==0:
                            count_val+=1
                            Y_val.append(0)
                            X_val.append(pix)
                            list_RIS_NOcol.append(RIS_NOcol[folderi])    # export excel & record test information
                            list_PATIENT_LOCAL_IDcol.append(PATIENT_LOCAL_IDcol[folderi])
                            list_DESCRIPTIONcol.append(DESCRIPTIONcol[folderi])
                            list_IMPRESSIONcol.append(IMPRESSIONcol[folderi]) 
                            list_UNC_PATHcol.append(UNC_PATHcol[folderi]) 
                            list_FILENAMEcol.append(FILENAMEcol[folderi])
                            list_FLAGcol.append(0)
                        else:
                            count_train+=1
                            Y_train.append(0)
                            X_train.append(pix)
                        count0+=1
                        print('CR_health:'+str(count0)+'/'+str(len(PATIENT_LOCAL_IDcol)))


                    elif FLAGcol[folderi]==1:
                        if count1== data_len:
                            continue                        
                        indexremain1=count1%tvt 
                        if indexremain1==0:
                            count_val+=1
                            Y_val.append(1)
                            X_val.append(pix)
                            list_RIS_NOcol.append(RIS_NOcol[folderi])    # export excel & record test information
                            list_PATIENT_LOCAL_IDcol.append(PATIENT_LOCAL_IDcol[folderi])
                            list_DESCRIPTIONcol.append(DESCRIPTIONcol[folderi])
                            list_IMPRESSIONcol.append(IMPRESSIONcol[folderi]) 
                            list_UNC_PATHcol.append(UNC_PATHcol[folderi]) 
                            list_FILENAMEcol.append(FILENAMEcol[folderi])
                            list_FLAGcol.append(1)
                        else:
                            count_train+=1
                            Y_train.append(1)
                            X_train.append(pix)
                        count1+=1
                        print('CR_sick:'+str(count1)+'/'+str(len(PATIENT_LOCAL_IDcol)))

    



    print('count0:',count0)
    print('count1:',count1)

    X_train=np.array(X_train)
    X_val=np.array(X_val)

    X_train=X_train.reshape(X_train.shape[0],1,X_train.shape[1],X_train.shape[2])
    X_val  =X_val.reshape(X_val.shape[0],1,X_val.shape[1],X_val.shape[2])

    X_train=np.repeat(X_train,3,axis=1) 
    X_val=np.repeat(X_val,3,axis=1) 

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
    sheet1.write(0,1,'PATIENT_LOCAL_ID')
    sheet1.write(0,2,'DESCRIPTION')
    sheet1.write(0,3,'IMPRESSION')
    sheet1.write(0,4,'UNC_PATH')
    sheet1.write(0,5,'FILENAME')
    sheet1.write(0,6,'TAG')
    
    colcount = 0
    for content in [list_RIS_NOcol, list_PATIENT_LOCAL_IDcol, list_DESCRIPTIONcol, list_IMPRESSIONcol, list_UNC_PATHcol, list_FILENAMEcol,list_FLAGcol]:
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


def operting_numpy():

    #X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    X_normal = np.load('/media/disk_sdb/numpy/ZengZhiZao_zy/CR_1w_chest10565_224_X_train0.npy')
    Y_normal = np.load('/media/disk_sdb/numpy/ZengZhiZao_zy/CR_1w_chest10565_224_Y_train0.npy')
    X_abnormal=np.load('/media/disk_sdb/numpy/ZengZhiZao_zy/CR_1w_chest10376_224_X_train1.npy')
    Y_abnormal=np.load('/media/disk_sdb/numpy/ZengZhiZao_zy/CR_1w_chest10376_224_Y_train1.npy')

    print (Y_normal[0])
    print (Y_abnormal[0])

    X_train_normal = X_normal[0:7000]
    Y_train_normal = Y_normal[0:7000]

    
    X_train_abnormal = X_abnormal[0:7000]
    Y_train_abnormal = Y_abnormal[0:7000]

    
    X_train = np.concatenate((X_train_normal,X_train_abnormal))

    Y_train = np.concatenate((Y_train_normal,Y_train_abnormal))

    
    X_val_normal = X_normal[7000:10000]
    Y_val_normal = Y_normal[7000:10000]

    
    X_val_abnormal = X_abnormal[7000:10000]
    Y_val_abnormal = Y_abnormal[7000:10000]
    
    X_val = np.concatenate((X_val_normal,X_val_abnormal))
    Y_val = np.concatenate((Y_val_normal,Y_val_abnormal))
    
    print("shape of X_train: ",X_train.shape)
    print("shape of Y_train: ",Y_train.shape)
    print("shape of X_val: ",X_val.shape)
    print("shape of Y_val: ",Y_val.shape) 
    

    np.save('/media/disk_sdb/numpy/ZengZhiZao_zy/CR_1w_chest'+str(7000)+'_'+str(224)+'zzz_X_train.npy',X_train)
    np.save('/media/disk_sdb/numpy/ZengZhiZao_zy/CR_1w_chest'+str(7000)+'_'+str(224)+'zzz_Y_train.npy',Y_train)
    np.save('/media/disk_sdb/numpy/ZengZhiZao_zy/CR_1w_chest'+str(3000)+'_'+str(224)+'zzz_X_val.npy',X_val)
    np.save('/media/disk_sdb/numpy/ZengZhiZao_zy/CR_1w_chest'+str(3000)+'_'+str(224)+'zzz_Y_val.npy',Y_val) 

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


