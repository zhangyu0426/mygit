#!/usr/bin/env python
"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes

sudo THEANO_FLAGS=mode=FAST_RUN,device=gpu,cuda.root=/usr/local/cuda,floatX=float32,optimizer_including=conv_meta python cr_cnn_lasagne.py 'cnn' 100 500 'trained_model' 

if encounter errors do
sudo ldconfig /usr/local/cuda/lib64 
"""


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
import cv2
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

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_data(newsize=[224,224],tvt=5):
    ration = 500
    print('start...')
    ypathlist=['/home/tuixiangbeijingtest0/workspace/xls/US_breast_abnormal_0808.xls','/home/tuixiangbeijingtest0/workspace/xls/US_breast_normal_0808.xls']
    textcol=[]
    valcol=[]
    indtext0=[]
    indtext1=[]
    indtext2=[]


    LASTSAVETIME_text = []
    PATIENT_ID_text = []
    EXAM_ITEMSSTRcol=[]
    DESCRIPTIONcol=[]
    IMPRESSIONcol=[]
    filename_col=[]
    hh_col = []
    tag_col = []




    list_PATIENT_ID_train = []
    list_PATIENT_ID_test = []
    '''
    define for val
    '''
    list_LASTSAVETIME_val = []
    list_PATIENT_ID = []
    list_filename = []
    list_tag=[]
    list_DESCRIPTION = []
    list_IMPRESSION = [] 
    list_zore_one = []
    

   
    for ypath in ypathlist: #ypathlist is datasource
        wb=xlrd.open_workbook(ypath)
        sh=wb.sheet_by_index(0)

        LASTSAVETIME_text+=sh.col_values(0)[1:]
        PATIENT_ID_text+=sh.col_values(1)[1:]
        EXAM_ITEMSSTRcol+= sh.col_values(2)[1:]
        DESCRIPTIONcol+=sh.col_values(3)[1:]
        IMPRESSIONcol+=sh.col_values(4)[1:]
        filename_col+=sh.col_values(5)[1:]
        hh_col+=sh.col_values(6)[1:]
        tag_col+=sh.col_values(7)[1:]



        
    #part='CHEST'
    #position='PA' 
    #print(sh.nrows)
    #print(len(LASTSAVETIME_text))

    indexcount0=0
    indexcount1=0
    count0=0
    count1=0
    X_train, Y_train, X_val, Y_val, X_test, Y_test=[],[],[],[],[],[]
    time_start = time.time()
    randomlist=range(len(LASTSAVETIME_text))
    random.shuffle(randomlist)
    for folderi in randomlist:
        if(count1==ration and count0==ration):
            break
        folder=filename_col[folderi]
        if len(folder)==0:
            pass
        else:
            newpath='/media/disk_sda/srv/US/mnt/US/usimage1  BAK/'+LASTSAVETIME_text[folderi]+r'/'+PATIENT_ID_text[folderi]+r'/'+filename_col[folderi]+'hh'   #img_path
            if os.path.exists(newpath): 
                tmp=dicom.read_file(newpath)
                if 1==1:# hasattr(tmp,'BodyPartExamined') and tmp.BodyPartExamined==part and hasattr(tmp,'ViewPosition') and tmp.ViewPosition==position :              
                    if tag_col[folderi]==0:
                        try:
                            pix=tmp.pixel_array
                            pix = pix[:,100:,100:]
                            pix=np.swapaxes(pix,0,2)
                            print('origin_0:',pix.shape)
                            pix=cv2.resize(pix,(newsize[0],newsize[1]),interpolation=cv2.INTER_AREA)
                            pix=pix-np.mean(pix)
                            print('final_0:',pix.shape)
                            #pix=subchannel(pix,7)
                        except: continue
                        indexremain0=count0%tvt 
                        if indexremain0==0:
                            Y_val.append(0)
                            X_val.append(pix)
                            list_LASTSAVETIME_val.append(LASTSAVETIME_text[folderi])    # export excel & record test information
                            list_PATIENT_ID.append(PATIENT_ID_text[folderi])
                            list_DESCRIPTION.append(DESCRIPTIONcol[folderi])
                            list_IMPRESSION.append(IMPRESSIONcol[folderi]) 
                            list_filename.append(filename_col[folderi]) 
                            list_tag.append(tag_col[folderi])
                            list_zore_one.append(0)
                            #print(X_val)
                        elif indexremain0==1:
                            Y_test.append(0)
                            X_test.append(pix)
                            list_PATIENT_ID_test.append(PATIENT_ID_text[folderi])
                            #print(X_test)
                        else:
                            Y_train.append(0)
                            X_train.append(pix)
                            list_PATIENT_ID_train.append(PATIENT_ID_text[folderi])
                            #print(X_train)
                        count0+=1
                        print('US_breast_normal:'+str(count0)+'/'+str(ration))
  
                    elif tag_col[folderi]==1: 
                        try:
                            pix=tmp.pixel_array
                            pix = pix[:,100:,100:]
                            pix=np.swapaxes(pix,0,2)
                            print('origin_1:',pix.shape)
                            pix=cv2.resize(pix,(newsize[0],newsize[1]),interpolation=cv2.INTER_AREA)
                            pix=pix-np.mean(pix)
                            print('final_1:',pix.shape)
                            #pix=subchannel(pix,7)
                        except:continue
                        indexcount1=count1%tvt
                        if indexcount1==0:
                            Y_val.append(1)
                            X_val.append(pix)
                            list_LASTSAVETIME_val.append(LASTSAVETIME_text[folderi])    # export excel & record test information
                            list_PATIENT_ID.append(PATIENT_ID_text[folderi])
                            list_DESCRIPTION.append(DESCRIPTIONcol[folderi])
                            list_IMPRESSION.append(IMPRESSIONcol[folderi]) 
                            list_filename.append(filename_col[folderi]) 
                            list_tag.append(tag_col[folderi])
                            list_zore_one.append(1)
                        elif indexcount1==1:
                            Y_test.append(1)
                            X_test.append(pix)
                            list_PATIENT_ID_test.append(PATIENT_ID_text[folderi])
                        else:
                            Y_train.append(1)
                            X_train.append(pix)
                            list_PATIENT_ID_train.append(PATIENT_ID_text[folderi])
                        count1+=1
                        print('US_breast_cancer:'+str(count1)+'/'+str(ration))

                        

    print('count0:',count0)
    print('count1:',count1)
 
    X_train=np.array(X_train)
    X_val=np.array(X_val)
    X_test=np.array(X_test)
    

    X_train=np.swapaxes(X_train,1,3)
    X_val=np.swapaxes(X_val,1,3)
    X_test=np.swapaxes(X_test,1,3)

    #X_train=X_train[:,0:48,...]
    print(X_train.shape)

#    X_val=X_val[:,0:48,...]
#    X_test=X_test[:,0:48,...]

    #X_train=np.repeat(X_train,3,axis=1) 
    #X_val=np.repeat(X_val,3,axis=1) 
    #X_test=np.repeat(X_test,3,axis=1)     

    X_train=X_train.astype('float32')
    X_val=X_val.astype('float32')
    X_test=X_test.astype('float32')

    Y_train=np.array(Y_train)
    Y_val=np.array(Y_val)
    Y_test=np.array(Y_test)
    Y_train = Y_train.astype(np.uint8)
    Y_val = Y_val.astype(np.uint8)
    Y_test = Y_test.astype(np.uint8)
    print('training dataset has shape', X_train.shape)
    print('validation data has shape', X_val.shape)
    print('test data has shape', X_test.shape)
    unique,counts=np.unique(Y_train,return_counts=True)
    print('training data', counts,float(counts[0])/float((counts[1]+counts[0])))
    unique,counts=np.unique(Y_val,return_counts=True)
    print('validation data',counts,float(counts[0])/float((counts[1]+counts[0])))
    unique,counts=np.unique(Y_test,return_counts=True)
    print('testing data',counts,float(counts[0])/float((counts[1]+counts[0])))
    print(X_train.max(),X_train.min(),np.mean(X_train))
 





    np_risno = xlwt.Workbook(encoding='utf-8')
    sheet1=np_risno.add_sheet('Sheet1')
    sheet1.write(0,0,'LASTSAVETIME')
    sheet1.write(0,1,'PATIENT_ID')
    sheet1.write(0,2,'DESCRIPTION')
    sheet1.write(0,3,'IMPRESSION')
    sheet1.write(0,4,'filename')
    sheet1.write(0,5,'tag')

    colcount = 0
    for content in [list_LASTSAVETIME_val, list_PATIENT_ID, list_DESCRIPTION, list_IMPRESSION, list_filename, list_tag]:
        xlcount = 1
        for item in content:      
            sheet1.write(xlcount,colcount,item)
            xlcount+=1   
        colcount += 1
    np_risno.save('/home/tuixiangbeijingtest0/workspace/xls/cuted_US_Breast_validation_patientID'+str(ration)+'_'+str(newsize[0])+'.xls') 

    np.save('/media/disk_sdb/numpy/US_zy/arer_cv2_cuted_Breast'+str(ration)+'_'+str(newsize[0])+'random_X_train.npy',X_train)
    np.save('/media/disk_sdb/numpy/US_zy/arer_cv2_cuted_Breast'+str(ration)+'_'+str(newsize[0])+'random_Y_train.npy',Y_train)
    np.save('/media/disk_sdb/numpy/US_zy/arer_cv2_cuted_Breast'+str(ration)+'_'+str(newsize[0])+'random_X_val.npy',X_val)
    np.save('/media/disk_sdb/numpy/US_zy/arer_cv2_cuted_Breast'+str(ration)+'_'+str(newsize[0])+'random_Y_val.npy',Y_val)      
    np.save('/media/disk_sdb/numpy/US_zy/arer_cv2_cuted_Breast'+str(ration)+'_'+str(newsize[0])+'random_X_test.npy',X_test)
    np.save('/media/disk_sdb/numpy/US_zy/arer_cv2_cuted_Breast'+str(ration)+'_'+str(newsize[0])+'random_Y_test.npy',Y_test)            
    time_end = time.time()
    print('runing time is :'+str(time_end-time_start))
    


load_data()



