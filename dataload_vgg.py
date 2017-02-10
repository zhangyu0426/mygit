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

import xlrd
import string
import glob
import dicom
import scipy.io as sio
import scipy.misc as misc
import random

from generic_utils import subchannel

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.


def load_data(newsize=[224,224],tvt=5):
    print('start...')
    starttime = time.time()
    #ypathlist=['/home/tuixiangbeijingtest0/Desktop/workfolder/xls/RX_DATA_rx_all_img_final.xls']
    ypathlist=['/home/tuixiang1/Desktop/workfolder/xls/chest_data_3W.xls']
    textcol=[]
    valcol=[]
    indtext0=[]
    indtext1=[]
    indtext2=[]
    rawtextcol=[]
    foldercol=[]
    foldercol1=[]
    #xlspath='/home/tuixiangbeijingtest0/Desktop/workfolder/xls/guanjianci_single_RX.xls'
    xlspath='/home/tuixiang1/Desktop/workfolder/xls/guanjianci_chest_zy.xls'
    vartab=xlrd.open_workbook(xlspath)
    varlist=vartab.sheet_by_index(0)
    textcol+=varlist.col_values(0)
    valcol+=varlist.col_values(2)

    for index in range(0,len(valcol)):
     if(valcol[index]==4):
        indtext0.append(textcol[index].strip())
        print('0:',textcol[index].strip())
     if(valcol[index]==1):#lung1
        indtext1.append(textcol[index].strip())
        print('1:',textcol[index].strip())
     if(valcol[index]==2):#heart1
        indtext2.append(textcol[index].strip())
        print('1:',textcol[index].strip())
   
    for ypath in ypathlist:
        wb=xlrd.open_workbook(ypath)
        sh=wb.sheet_by_index(0)
        rawtextcol+=sh.col_values(3)
        foldercol+=sh.col_values(5)
        foldercol1+=sh.col_values(4)
        
    part='CHEST'
    position='PA' 
    

    indexcount0=0
    indexcount1=0
    indexcount11=0
    count0=0
    count1=0
    count11=0
    X_train, Y_train, X_val, Y_val, X_test, Y_test=[],[],[],[],[],[]
    
    randomlist=range(1,sh.nrows)
    random.shuffle(randomlist)
    for folderi in randomlist:
        if(count11==3000 and count1==1500 and count0==1500):
            break;
        folder=foldercol[folderi]
        if len(folder)==0:
            pass
        else:
            #newpath=glob.glob(startpath+folder+'/Final*')
            newpath='/srv1/imgdata/'+foldercol1[folderi]+foldercol[folderi]+'hh'
            #print(newpath)
            if os.path.exists(newpath): 
                rawtext=rawtextcol[folderi]
                tmp=dicom.read_file(newpath)
                #if hasattr(tmp,'ImageLaterality') and tmp.ImageLaterality==ImageLaterality:
                if hasattr(tmp,'BodyPartExamined') and tmp.BodyPartExamined==part and hasattr(tmp,'ViewPosition') and tmp.ViewPosition==position :#lung&heart                   
                    if any(cont in rawtext for cont in indtext0) and (not all(cont in rawtext for cont in indtext1)) and (not all(cont in rawtext for cont in indtext2)) and count0<3000:
                        pix=tmp.pixel_array
                        pix=misc.imresize(pix,newsize)
                        pix=pix-np.mean(pix)
                        #pix=subchannel(pix,7)
                        indexremain0=count0%tvt
                        if indexremain0==0:
                            Y_val.append(0)
                            X_val.append(pix)
                        #elif indexremain0==1:
                            Y_test.append(0)
                            X_test.append(pix)
                        else:
                            Y_train.append(0)
                            X_train.append(pix)
                        count0+=1
  
                    elif any(cont in rawtext for cont in indtext1) and (not all(cont in rawtext for cont in indtext0)) and count1<1500:          
                        pix=tmp.pixel_array
                        pix=misc.imresize(pix,newsize)
                        pix=pix-np.mean(pix)
                        #pix=subchannel(pix,7)
                        indexcount1=count1%tvt
                        if indexcount1==0:
                            Y_val.append(1)
                            X_val.append(pix)
                        #elif indexcount1==1:
                            Y_test.append(1)
                            X_test.append(pix)
                        else:
                            Y_train.append(1)
                            X_train.append(pix)
                        count1+=1
                    elif any(cont in rawtext for cont in indtext2) and (not all(cont in rawtext for cont in indtext0)) and count11<1500:                  
                        pix=tmp.pixel_array
                        pix=misc.imresize(pix,newsize)
                        pix=pix-np.mean(pix)
                        #pix=subchannel(pix,7)
                        indexcount11=count11%tvt
                        if indexcount11==0:
                            Y_val.append(1)
                            X_val.append(pix)
                        #elif indexcount11==1:
                            Y_test.append(1)
                            X_test.append(pix)
                        else:
                            Y_train.append(1)
                            X_train.append(pix)
                        count11+=1
                        

    print('count0:',count0)
    print('count1:',count1)
    print('count11:',count11)
 
    X_train=np.array(X_train)
    X_val=np.array(X_val)
    X_test=np.array(X_test)
    
    X_train=X_train.reshape(X_train.shape[0],1,X_train.shape[1],X_train.shape[2])
    X_train=np.repeat(X_train,3,axis=1)
    X_val=X_val.reshape(X_val.shape[0],1,X_val.shape[1],X_val.shape[2])
    X_val=np.repeat(X_val,3,axis=1)
    X_test=X_test.reshape(X_test.shape[0],1,X_test.shape[1],X_test.shape[2])
    X_test=np.repeat(X_test,3,axis=1)
        
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
    np.save('/home/tuixiang1/Desktop/workfolder/Lasagne/numpy/vgg16_1channel/xinfei3000_random_X_train.npy',X_train)
    np.save('/home/tuixiang1/Desktop/workfolder/Lasagne/numpy/vgg16_1channel/xinfei3000_random_Y_train.npy',Y_train)
    np.save('/home/tuixiang1/Desktop/workfolder/Lasagne/numpy/vgg16_1channel/xinfei3000_random_X_val.npy',X_val)
    np.save('/home/tuixiang1/Desktop/workfolder/Lasagne/numpy/vgg16_1channel/xinfei3000_random_Y_val.npy',Y_val)      
    np.save('/home/tuixiang1/Desktop/workfolder/Lasagne/numpy/vgg16_1channel/xinfei3000_random_X_test.npy',X_test)
    np.save('/home/tuixiang1/Desktop/workfolder/Lasagne/numpy/vgg16_1channel/xinfei3000_random_Y_test.npy',Y_test)            
    
    print("total dataloading time "+ str(time.time()-starttime))
    return X_train, Y_train, X_val, Y_val, X_test, Y_test





def load_data1(newsize=[224*2,224*2],tvt=5):
    print('start...')
    starttime = time.time()
    #ypathlist=['/home/tuixiangbeijingtest0/Desktop/workfolder/xls/RX_DATA_rx_all_img_final.xls']
    ypathlist=['/home/tuixiang1/Desktop/workfolder/xls/chest_data_3W.xls']
    textcol=[]
    valcol=[]
    indtext0=[]
    indtext1=[]
    indtext2=[]
    rawtextcol=[]
    foldercol=[]
    foldercol1=[]
    #xlspath='/home/tuixiangbeijingtest0/Desktop/workfolder/xls/guanjianci_single_RX.xls'
    xlspath='/home/tuixiang1/Desktop/workfolder/xls/guanjianci_chest_zy.xls'
    vartab=xlrd.open_workbook(xlspath)
    varlist=vartab.sheet_by_index(0)
    textcol+=varlist.col_values(0)
    valcol+=varlist.col_values(2)

    for index in range(0,len(valcol)):
     if(valcol[index]==4):
        indtext0.append(textcol[index].strip())
        print('0:',textcol[index].strip())
     if(valcol[index]==1):#lung1
        indtext1.append(textcol[index].strip())
        print('1:',textcol[index].strip())
     if(valcol[index]==2):#heart1
        indtext2.append(textcol[index].strip())
        print('1:',textcol[index].strip())
   
    for ypath in ypathlist:
        wb=xlrd.open_workbook(ypath)
        sh=wb.sheet_by_index(0)
        rawtextcol+=sh.col_values(3)
        foldercol+=sh.col_values(5)
        foldercol1+=sh.col_values(4)
        
    part='CHEST'
    position='PA' 
    

    indexcount0=0
    indexcount1=0
    indexcount11=0
    count0=0
    count1=0
    count11=0
    X_train, Y_train, X_val, Y_val, X_test, Y_test=[],[],[],[],[],[]
    
    randomlist=range(1,sh.nrows)
    random.shuffle(randomlist)
    for folderi in randomlist:
        if(count11==3000 and count1==1500 and count0==1500):
            break;
        folder=foldercol[folderi]
        if len(folder)==0:
            pass
        else:
            #newpath=glob.glob(startpath+folder+'/Final*')
            newpath='/srv1/imgdata/'+foldercol1[folderi]+foldercol[folderi]+'hh'
            #print(newpath)
            if os.path.exists(newpath): 
                rawtext=rawtextcol[folderi]
                tmp=dicom.read_file(newpath)
                #if hasattr(tmp,'ImageLaterality') and tmp.ImageLaterality==ImageLaterality:
                if hasattr(tmp,'BodyPartExamined') and tmp.BodyPartExamined==part and hasattr(tmp,'ViewPosition') and tmp.ViewPosition==position :#lung&heart                   
                    if any(cont in rawtext for cont in indtext0) and (not all(cont in rawtext for cont in indtext1)) and (not all(cont in rawtext for cont in indtext2)) and count0<3000:
                        pix=tmp.pixel_array
                        pix=misc.imresize(pix,newsize)
                        pix=pix-np.mean(pix)
                        pix=subchannel(pix,2)
                        indexremain0=count0%tvt
                        if indexremain0==0:
                            Y_val.append(0)
                            X_val.append(pix)
                        #elif indexremain0==1:
                            Y_test.append(0)
                            X_test.append(pix)
                        else:
                            Y_train.append(0)
                            X_train.append(pix)
                        count0+=1
  
                    elif any(cont in rawtext for cont in indtext1) and (not all(cont in rawtext for cont in indtext0)) and count1<1500:          
                        pix=tmp.pixel_array
                        pix=misc.imresize(pix,newsize)
                        pix=pix-np.mean(pix)
                        pix=subchannel(pix,2)
                        indexcount1=count1%tvt
                        if indexcount1==0:
                            Y_val.append(1)
                            X_val.append(pix)
                        #elif indexcount1==1:
                            Y_test.append(1)
                            X_test.append(pix)
                        else:
                            Y_train.append(1)
                            X_train.append(pix)
                        count1+=1
                    elif any(cont in rawtext for cont in indtext2) and (not all(cont in rawtext for cont in indtext0)) and count11<1500:                  
                        pix=tmp.pixel_array
                        pix=misc.imresize(pix,newsize)
                        pix=pix-np.mean(pix)
                        pix=subchannel(pix,2)
                        indexcount11=count11%tvt
                        if indexcount11==0:
                            Y_val.append(1)
                            X_val.append(pix)
                        #elif indexcount11==1:
                            Y_test.append(1)
                            X_test.append(pix)
                        else:
                            Y_train.append(1)
                            X_train.append(pix)
                        count11+=1
                        

    print('count0:',count0)
    print('count1:',count1)
    print('count11:',count11)
 
    X_train=np.array(X_train)
    X_val=np.array(X_val)
    X_test=np.array(X_test)
    
    X_train=X_train[:,0:3,...]
    X_train=X_train.reshape(X_train.shape[0],3,X_train.shape[2],X_train.shape[3])
    X_val=X_val[:,0:3,...]
    X_val=X_val.reshape(X_val.shape[0],3,X_val.shape[2],X_val.shape[3])
    X_test=X_test[:,0:3,...]
    X_test=X_test.reshape(X_test.shape[0],3,X_test.shape[2],X_test.shape[3])
        
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
    np.save('/home/tuixiang1/Desktop/workfolder/Lasagne/numpy/vgg16_3channel/xinfei3000_random_X_train.npy',X_train)
    np.save('/home/tuixiang1/Desktop/workfolder/Lasagne/numpy/vgg16_3channel/xinfei3000_random_Y_train.npy',Y_train)
    np.save('/home/tuixiang1/Desktop/workfolder/Lasagne/numpy/vgg16_3channel/xinfei3000_random_X_val.npy',X_val)
    np.save('/home/tuixiang1/Desktop/workfolder/Lasagne/numpy/vgg16_3channel/xinfei3000_random_Y_val.npy',Y_val)      
    np.save('/home/tuixiang1/Desktop/workfolder/Lasagne/numpy/vgg16_3channel/xinfei3000_random_X_test.npy',X_test)
    np.save('/home/tuixiang1/Desktop/workfolder/Lasagne/numpy/vgg16_3channel/xinfei3000_random_Y_test.npy',Y_test)            
    
    print("total dataloading time "+ str(time.time()-starttime))
    return X_train, Y_train, X_val, Y_val, X_test, Y_test




##################
##################   Execute Code, and get concluded data sets
##################  we get VGG16 1 channel data, & VGG16 3 channel data 

#load_data(newsize=[224,224],tvt=5)
#load_data1(newsize=[224*2,224*2],tvt=5)