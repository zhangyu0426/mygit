#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function


"""
    pics_of_one : number of each one should have
    ris_num : number of people we get
目标：
1，肿瘤，良性，正常 三种情况。 1，2，0
2，生成excel保存训练数据，测试数据的人的信息。

原始实现：
1，先读取从数据库取出的信息，此时每个ris_no有48个，然后set去重。
2，模5，将这些ris_no分成训练数据，测试数据，验证数据。
3，循环整体excel表中的数据，把数据分类，分类时先按照0，1分类，再按照第二步的类型分类。
4，执行完第三步的循环，生成numpy文件。

新的实现：
在原来基础上  1，增加病情为良性的。 2，输出validation数据的信息到excel。
1，增加良性的情况，在原来的循环里加入一条分支，和之前的1，0分支并列。在此分支下面加上训练，测试，验证的分类。
2，输出validation的数据到excel表，需要先把这些东西存到list中去（在0，1，2分类下面的val类里append到list中），excel里有几个字段，就有几个list。
先初始化一个xlwt的对象wb_output，然后调wb_output的add_sheet方法，写入excel的若干个字段。然后循环写入数据，遍历每一个list集合，再每个list集合里面循环写入这一列的内容。
"""

import sys
import os
import time

import numpy as np
import time
import cPickle as pickle

import scipy.io as sio
import scipy.misc as misc
#import matplotlib.pyplot as plt
import time
#from sklearn.datasets import make_classification

#from sklearn.cross_validation import train_test_split

import xlrd
import xlwt
import string
import glob
import dicom
import scipy.io as sio
import scipy.misc as misc
import random


from generic_utils import subchannel

import cv2


def load_data1(newsize=[112,112],tvt=5):
    print('start...')
#    ImageLaterality='R' 
    #outputexcelpath = '/home/tuixiangbeijingtest0/Desktop/workfolder/xls/val_output_1300.xls'


    ypathlist=['/home/tuixiangbeijingtest0/Desktop/workfolder/xls/1300_youbing_abnormal_final_5W.xls',
               '/home/tuixiangbeijingtest0/Desktop/workfolder/xls/1300_meibing_normal_final_5W.xls']
               
    textcol=[]
    valcol=[]
    indtext0=[]
    indtext1=[]
    indtext2=[]
    rawtextcol=[]
    foldercol=[]
    foldercol1=[]
    riscol = []
    picnumcol = []
    descriptioncol = []
    impressioncol = []

    list_riscol_val = []
    list_foldercol1_val = []
    list_foldercol_val = []
    list_descriptioncol_val = []
    list_impressioncol_val = []
    list_rawtextcol_val = []


    for ypath in ypathlist: # read the excel
        wb=xlrd.open_workbook(ypath)
        for sheet_i in range(wb.nsheets):
            sh=wb.sheet_by_index(sheet_i)
            if len(sh.col_values(0)) > 1:
                picnumcol += sh.col_values(7)[1:]#slice
                rawtextcol+=sh.col_values(6)[1:]
                foldercol+=sh.col_values(5)[1:] #unique record :file directory
                foldercol1+=sh.col_values(4)[1:]
                riscol+=sh.col_values(0)[1:]
                descriptioncol+=sh.col_values(2)[1:]
                impressioncol+=sh.col_values(3)[1:]
                
    
    rislist=list(set(riscol))#delete the duplicate ris_no
    for one in rislist:
        print(one)
        
        
    print('*******************************************')
    print("count of ris_no: %d"%len(rislist))
    
  
        
    count441_p = 0
    count441_n = 0
    count441_2 = 0    
    
    trainlist, vallist, testlist = [],[],[]
    randomlist = range(0,len(rislist))
    random.shuffle(randomlist)
    for i in randomlist:
        remain = i%5
        if remain == 0:
            vallist.append(rislist[i])
        elif remain == 1:
            testlist.append(rislist[i])
        else:
            trainlist.append(rislist[i])#1/5 var_data 1/5 test_data 3/5 train_data
            
    print("count of trian: %d"%len(trainlist))
    print("count of val: %d"%len(vallist))
    print("count of test: %d"%len(testlist))
    print('*******************************************')

    part='CHEST'
    position='PA' 
    
    dict_ris_pics = {} #count pics number of each ris
    dict_each_X = {}
    num_pics_one = {}
    counts_pics_one = {}
    interval_pics_one = {}
    #dict_each_Y = {}
    for each_ris in rislist:
#        dict_ris_pics[each_ris] = 0
        dict_each_X[each_ris] = []
        #dict_each_Y[each_ris] = []
        num_pics_one[each_ris] = 0

        
        
    count_val = 0
    count_test = 0
    count_train = 0
    count_val1 = 0
    count_test1 = 0
    count_train1 = 0
    
    list_train = []
    list_test = []
    list_val = []
    list_train1 = []
    list_test1= []
    list_val1 = []
 
    loss_list = []
    
    count0=0
    count1=0
    count2=0
    ris_num = 30

    X_train, Y_train, X_val, Y_val, X_test, Y_test=[],[],[],[],[],[]
    randomlist=range(len(rawtextcol))   # 
    for folderi in randomlist:
        folder=foldercol[folderi]
#        if count441_p + count441_n >= 1000 :
#            break
        if len(folder)==0:
            pass
        else:
            newpath='/media/tuixiangbeijingtest0/f989fbc9-24a9-42ce-802e-902125737abd/srv/CT1300'+foldercol1[folderi]+foldercol[folderi]+'hh'
            if os.path.exists(newpath): #path exists,return true. path not exists,return false
                rawtext=rawtextcol[folderi]#0,1 health or sick
                tmp=dicom.read_file(newpath)
                if 1==1:
                    if rawtext==0 :#and count0 < 9000:
                        pix=tmp.pixel_array         
                        pix=cv2.resize(pix,(newsize[0],newsize[1]))
                        pix=pix-np.mean(pix)
                        
                        if riscol[folderi] in vallist:
                            
                            dict_each_X[riscol[folderi]].append(pix)
                            num_pics_one[riscol[folderi]] += 1
                            if  num_pics_one[riscol[folderi]] == 48:
                                Y_val.append(0)
                                X_val.append(dict_each_X[riscol[folderi]])
                                count441_p += 1
                                count0 +=48

                                list_riscol_val.append(riscol[folderi])
                                list_foldercol1_val.append(foldercol1[folderi])
                                list_foldercol_val.append(foldercol[folderi])
                                list_descriptioncol_val.append(descriptioncol[folderi])
                                list_impressioncol_val.append(impressioncol[folderi])
                                list_rawtextcol_val.append(rawtextcol[folderi])

                                print(str(count441_p)+'/'+str(count441_p + count441_n + count441_2))

                        elif riscol[folderi] in testlist:
                            
                            dict_each_X[riscol[folderi]].append(pix)
                            num_pics_one[riscol[folderi]] += 1
                            if  num_pics_one[riscol[folderi]] == 48:
                                Y_test.append(0)
                                X_test.append(dict_each_X[riscol[folderi]])
                                count441_p += 1
                                count0 +=48
                                print(str(count441_p)+'/'+str(count441_p + count441_n + count441_2))
                            
                        else:#train_list
                            
                            dict_each_X[riscol[folderi]].append(pix)
                            num_pics_one[riscol[folderi]] += 1
                            if  num_pics_one[riscol[folderi]] == 48:
                                Y_train.append(0)
                                X_train.append(dict_each_X[riscol[folderi]])
                                count441_p += 1
                                count0 +=48
                                print(str(count441_p)+'/'+str(count441_p + count441_n + count441_2))

                    elif rawtext==1 :#and count1 < 9000:   
                        pix=tmp.pixel_array        
                        pix=cv2.resize(pix,(newsize[0],newsize[1]))
                        pix=pix-np.mean(pix)

                        
                        if riscol[folderi] in vallist:
                            
                            dict_each_X[riscol[folderi]].append(pix)
                            num_pics_one[riscol[folderi]] += 1
                            if  num_pics_one[riscol[folderi]] == 48:
                                Y_val.append(1)
                                X_val.append(dict_each_X[riscol[folderi]])
                                count441_n += 1
                                count1 +=48

                                list_riscol_val.append(riscol[folderi])
                                list_foldercol1_val.append(foldercol1[folderi])
                                list_foldercol_val.append(foldercol[folderi])
                                list_descriptioncol_val.append(descriptioncol[folderi])
                                list_impressioncol_val.append(impressioncol[folderi])
                                list_rawtextcol_val.append(rawtextcol[folderi])

                                print(str(count441_n)+'/'+str(count441_p + count441_n + count441_2))
                                
                        elif riscol[folderi] in testlist:
                            dict_each_X[riscol[folderi]].append(pix)
                            num_pics_one[riscol[folderi]] += 1
                            if  num_pics_one[riscol[folderi]] == 48:
                                Y_test.append(1)
                                X_test.append(dict_each_X[riscol[folderi]])
                                count441_n += 1
                                count1 +=48
                                print(str(count441_n)+'/'+str(count441_p + count441_n + count441_2))
                                
                        else:
                            dict_each_X[riscol[folderi]].append(pix)
                            num_pics_one[riscol[folderi]] += 1
                            if  num_pics_one[riscol[folderi]] == 48:
                                Y_train.append(1)
                                X_train.append(dict_each_X[riscol[folderi]])
                                count441_n += 1
                                count1 +=48
                                print(str(count441_n)+'/'+str(count441_p + count441_n + count441_2))

                    elif rawtext==2 :#and count2 < 9000:
                        pix=tmp.pixel_array         
                        pix=cv2.resize(pix,(newsize[0],newsize[1]))
                        pix=pix-np.mean(pix)
                        
                        if riscol[folderi] in vallist:
                            
                            dict_each_X[riscol[folderi]].append(pix)
                            num_pics_one[riscol[folderi]] += 1
                            if  num_pics_one[riscol[folderi]] == 48:
                                Y_val.append(0)
                                X_val.append(dict_each_X[riscol[folderi]])
                                count441_2 += 1
                                count2 +=48

                                list_riscol_val.append(riscol[folderi])
                                list_foldercol1_val.append(foldercol1[folderi])
                                list_foldercol_val.append(foldercol[folderi])
                                list_descriptioncol_val.append(descriptioncol[folderi])
                                list_impressioncol_val.append(impressioncol[folderi])
                                list_rawtextcol_val.append(rawtextcol[folderi])

                                print(str(count441_2)+'/'+str(count441_p + count441_n + count441_2))

                        elif riscol[folderi] in testlist:
                            
                            dict_each_X[riscol[folderi]].append(pix)
                            num_pics_one[riscol[folderi]] += 1
                            if  num_pics_one[riscol[folderi]] == 48:
                                Y_test.append(0)
                                X_test.append(dict_each_X[riscol[folderi]])
                                count441_2 += 1
                                count2 +=48
                                print(str(count441_2)+'/'+str(count441_p + count441_n + count441_2))
                            
                        else:
                            
                            dict_each_X[riscol[folderi]].append(pix)
                            num_pics_one[riscol[folderi]] += 1
                            if  num_pics_one[riscol[folderi]] == 48:
                                Y_train.append(0)
                                X_train.append(dict_each_X[riscol[folderi]])
                                count441_2 += 1
                                count2 +=48
                                print(str(count441_2)+'/'+str(count441_p + count441_n + count441_2))                                
                                
    for one in rislist:
        if num_pics_one[one] < 48:
            loss_list.append(one)
            
    

    print('count0:',count0)
    print('count1:',count1) 
    print('count2:',count2) 

    print('count441_p:',count441_p)
    print('count441_n:',count441_n) 
    print('count441_2:',count441_2) 

    print(loss_list)     

    X_train=np.array(X_train)
    X_val=np.array(X_val)
    X_test=np.array(X_test)
    
    print("orign x_train shape: "+ str(X_train.shape))
    
    
    
    #X_train1, X_val1, X_test1 = ReSortX(X_train, X_val, X_test)    
    
    #_train1 =  Y_train
    #Y_test1 = Y_test
    #Y_val1 = Y_val
    

    X_train=X_train.reshape(X_train.shape[0], 3, 16, X_train.shape[2],X_train.shape[3])
    X_val=X_val.reshape(X_val.shape[0],3, 16,X_val.shape[2],X_val.shape[3])
    X_test=X_test.reshape(X_test.shape[0],3, 16,X_test.shape[2],X_test.shape[3])
    
    Y_train=np.array(Y_train)
    Y_val=np.array(Y_val)
    Y_test=np.array(Y_test)
    
    
    #Y_train1=np.array(Y_train1)
    #Y_val1=np.array(Y_val1)
    #Y_test1=np.array(Y_test1)
    
    
    randomlist = range(len(X_train))
    random.shuffle(randomlist)
    X_train = X_train[randomlist]
    Y_train = Y_train[randomlist]
    #X_train1 = X_train1[randomlist]
    #Y_train1 = Y_train1[randomlist]
    

    randomlist = range(len(X_val))
    random.shuffle(randomlist)
    X_val = X_val[randomlist]
    Y_val = Y_val[randomlist]
    #X_val1 = X_val1[randomlist]
    #Y_val1 = Y_val1[randomlist]
    
    
    randomlist = range(len(X_test))
    random.shuffle(randomlist)
    X_test = X_test[randomlist]
    Y_test = Y_test[randomlist]
    #X_test1 = X_test1[randomlist]
    #Y_test1 = Y_test1[randomlist]

    X_train=X_train.astype('float32')
    X_val=X_val.astype('float32')
    X_test=X_test.astype('float32')
    #X_train1=X_train1.astype('float32')
    #X_val1=X_val1.astype('float32')
    #X_test1=X_test1.astype('float32')


    
    Y_train = Y_train.astype(np.uint8)
    Y_val = Y_val.astype(np.uint8)
    Y_test = Y_test.astype(np.uint8)
    #Y_train1 = Y_train1.astype(np.uint8)
    #Y_val1 = Y_val1.astype(np.uint8)
    #Y_test1 = Y_test1.astype(np.uint8)
    
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


    '''
    print('----------------------second dataset-----------------------------')
    print('training dataset has shape', X_train1.shape)
    print('validation data has shape', X_val1.shape)
    print('test data has shape', X_test1.shape)
    unique,counts=np.unique(Y_train1,return_counts=True)
    print('training data', counts,float(counts[0])/float((counts[1]+counts[0])))
    unique,counts=np.unique(Y_val1,return_counts=True)
    print('validation data',counts,float(counts[0])/float((counts[1]+counts[0])))
    unique,counts=np.unique(Y_test1,return_counts=True)
    print('testing data',counts,float(counts[0])/float((counts[1]+counts[0])))
    print(X_train1.max(),X_train1.min(),np.mean(X_train1))
    '''

    np.save('/home/tuixiangbeijingtest0/Desktop/workfolder/numpy/ct_zy/ct1300_48arrange_X_train_0714.npy',X_train)
    np.save('/home/tuixiangbeijingtest0/Desktop/workfolder/numpy/ct_zy/ct1300_48arrange_Y_train_0714.npy',Y_train)
    np.save('/home/tuixiangbeijingtest0/Desktop/workfolder/numpy/ct_zy/ct1300_48arrange_X_val_0714.npy',X_val)
    np.save('/home/tuixiangbeijingtest0/Desktop/workfolder/numpy/ct_zy/ct1300_48arrange_Y_val_0714.npy',Y_val)      
    np.save('/home/tuixiangbeijingtest0/Desktop/workfolder/numpy/ct_zy/ct1300_48arrange_X_test_0714.npy',X_test)
    np.save('/home/tuixiangbeijingtest0/Desktop/workfolder/numpy/ct_zy/ct1300_48arrange_Y_test_0714.npy',Y_test) 
    
    wb_output = xlwt.Workbook(encoding='utf-8')
    sheet_output = wb_output.add_sheet('val_output_1300') 
    sheet_output.write(0,0,'ris_no')   
    sheet_output.write(0,1,'path1')
    sheet_output.write(0,2,'path2')
    sheet_output.write(0,3,'description')
    sheet_output.write(0,4,'impression')
    sheet_output.write(0,5,'flag')

    colcount = 0
    for content in [list_riscol_val,list_foldercol1_val,list_foldercol_val,list_descriptioncol_val,list_impressioncol_val,list_rawtextcol_val]:
        xlcount = 1
        for item in content:
            sheet_output.write(xlcount,colcount,item)
            xlcount = xlcount+1
        colcount = colcount+1

    wb_output.save('/home/tuixiangbeijingtest0/Desktop/workfolder/xls/val_output_1300_0714_zy_test.xls')


'''
    np.save('/home/tuixiangbeijingtest0/Desktop/workfolder/numpy/ct_sdy/ct1300_48arrange_interval_X_train.npy',X_train1)
    np.save('/home/tuixiangbeijingtest0/Desktop/workfolder/numpy/ct_sdy/ct1300_48arrange_interval_Y_train.npy',Y_train1)
    np.save('/home/tuixiangbeijingtest0/Desktop/workfolder/numpy/ct_sdy/ct1300_48arrange_interval_X_val.npy',X_val1)
    np.save('/home/tuixiangbeijingtest0/Desktop/workfolder/numpy/ct_sdy/ct1300_48arrange_interval_Y_val.npy',Y_val1)      
    np.save('/home/tuixiangbeijingtest0/Desktop/workfolder/numpy/ct_sdy/ct1300_48arrange_interval_X_test.npy',X_test1)
    np.save('/home/tuixiangbeijingtest0/Desktop/workfolder/numpy/ct_sdy/ct1300_48arrange_interval_Y_test.npy',Y_test1) 




def ReSortX(X_train, X_val, X_test):
    X_train_temp = np.zeros((X_train.shape[0], 3, 16, X_train.shape[2], X_train.shape[3]))
    X_val_temp = np.zeros((X_val.shape[0], 3 ,16, X_val.shape[2], X_val.shape[3]))
    X_test_temp = np.zeros((X_test.shape[0], 3, 16, X_test.shape[2], X_test.shape[3]))
    for i in range(3):
        X_train_temp[:, i, ...] = X_train[:, range(i,48,3), ...]
        X_val_temp[:, i, ...] = X_val[:, range(i,48,3), ...]
        X_test_temp[:, i, ...] = X_test[:, range(i,48,3), ...]
        
    return X_train_temp, X_val_temp, X_test_temp
'''    
    
if __name__ == '__main__':
    load_data1(newsize=[112,112],tvt=5)

