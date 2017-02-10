#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function


"""
    pics_of_one : number of each one should have
    ris_num : number of people we get
    sudo THEANO_FLAGS=mode=FAST_RUN,device=gpu0,cuda.root=/usr/local/cuda,floatX=float32,optimizer_including=cudnn python get_feature2.py
"""

import sys
import os
import time

import numpy as np
import time
import cPickle as pickle

import scipy.io as sio
import scipy.misc as misc
import time
import xlrd
import csv
import string
import glob
import dicom
import random
import cv2
import Image
from googlenet import google_build, read_model_param
import lasagne
import theano
import theano.tensor as T
import logging
#sys.path.append('/home/tuixiang1/workspace/code_whf/gqiang_lstm_mask_lung_3d_attention.git/')

dicom_dir1='/media/disk_sdb/CT1500_1010/all_ct_2013'
dicom_dir2='/media/disk_sdb/CT1500_1010/all_ct_2014'
dicom_dir3='/media/disk_sdb/CT1500_1010/all_ct_2015'
dicom_dir4='/media/disk_sdb/CT1500_1010/all_ct_2016'
dicom_dir=[dicom_dir1,dicom_dir2,dicom_dir3,dicom_dir4]

ypathlist=['/media/disk_sdb/CT1500_1010/1500_sick_path_60w.csv']
select_num=100
npsave_dir='/media/disk_sdb/numpy/feature'
newsize=[299,299]
googlenet_path='save/googlenet.pkl'
logpath='/home/tuixiangbeijingtest0/workspace/var_lstm/log_sick3.txt'
#mask=500
#
model = google_build()
layers=lasagne.layers.get_all_layers(model['loss3/classifier'])
read_model_param(layers, googlenet_path)
print('feature extraction param loaded')
X=T.tensor4('input')
output=lasagne.layers.get_output(model['pool5/7x7_s1'], X)
get_fun=theano.function(inputs=[X], outputs=output, allow_input_downcast=True)

def saveData(path,name,data):
    savePath = path+'/'+name
    with open(savePath, 'w') as f:
        pickle.dump(data,f)

def create_logging(filepath='info.log'):
    import logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=filepath,
                        filemode='w')
    #print('create logging success')     
    #publicVar.logging=logging
    return logging
    
def get1024(input_x, gap):
        '''
        #load pretrained model and get 1024 feature of a pic
        input_x=np.reshape(input_x, (-1, 3, 299, 299))
        model = google_build()
        layers=get_all_layers(model['loss3/classifier'])
        read_model_param(layers, googlenet_path)

        X=T.tensor4('input')
        output=get_output(model['pool5/7x7_s1'], X)
        get_fun=theano.function(inputs=[X], outputs=output, allow_input_downcast=True)
        '''
        input_length = len(input_x)
        steps = input_length/gap
        leftover = input_length%gap
        rec=[]
        for i in range(steps):
            tmp=get_fun(input_x[i*gap:(i+1)*gap])
            if i==0:
                rec=tmp
            else:
                rec=np.concatenate((rec, tmp), axis=0)
        lo=get_fun(input_x[-leftover:])
        if len(rec)>0:
            rec=np.concatenate((rec, lo), axis=0)
        else:
            rec=lo
        print(input_length, rec.shape)
        return rec
  
def pix_process(filepath):
    
    tmp=dicom.read_file(filepath)
    inum=tmp.InstanceNumber
    thick = tmp.SliceThickness
    if float(thick) ==1.25:
        os.remove(filepath)
        pix=tmp.pixel_array
        center=-600
        width=1500
        slop=tmp.RescaleSlope
        intercept=tmp.RescaleIntercept
        pix = pix*slop+intercept
        low=center-width/2
        hig=center+width/2
        pix_out=np.zeros(pix.shape)
        w1=np.where(pix>low) and np.where(pix<hig)
        pix_out[w1]=((pix[w1]-center+0.5)/(width-1)+0.5)*255
        pix_out[np.where(pix<=low)]=pix[np.where(pix<=low)]=0
        pix_out[np.where(pix>=hig)]=pix[np.where(pix>=hig)]=255
        pxl = pix_out.astype('uint8')
        pxl = Image.fromarray(pxl)
        pxl = pxl.convert('RGB')
        img=misc.imresize(pxl, (299, 299, 3)) 
        img =np.swapaxes(img, 1, 2)
        img=np.swapaxes(img, 0, 1)
        img=img-np.mean(img)    
        return img,inum
    else:
        return 0,0



def load_data(dicom_dir,ypathlist,select_num,npsave_dir,newsize):
    print('start...')
               
    rawtextcol=[]
    foldercol=[]
    foldercol1=[]
    riscol = []
    personCount = 0
#    for ypath in ypathlist:
#        wb=xlrd.open_workbook(ypath)
#        for sheet_i in range(wb.nsheets):
#            sh=wb.sheet_by_index(sheet_i)
#            if len(sh.col_values(0)) > 1:
#                rawtextcol+=sh.col_values(6)[1:]
#                foldercol+=sh.col_values(5)[1:]
#                foldercol1+=sh.col_values(4)[1:]
#                riscol+=sh.col_values(0)[1:]
    #read csv
    for ypath in ypathlist:
#        print(ypath)
        reader = csv.reader(open(ypath,"rU"))
        foldercol1 += [row[1] for row in reader][1:]
        reader = csv.reader(open(ypath,"rU"))
        foldercol  += [row[2] for row in reader][1:]
        reader = csv.reader(open(ypath,"rU"))
        riscol     += [row[0] for row in reader][1:]#It's patient_local_id
        
    rislist=list(set(riscol))
    print('the number of patients: ',len(rislist))
    ris_path={}
    ris_path_num={}
    ris_fea={}
    Y=[]
    fea_total=[]
    ris_total=[]
    ris_inumSet_total=[]
    cnt=0
    for ris in rislist:
        ris_path[ris]=[]
        ris_path_num[ris]=0
        ris_fea[ris]=[]
        
        dicom_dir1=dicom_dir[0]
        dicom_dir2=dicom_dir[1]
        dicom_dir3=dicom_dir[2]
        dicom_dir4=dicom_dir[3]
        
        # temp = '/home/tuixiangbeijingtest0/workspace/var_lstm/temp'
    log = create_logging(logpath)
    for ris in rislist[len(rislist)/3*2:]:
        for i in range(len(riscol)):
            if riscol[i] == ris:
                #decompress dicom compare path
                dicom_path1=dicom_dir1+foldercol1[i]+foldercol[i]
                dicom_path2=dicom_dir2+foldercol1[i]+foldercol[i]
                dicom_path3=dicom_dir3+foldercol1[i]+foldercol[i]
                dicom_path4=dicom_dir4+foldercol1[i]+foldercol[i]
                if (os.path.isfile(dicom_path1)):
                    dicom_path1=dicom_path1.replace(' ', '\ ')
                    strgdcm='gdcmconv -w '+dicom_path1+' '+dicom_path1+'hh'
#                    print(strgdcm)
                    os.system(strgdcm)
                    dicom_path = dicom_path1+'hh'
                elif (os.path.isfile(dicom_path2)):
                    dicom_path2=dicom_path2.replace(' ', '\ ')
                    strgdcm='gdcmconv -w '+dicom_path2+' '+dicom_path2+'hh'
#                    print(strgdcm)
                    os.system(strgdcm)
                    dicom_path = dicom_path2+'hh'
                elif (os.path.isfile(dicom_path3)):
                    dicom_path3=dicom_path3.replace(' ', '\ ')
                    strgdcm='gdcmconv -w '+dicom_path3+' '+dicom_path3+'hh'
#                    print(strgdcm)
                    os.system(strgdcm)
                    dicom_path = dicom_path3+'hh'
                elif (os.path.isfile(dicom_path4)):
                    dicom_path4=dicom_path4.replace(' ', '\ ')
                    strgdcm='gdcmconv -w '+dicom_path4+' '+dicom_path4+'hh'
#                    print(strgdcm)
                    os.system(strgdcm)
                    dicom_path = dicom_path4+'hh'
                else:
#                    print(dicom_path1)
                    continue
              #  dicom_path=dicom_path+'hh'
                dicom_path=dicom_path.replace('\ ',' ')
                if os.path.exists(dicom_path):
#                    print('find hh')
                    ris_path[ris].append(dicom_path)
                    ris_path_num[ris] += 1
#                    os.remove(dicom_path)#delete dicom
        print(len(ris_path[ris]),ris_path_num[ris])
        if ris_path_num[ris]>100:           
            imglist=[]
            inumlist=[]
            for j in range(ris_path_num[ris]):
    #            print(ris_path[ris][j])
                try:
                    img,inum =pix_process(ris_path[ris][j])
        #            print(img.shape)
                    if inum != 0:
                        inumlist.append(inum)
                        imglist.append(img)
                    else:
                        continue
#                        ris_path_num[ris] -=1
#                        ris_path[ris].remove(ris_path[ris][j])
                except Exception as e:
                    log.info(e)
                    log.info(ris_path[ris][j])
            
            inumlist = np.asarray(inumlist)
            imglist = np.asarray(imglist)
            
            resortList = np.argsort(inumlist)
            
            inumlist = inumlist[resortList]
            imglist = imglist[resortList]
            
            inumlist = inumlist.tolist()
            imglist = imglist.tolist()   
            
            personCount +=1
            
            try:
                fea=get1024(imglist,50)
                Y.append(1)
                
                fea_tmp=np.array(fea)
                fea_tmp=np.reshape(fea_tmp,(fea_tmp.shape[0],1024))
                print(len(fea),fea_tmp.shape)
                
                fea_total.append(fea_tmp)
                ris_inumSet_total.append(inumlist)
                ris_total.append(ris)
                cnt += 1
#                print(ris,len(ris_path_num[ris]))
                print('current patients: ',cnt)
            except Exception as e:
                log.info(e)
                log.info(ris)


            if cnt%10==0:
                print(cnt,len(fea_total),len(Y))
                Y1=np.array(Y)
                Y1=Y1.astype(np.uint8)
                print(Y1)
                all={}
                all['ris']=ris_total
                all['ris InstancesNumber set']=ris_inumSet_total
                all['feature']=fea_total
                all['Y']=Y1
                saveData(npsave_dir,'feature_cnt'+str(cnt/2)+'.pick',all)
    
    print(len(fea_total),len(Y))
    Y=np.array(Y)
    Y=Y.astype(np.uint8)
    print(Y)
    all={}
    all['ris']=ris_total
    all['ris InstancesNumber set']=ris_inumSet_total
    all['feature']=fea_total
    all['Y']=Y
    saveData(npsave_dir,'feature_sick3.pkl',all)
    # np.save(npsave_dir+'X_fea_normal.npy',fea_total)
    # np.save(npsave_dir+'Y_fea_normal.npy',Y)
    print("qualified num:",personCount)
    print("all num:",len(rislist))
    
                
load_data(dicom_dir,ypathlist,select_num,npsave_dir,newsize)





























