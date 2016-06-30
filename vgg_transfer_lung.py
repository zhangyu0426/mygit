# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:25:59 2015
This creates the file iterator
cd Desktop/workfolder/code/
sudo THEANO_FLAGS=mode=FAST_RUN,device=gpu,cuda.root=/usr/local/cuda,optimizer_including=cudnn,floatX=float32 python cr_cnn_chest.py


@author: tuixiang
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import gzip
import six.moves.cPickle
import sys
from keras.optimizers import SGD, Adagrad, RMSprop, Adadelta
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, AutoEncoder
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
import os 
import scipy.io as sio
import scipy.misc as misc
import matplotlib.pyplot as plt
import time
import cv2
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score


import sys
#sys.path.append('/home/tuixiangbeijingtest0/Desktop/theano/UnbalancedDataset/UnbalancedDataset')
#sys.path.append('/home/tuixiangbeijingtest0/Desktop/theano/JuneCodes')

'''
from unbalanced_dataset.unbalanced_dataset import UnbalancedDataset

from unbalanced_dataset.over_sampling import OverSampler
from unbalanced_dataset.over_sampling import SMOTE

from unbalanced_dataset.under_sampling import UnderSampler
from unbalanced_dataset.under_sampling import TomekLinks
from unbalanced_dataset.under_sampling import ClusterCentroids
from unbalanced_dataset.under_sampling import NearMiss
from unbalanced_dataset.under_sampling import CondensedNearestNeighbour
from unbalanced_dataset.under_sampling import OneSidedSelection
from unbalanced_dataset.under_sampling import NeighbourhoodCleaningRule

from unbalanced_dataset.ensemble_sampling import EasyEnsemble
from unbalanced_dataset.ensemble_sampling import BalanceCascade

from unbalanced_dataset.pipeline import SMOTEENN
from unbalanced_dataset.pipeline import SMOTETomek
'''
from sklearn.cross_validation import train_test_split
#from birnn import *   # for bidirectional lstm

import xlwt
import xlrd
import string
import glob

import codecs
import csv
sys.path.append('/home/tuixiangbeijingtest0/Desktop/workfolder/code/')
import OperatIni
from compiler.ast import flatten
import dicom 


#-------------------------------
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

##################### data ############################################

          
ypathlist=['/home/tuixiangbeijingtest0/Desktop/workfolder/xls/chest_data_3W.xls']
textcol=[]
valcol=[]
indtext2=[]
indtext1=[]
xlspath='/home/tuixiangbeijingtest0/Desktop/workfolder/xls/guanjianci_chest_zy.xls'
vartab=xlrd.open_workbook(xlspath)
varlist=vartab.sheet_by_index(0)
textcol+=varlist.col_values(0)
valcol+=varlist.col_values(2)
indtext0=[]
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
 
    
##################### testing load data ############################################
part='CHEST'
position='PA'
def load_data_1(startpath='/home/tuixiangbeijingtest0/Desktop/workfolder/CR/',
             big_batch=10000,
             verbose=1,
             nb_classes = 2,
             ycolindex=25,  ## the column that we want to investigate
             big_batch_index = 0):

    starttime=time.time()
    rawtextcol=[]
    foldercol=[]
    foldercol1=[]
    for ypath in ypathlist:
        wb=xlrd.open_workbook(ypath)
        sh=wb.sheet_by_index(0)
        rawtextcol+=sh.col_values(3)
        foldercol+=sh.col_values(5)
        foldercol1+=sh.col_values(4)
        #result0=sh.col_values(3)
        #bodypartcol+=sh.col_values(3)
        #ycol+=sh.col_values(ycolindex)
        
    Xtest1=[]
    Xtest2=[]    
    X_overall=[]
    Y_overall=[]
    chestposlist=[]
    rawtextlist=[]
    count=0
    count1=0
    count11=0
    selectcount=0
    selectratio=1

    #xls0=xlwt.Workbook(encoding='utf-8')
    #xls1=xlwt.Workbook(encoding='utf-8')
    #xls2=xlwt.Workbook(encoding='utf-8')
    #sheet0=xls0.add_sheet('Sheet1')
    #sheet1=xls1.add_sheet('Sheet1')
    #sheet2=xls2.add_sheet('Sheet1')
   #xls=xlwt.Workbook(encoding='utf-8')
   #sheet1=xls.add_sheet('sheet1')
    #for folderi in range(1,1000):
    for folderi in range(1,sh.nrows):
        #if count==450 and count1==225 and count11==225:
            #break
        folder=foldercol[folderi]
        if len(folder)==0:
            pass
        else:
            #newpath=glob.glob(startpath+folder+'/Final*')
            newpath='/media/tuixiangbeijingtest0/f989fbc9-24a9-42ce-802e-902125737abd/srv/ruxian2'+foldercol1[folderi]+foldercol[folderi]+'hh'
            #print(newpath)
            if os.path.exists(newpath): 
                #bodypart=bodypartcol[folderi]
                #print(newpath)
                rawtext=rawtextcol[folderi]
                tmp=dicom.read_file(newpath)#dicom.read_file("//home/tuixiangbeijingtest0/Desktop/Image_Decom/image1")
                #bodypart=tmp.BodyPartExamined#bodypart=tmp['Body Part Examined'][0]
                #rawtext=rawtext.replace('.','')
                if hasattr(tmp,'BodyPartExamined') and tmp.BodyPartExamined==part and hasattr(tmp,'ViewPosition') and tmp.ViewPosition==position :#bodypart: #体位 X光
                    #ifbool=True             
                    #for cont in controltextlist:
                        #if not cont in rawtext:
                            #ifbool=False
                            #break
                    #ifbool=True
                    #if str1 in rawtext:
                    if any(cont in rawtext for cont in indtext0) and (not all(cont in rawtext for cont in indtext1)) and (not all(cont in rawtext for cont in indtext2)) and count< 100: 
                    #if (ifbool==True and not any(cont in rawtext for cont in indtext) and not any(cont in rawtext for cont in indtext2)): #or (any(wl in rawtext for wl in wllist) and all(zdy in rawtext for zdy in zdylist)  and not any(cont in rawtext for cont in indtext) and not any(cont in rawtext for cont in indtext2)):        
                        #if any(cont in rawtext for cont in controltextlist):
                        #tmp=sio.loadmat(newpath[0])
                        pos=tmp.ViewPosition#pos=tmp['View Position']
                        
                        if pos=='PA': #部位 X光
                            if selectcount%selectratio==0:
                                #if count>=1000:
                                    #continue
                                rawtextlist.append(rawtext)
                                #sheet0.write(count,0,rawtext)
                                #sheet0.write(count,1,'0')
                                #sheet2.write(count,0,rawtext)
                                #sheet2.write(count,1,'0')
                                chestposlist.append(pos)
                                imageData=cv2.resize(tmp.pixel_array,(224,224))#imageData=cv2.resize(x1.pixel_array,(224,224))
                                #imageData=misc.imresize(imageData,newsize)
                                #imageData=imageData[45:173,60:188]
                                #imageData=misc.imresize(imageData,newsize)
                                X_overall.append(imageData)
                                Y_overall.append(0)
                                Xtest1.append(imageData)
                                count+=1
                                #continue
                                    
                                '''
                                foldername='/home/tuixiangbeijingtest0/Desktop/workfolder/ttest/'+str(count)+'.png'                               
                                plt.clf()
                                plt.xlabel(0)
                                plt.imshow(imageData,cmap=plt.cm.Greys_r)
                                plt.savefig(foldername)
                                '''
                    elif any(cont in rawtext for cont in indtext1) and (not all(cont in rawtext for cont in indtext0)) and count1<50:            
                    #elif any(cont in rawtext for cont in indtext) and not(str2 in rawtext) and not(str3 in rawtext) and not(str4 in rawtext) and not(str5 in rawtext) and not(str6 in rawtext) and count1<550:
                        #tmp=sio.loadmat(newpath[0])
                        pos=tmp.ViewPosition #
                        if pos=='PA':                            
                            selectcount+=1                            
                            #if count1>=1000:
                                #continue
                            rawtextlist.append(rawtext)
                            #sheet1.write(count1,1,rawtext)
                            #sheet1.write(count1,2,'1')
                            #sheet1.write(count1,3,'heart')
                            #sheet2.write(count1+1000,0,rawtext)
                            #sheet2.write(count1+1000,1,'1')
                            chestposlist.append(pos)
                            imageData=cv2.resize(tmp.pixel_array,(224,224))#imageData=cv2.resize(x1.pixel_array,(224,224))
                            #imageData=misc.imresize(imageData,newsize)
                            #imageData=imageData[45:173,60:188]
                            #imageData=misc.imresize(imageData,newsize)
                            
                            X_overall.append(imageData)
                            Y_overall.append(1)
                            Xtest2.append(imageData)
                            count1+=1
                            #continue
                            #print(count1)
                            #print(rawtext)
                            #sheet1.write(count1,1,rawtext)  
                            
                            #elif any(cont in rawtext for cont in indtext2):
                            #’‘’(any(wl in rawtext for wl in wllist) and not any(cont in rawtext for cont in controltextlist)) or‘’‘        
                     
                     #(any(wl in rawtext for wl in wllist) and 
                    #elif ((any(wl in rawtext for wl in wllist) and any(cont in rawtext for cont in indtext2)) or any(cont in rawtext for cont in indtext2)) and not(str2 in rawtext) and count11<500:
                    #elif ((any(wl in rawtext for wl in wllist) and any(cont in rawtext for cont in indtext2)) or any(cont in rawtext for cont in indtext2)) and not(str2 in rawtext) and not(str3 in rawtext) and not(str4 in rawtext) and not(str5 in rawtext) and not(str6 in rawtext) and count11<450: 
                    #elif any(cont in rawtext for cont in indtext2) and not(str2 in rawtext) and not(str3 in rawtext)  and count11<500:
                    elif any(cont in rawtext for cont in indtext2) and (not all(cont in rawtext for cont in indtext0)) and count11<50:                           
                         pos=tmp.ViewPosition #
                         if pos=='PA':                            
                             selectcount+=1
                             rawtextlist.append(rawtext)
                             #sheet1.write(count11+500,1,rawtext)
                             #sheet1.write(count11+500,2,'1')
                             #sheet1.write(count11+500,3,'lung')
                             #sheet2.write(count11+1500,0,rawtext)
                             #sheet2.write(count11+1500,1,'1')
                             chestposlist.append(pos)
                             imageData=cv2.resize(tmp.pixel_array,(224,224))#imageData=cv2.resize(x1.pixel_array,(224,224))
                            #imageData=misc.imresize(imageData,newsize)
                            #imageData=imageData[45:173,60:188]
                            #imageData=misc.imresize(imageData,newsize)
                            
                             X_overall.append(imageData)
                             Y_overall.append(1)
                             Xtest2.append(imageData)
                             count11+=1
                             #continue
                             #print(count1)
                             #print(rawtext)      
                             #sheet1.write(count1,1,rawtext.strip()) 
                             #xls.save('name1_1.xls')
                             '''
                             foldername='/home/tuixiangbeijingtest0/Desktop/workfolder/ttest/'+str(count)+'.png'
                             plt.clf()
                             plt.xlabel(1)
                             plt.imshow(imageData,cmap=plt.cm.Greys_r)
                             plt.savefig(foldername)
                             '''
                             
    #xls0.save('/home/tuixiangbeijingtest0/Desktop/ftp/heart&lung_0.xls')
    #xls1.save('/home/tuixiangbeijingtest0/Desktop/ftp/heart&lung_1.xls')                              
    #xls2.save('/home/tuixiangbeijingtest0/Desktop/ftp/heart&lung_1000.xls')
    X_overall=np.array(X_overall)/5000.0
    
    Xshape=X_overall.shape
    print(Xshape)
    X_overall=X_overall.reshape(Xshape[0],Xshape[1]*Xshape[2])
    Y_overall=np.array(Y_overall)
    if verbose:
        print("total time "+ str(time.time()-starttime))
        unique, counts=np.unique(Y_overall,return_counts=True)
        print(np.mean(np.array(Xtest1)),np.mean(np.array(Xtest2)))
        print(np.std(np.array(Xtest1)),np.std(np.array(Xtest2)))
        print(counts,float(counts[0])/float((counts[1]+counts[0])))
        
        
        
        '''
      #author: yangfan 
      #功能 根据混淆矩阵的索取得对应报告
      #参数 indexlist 混淆矩阵对应的报告
      #返回值 xlsbg.save 含有报告的xls 
      #使用方法 先得到混淆矩阵 然后取消此处的注视 然后运行模型代码 当控制台输出报告生成完毕 可停止模型
      #注意事项 1.保存位置在/home/tuixiangbeijingtest0/Desktop/workfolder/xls/bgxls/下 
               #2.根据不同情况取不同的表名
               #3.哪个模型的混淆矩阵下标list对应哪个模型 不能混淆 否则会出现匹配报告错误的情况 此处要细心
    
    xlsbg=xlwt.Workbook(encoding='utf-8')
    sheetbg=xlsbg.add_sheet('Sheet1')
    X_validate_a=np.load('/home/tuixiangbeijingtest0/Desktop/workfolder/weights/vgg_transfer_lung/1462865076/X_validate.npy')
    xlcount=0
    indexlist=[66, 80, 124, 176, 190, 282, 301, 363, 382]

    

  
    
    print('正在生成报告...')
    for x_over in range(0,len(X_overall)):
        for index in indexlist:
            if(X_overall[x_over]==X_validate_a[index]).all():
                #print(rawtextlist[xcount])
                                
                #for word in indtext2:
                    #if word in rawtextlist[x_over]:
                        #print('new')
                        #print('----------------------------------------------------')
                        #print(rawtextlist[x_over])
                        #print('111:',word)
                        #print('----------------------------------------------------')
                        #print('end')
                        #break
                     
                sheetbg.write(xlcount,1,rawtextlist[x_over])
                xlcount+=1
                #print(rawtextlist[x_over])
    xlsbg.save('/home/tuixiangbeijingtest0/Desktop/workfolder/xls/bgxls/146286507456_15_01.xls')
    print('报告生成完毕') 
    '''
    return X_overall, Y_overall


def simple_CNN_model(image_dense,nb_classes,optim):
    ''' moved down for the looping
    acti='relu'
    dro=0
    inner1=16
    convx=5
    hr=128 ## hidden layer for dense
    hr2=2  ## hidden layer from convolution
    '''
    # model: fits into CNN layer
    model = Sequential()
    
    '''1st meta layer'''
    model.add(Convolution2D(inner1, 3, convx, convx, border_mode='same'))  ### (M+N-1)
    model.add(Activation(acti))
    model.add(Dropout(dro))
    
    
    for covlayeri in range(convlayercount):    
        model.add(Convolution2D(inner1, inner1, convx, convx, border_mode='same'))  ### (M+N-1)
        model.add(Activation(acti))
        model.add(Dropout(dro))

    model.add(Convolution2D(inner1, inner1, convx, convx, border_mode='same'))  ### (M+N-1)
    model.add(Activation(acti))
    if maxpoolcount==4:
        model.add(MaxPooling2D(poolsize=(maxP, maxP)))
    model.add(Dropout(dro))

    '''2nd meta layer'''
    model.add(Convolution2D(inner1, inner1, convx, convx, border_mode='same'))  ### (M+N-1)
    model.add(Activation(acti))
    model.add(Dropout(dro))

    for covlayeri in range(convlayercount):    
        model.add(Convolution2D(inner1, inner1, convx, convx, border_mode='same'))  ### (M+N-1)
        model.add(Activation(acti))
        model.add(Dropout(dro))
      
    model.add(Convolution2D(inner1, inner1, convx, convx,border_mode='same'))  ### the default is valid
    model.add(Activation(acti))
    model.add(MaxPooling2D(poolsize=(maxP, maxP)))
    model.add(Dropout(dro))
    
    '''3rd meta layer'''
    model.add(Convolution2D(inner1, inner1, convx, convx, border_mode='same'))  ### (M+N-1)
    model.add(Activation(acti))
    model.add(Dropout(dro))

    for covlayeri in range(convlayercount):    
        model.add(Convolution2D(inner1, inner1, convx, convx, border_mode='same'))  ### (M+N-1)
        model.add(Activation(acti))
        model.add(Dropout(dro))
        
    model.add(Convolution2D(inner1, inner1, convx, convx,border_mode='same'))  ### the default is valid
    model.add(Activation(acti))
    if maxpoolcount==4:
        model.add(MaxPooling2D(poolsize=(maxP, maxP)))
    model.add(Dropout(dro))

    '''4th meta layer''' 
    model.add(Convolution2D(inner1, inner1, convx, convx, border_mode='same'))  ### (M+N-1)
    model.add(Activation(acti))
    model.add(Dropout(dro))

    for covlayeri in range(convlayercount):    
        model.add(Convolution2D(inner1, inner1, convx, convx, border_mode='same'))  ### (M+N-1)
        model.add(Activation(acti))
        model.add(Dropout(dro))

           
    model.add(Convolution2D(inner1, inner1, convx, convx, border_mode='same'))  ### (M+N-1)
    model.add(Activation(acti))
    model.add(MaxPooling2D(poolsize=(maxP, maxP)))
    model.add(Dropout(dro))
    
    
    ''' FC layer'''
    model.add(Flatten())
    
    model.add(Dense(inner1*image_dense/(maxP**2)**maxpoolcount, hr))
    model.add(Activation(acti))
    model.add(Dropout(dro))

    
    model.add(Dense(hr,nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optim)
    return model

# modified by June:旋转图片
# see rotation of the image
def rotatetionImage(image,angle):
    image_center=tuple(np.array(image.shape)/2)
    rot_mat=cv2.getRotationMatrix2D(image_center,angle,1.0)
    result=cv2.warpAffine(image,rot_mat,image.shape,flags=cv2.INTER_LINEAR)
    #print('result_type:',type(result))
    #print(result.shape)
    #time.sleep(5)
    return result
    
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.val_acc=[]
        
    def on_epoch_end(self,batch,logs={}):
        self.val_acc.append(logs.get('val_acc'))


batch_size=5 #一次 ## note that batch size affect memory overflow problem
nb_classes=2 #分类 
nb_epoch = 1#轮数
pict_ratio=[1,1] 
image_factor=224 #像素 X光
image_height=1 #高 X光
image_rgb=1 #rgb X光

newsize=[x*image_factor for x in pict_ratio]  ## currently 200*160
image_length=newsize[0] #长 X光
image_width=newsize[1] #宽 X光
verbose = 1 
nearest_neigh = 5
random_state = 42 # 分割数据
test_train_ratio = 0.25 # 测试和训练的 分割率

#June:选择是否生成左右轻微旋转的图片
#加大训练量
gen_rot_data=0 # 0 or 1
gen_rot_angle_max=5 #这个病人向左右旋转最大角度
train_test_split_start=1
####################load data######################################
X_overall, Y_overall = load_data_1(big_batch=1000000,
                                verbose=verbose,
                                nb_classes = nb_classes,
                                big_batch_index = 0)
'''                                
plt.clf()
ind=X_overall.shape[0]-1
haha=X_overall[ind,:]
haha=haha.reshape(256,256)
plt.imshow(haha)
'''

 
#plt.show()

## The following are for testing:
# from sklearn.datasets import make_classification
#x, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
#                       n_informative=3, n_redundant=1, flip_y=0,
#                       n_features=500*400, n_clusters_per_class=1,
#                       n_samples=500, random_state=10)
#X_overall = x.reshape(x.shape[0],500*400)
#Y_overall = y

####################balance data ######################################
# -- here you do not need to flattern x, because the first dimension of x is the number of data points        
'''
stime=time.time()    
X_overall, Y_overall = balance_data(X_overall,
                                    Y_overall,
                                    nearest_neigh,
                                    verbose)
print(time.time()-stime,np.unique(Y_overall,return_counts=True))
'''
#####################split data for training and validation #############

rotaDataPath='/home/tuixiangbeijingtest0/Desktop/workfolder/rotaData/'                        
classesname=pyname=str(sys.argv[0][sys.argv[0].rfind(os.sep)+1:]).split('.')[0]
if os.path.exists(rotaDataPath+classesname):
    if len(os.listdir(rotaDataPath+classesname))>=gen_rot_angle_max:
        if os.path.exists(rotaDataPath+classesname+'/'+str(gen_rot_angle_max)+'/train_test_split'):
            if len(os.listdir(rotaDataPath+classesname+'/'+str(gen_rot_angle_max)+'/train_test_split'))>0:
                X_train=np.load(rotaDataPath+classesname+'/'+str(gen_rot_angle_max)+'/train_test_split/X_train.npy')
                X_validate=np.load(rotaDataPath+classesname+'/'+str(gen_rot_angle_max)+'/train_test_split/X_validate.npy')
                Y_train=np.load(rotaDataPath+classesname+'/'+str(gen_rot_angle_max)+'/train_test_split/Y_train.npy')
                Y_validate=np.load(rotaDataPath+classesname+'/'+str(gen_rot_angle_max)+'/train_test_split/Y_validate.npy')
                gen_rot_data=0
                train_test_split_start=0
if train_test_split_start==1:
    X_train, X_validate, Y_train, Y_validate = train_test_split(
                            X_overall, Y_overall,
                            test_size=test_train_ratio,
                            random_state=random_state)
    X_validateSave=X_validate
    Y_validateSave=Y_validate
    #Y_validate=np.load('/home/tuixiangbeijingtest0/Desktop/workfolder/weights/vgg_transfer_lung/1460186895/Y_validate.npy')
    #X_validate=np.load('/home/tuixiangbeijingtest0/Desktop/workfolder/weights/vgg_transfer_lung/1460186895/X_validate.npy')  
        
#---------------load_model------------
def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64,3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dro))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dro))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optim)
    return model

'''
June Edited:to produce even more
data,we want to create more data by
slight variation to the image rotation
是否利用旋转图片大量增加训练图片数量
'''
print('X_train_shape:',X_train.shape)
print('Y_train_shape:',Y_train.shape)

X_train2=[]
Y_train2=[]
#print('Y_overall:',len(Y_overall))
pindex=1
save_max=1
save_list=range(1,gen_rot_angle_max+1)
if gen_rot_data:
    assert(gen_rot_angle_max > 0)
    for idx in range(0,len(Y_train)):
        xo=X_train[idx]
        xo=np.reshape(xo,(image_length,image_width))
        yo=Y_train[idx]
        '''
        print('X_overall:',X_overall)
        print('X_overall_type:',type(X_overall))
        print('X_overall_len:',len(X_overall))
        print('X_overall_len_len:',len(X_overall[0]))
        print('length',image_length)
        print('width',image_width)                
        print('Y_overall:',Y_overall)
        print('Y_overall_type:',type(Y_overall))
        '''
        
        for angle in range(1,gen_rot_angle_max+1):
            rimg=rotatetionImage(xo,angle)
            #b=np.reshape(rimg,(1,np.product(rimg.shape)))
            X_train2.append(rimg)
            Y_train2.append(yo)
            print('idx:',idx,',angle:',angle,',',pindex)
            pindex+=1
            rimg=rotatetionImage(xo,-(angle))
            #b=np.reshape(rimg,(1,np.product(rimg.shape)))
            X_train2.append(rimg)
            Y_train2.append(yo)
            print('idx:',idx,',angle:',-(angle),',',pindex)
            pindex+=1
            
        if len(Y_train2)>=1000 or idx==len(Y_train)-1:
            if not os.path.exists(rotaDataPath+classesname):
                os.mkdir(rotaDataPath+classesname)
                for i in range(1,gen_rot_angle_max+1):
                    os.mkdir(rotaDataPath+classesname+'/'+str(i))
                    os.mkdir(rotaDataPath+classesname+'/'+str(i)+'/X_train')
                    os.mkdir(rotaDataPath+classesname+'/'+str(i)+'/Y_train')   
                    os.mkdir(rotaDataPath+classesname+'/'+str(i)+'/train_test_split')
            X_train_temp=np.array(X_train2)
            X_train_temp_shape=X_train_temp.shape
            print('X_train_temp_shape:',X_train_temp_shape)  
            print('X_train2_shape:',np.reshape(X_train_temp,(X_train_temp_shape[0],X_train_temp_shape[1]*X_train_temp_shape[2])).shape)
            print('Y_train2_shape:',np.array(Y_train2).shape)    
            time.sleep(2)
            if len(os.listdir(rotaDataPath+classesname+'/'+str(save_max)+'/train_test_split'))<1:
                np.save(rotaDataPath+classesname+'/'+str(save_max)+'/X_train/'+str(idx+1)+'.npy',X_train_temp.reshape(X_train_temp_shape[0],X_train_temp_shape[1]*X_train_temp_shape[2]))
                np.save(rotaDataPath+classesname+'/'+str(save_max)+'/Y_train/'+str(idx+1)+'.npy',np.array(Y_train2))
                X_train2=[]
                Y_train2=[]
                
            X_train_save=X_train
            Y_train_save=Y_train
            for i in range(1,save_max+1):
                X_listFile=os.listdir(rotaDataPath+classesname+'/'+str(save_max)+'/X_train')
                Y_listFile=os.listdir(rotaDataPath+classesname+'/'+str(save_max)+'/Y_train')
                if len(X_listFile)>0 and len(Y_listFile)>0:
                    for X_fileName in X_listFile:
                        X_train_save=np.append(X_train_save,np.load(rotaDataPath+classesname+'/'+str(save_max)+'/X_train/'+X_fileName),axis=0)
                    for Y_fileName in Y_listFile:
                        Y_train_save=np.append(Y_train_save,np.load(rotaDataPath+classesname+'/'+str(save_max)+'/Y_train/'+Y_fileName))    
            train_test_listFile=os.listdir(rotaDataPath+classesname+'/'+str(save_max)+'/train_test_split')
            if len(train_test_listFile)<4:
                np.save(rotaDataPath+classesname+'/'+str(save_max)+'/train_test_split/X_train.npy',X_train)
                np.save(rotaDataPath+classesname+'/'+str(save_max)+'/train_test_split/X_validate.npy',X_validate)
                np.save(rotaDataPath+classesname+'/'+str(save_max)+'/train_test_split/Y_train.npy',Y_train)
                np.save(rotaDataPath+classesname+'/'+str(save_max)+'/train_test_split/Y_validate.npy',Y_validate)
            save_max+=1                
unique, counts=np.unique(Y_overall,return_counts=True)
print(counts,float(counts[0])/float((counts[1]+counts[0])))
unique, counts=np.unique(Y_validate,return_counts=True)
print(counts,float(counts[0])/float((counts[1]+counts[0])))
unique, counts=np.unique(Y_train,return_counts=True)
print(counts,float(counts[0])/float((counts[1]+counts[0])))
'''
X_train=X_overall[0:5000,:]
Y_train=Y_overall[0:5000]
X_validate=X_overall[5000:,:]
Y_validate=Y_overall[5000:]
'''
print('X_train:',X_train.shape)
time.sleep(5)

X_train = X_train.reshape(X_train.shape[0], 1 , pict_ratio[0]*image_factor , pict_ratio[1]*image_factor)
X_validate = X_validate.reshape(X_validate.shape[0], 1, pict_ratio[0]*image_factor , pict_ratio[1]*image_factor)

X_train = X_train.astype("float32")
X_train=X_train-np.mean(X_train)
X_validate = X_validate.astype("float32")
X_validate=X_validate-np.mean(X_validate)

X_train=np.repeat(X_train,3,axis=1)      
X_validate=np.repeat(X_validate,3,axis=1)        

Y_train= np_utils.to_categorical(Y_train, nb_classes)
Y_validate_keep=Y_validate
Y_validate= np_utils.to_categorical(Y_validate, nb_classes)



           
method='CNN'
#plt.imshow(X_train[0,0,],cmap = cm.Greys_r)

#imageDense=pict_ratio[0]*pict_ratio[1]*image_factor*image_factor
imageDense=newsize[0]*newsize[1]
print(imageDense)
#model = simple_CNN_model(imageDense,nb_classes)
acti='relu'
#dro=0
#inner1=16
#convx=5
hr=128 ## hidden layer for dense
hr2=2  ## hidden layer from convolution
''' adagrad param 0: inner1=32, hr=32, dro=0, 16 convolution [5] (1 in 4 44 max pool), 2 FC, val_acc 80% after 100'''
''' adagrad param 1: inner1=32, hr=32, dro=0.1, 16 convolution [5] (1 in 4 44 max pool), 2 FC, val_acc 80% after 100'''

drolist=[0,0.25,0.5,0.75]
inner1list=[8,16,32,64]
convxlist=[3,5,7]
hrlist=[16,32,64,128,256]

drolist=[0.7,0.8,0.9]  ### drop out
inner1list=[32]  ### number of filters for conv layers
convxlist=[5]  ### convolution filter size
maxPlist=[4] ### filter size for each max pool
maxpoolcountlist=[4]  ### how many max pool layers to apply
convlayercount=0 ### number of middle layers for a meta conv layer
hrlist=[32]


'''
Lr=0.0000001
Decay=1e-6
Momentum=0.99
Nesterov=True
sgd = SGD(lr=Lr, decay=Decay, momentum=Momentum, nesterov=Nesterov)
adagrad=Adagrad(lr=Lr)
rmsprop=RMSprop(lr=Lr) #lr=0.0001
optim=adagrad
'''


isSaveModel=1 # 1 = trainModel ,other = loadModel
remarktext=u'心肺' #remark
loadModelPath='/home/tuixiangbeijingtest0/Desktop/workfolder/weights/vgg_transfer_lung/1462871961/vgg_transfer_lung1462871961'
val_acc_max=[]
param_hist=[]

drolist=[0]
inner1list=[]
convxlist=[5]
hrlist=[]
convlayercountlist=[0]




#-----------------load_param------------------------------
pyname=str(sys.argv[0][sys.argv[0].rfind(os.sep)+1:])
classesname=pyname.split('.')[0]
ini=OperatIni.manageIni()
if ini.isFileExists(config_file_path=classesname):
    paralist=ini.getIniclasses(config_file_path=classesname,section='modelpara')
    drolist=[paralist[0][1]]
    inner1list=[paralist[1][1]]
    convxlist=[paralist[2][1]]
    hrlist=[paralist[3][1]]
    maxPlist=[paralist[4][1]]
    maxpoolcountlist=[paralist[5][1]]
    convlayercountlist=[paralist[6][1]]
   
#forCount=0     
#for convlayercount in convlayercountlist:
#    for dro, inner1, convx, hr, maxP, maxpoolcount in [(dro,inner1,convx,hr, maxP,maxpoolcount) for maxpoolcount in maxpoolcountlist for maxP in maxPlist for hr in hrlist for dro in drolist for inner1 in inner1list for convx in convxlist]:    
#       forCount+=1
lrlist=[0.0001]
drolist=[0.8] 
methodname='' 
for dro in drolist:
    
    for lr in lrlist:
        Decay=1e-6
        Momentum=0.99
        Nesterov=True
        sgd = SGD(lr=lr, decay=Decay, momentum=Momentum, nesterov=Nesterov)
        rmsprop=RMSprop(lr=lr) #lr=0.000d1
        adagrad=Adagrad(lr=lr)
        ind=0
        #if lr==0.00001 and (dro==0.8 or (dro==0.9 and ind ==1)):
            #continue
        for method in [adagrad,rmsprop,sgd]:
            ind+=1    
            optim=method
            if ind==1:
                
                methodname='adagrad'           
            elif ind==2:
                
                methodname='rmsprop'
            elif ind==3:
                
                methodname='sgd'
                
        
            print('vgg')
            
            model = VGG_16()
            model.load_weights2('/home/tuixiangbeijingtest0/Desktop/workfolder/vgg/vgg16_weights.h5',35)
            #model = simple_CNN_model(imageDense,nb_classes,optim)
            if isSaveModel==1:
                history=LossHistory()
                model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_validate, Y_validate),callbacks=[history])
                val_acc_list=history.val_acc
                max_val_acc=np.max(val_acc_list)  
                val_acc_max.append(max_val_acc)
                #param_hist.append([dro,inner1,convx,hr])
                print(np.max(val_acc_max))
                print(param_hist)
            else:
                if ini.isFileExists(config_file_path=classesname):
                    loadModelPath=ini.getIni(config_file_path=classesname,div_name='modelpath',user_key='loadpath')
                print('load&fitt...')     
                print(loadModelPath)   ####  check loading path is correct
                model.load_weights(loadModelPath)
                history=LossHistory()
                model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_validate, Y_validate),callbacks=[history])
                val_acc_list=history.val_acc
                max_val_acc=np.max(val_acc_list)  
                val_acc_max.append(max_val_acc)
                #param_hist.append([dro,inner1,convx,hr])
                print(np.max(val_acc_max))
                print(param_hist)
            time.sleep(60*15)
              
            #predicted_col=model.predict_classes(X_validate,batch_size=batch_size)
            y_score=model.predict_proba(X_validate,batch_size=batch_size)
            #auc=roc_auc_score(Y_validate,y_score)
            #paratext='\n'+'dro:'+str(dro)+'\n'+'inner1:'+str(inner1)+'\n'+'convx:'+str(convx)+'\n'+'hr:'+str(hr)+'\n'+'maxP:'+str(maxP)+'\n'+'maxpoolcount:'+str(maxpoolcount)+'\n'+'convlayercount:'+str(convlayercount)+'\n'
            paratext='\n'+'lr:'+str(lr)+'\n'+'methodName:'+str(methodname)+'\n'+'dro:'+str(dro)+'\n'
            
            
            
            #------------------------Save-model----------------------------------
            if isSaveModel==1 or isSaveModel==2:
                print('saving...')
                import shutil
                tid=str(int(time.time()))
                codepath='/home/tuixiangbeijingtest0/Desktop/workfolder/code/'
                savepath='/home/tuixiangbeijingtest0/Desktop/workfolder/weights/'
                fullpath=savepath+classesname+'/'+tid+'/'
                
                if os.path.exists(savepath+classesname):
                    os.mkdir(fullpath)
                else:
                    os.mkdir(savepath+classesname)
                    os.mkdir(fullpath)
                savename=classesname
                model.save_weights(fullpath+savename+tid,True)
                shutil.copy(codepath+pyname,fullpath+savename+tid+'.py')
                
                #save validate
                np.save(fullpath+'y_score',y_score)
                np.save(fullpath+'X_validate',X_validateSave)
                np.save(fullpath+'Y_validate',Y_validateSave)
    
                
                paralist=[]
                paralist.append(['modelpath','loadpath',fullpath+savename+tid+'.py'])
                paralist.append(['modelpara','lr',lr])
                '''
                paralist.append(['modelpara','dro',dro]) 
                paralist.append(['modelpara','inner1',inner1])
                paralist.append(['modelpara','convx',convx])
                paralist.append(['modelpara','hr',hr])
                paralist.append(['modelpara','maxP',maxP])
                paralist.append(['modelpara','maxpoolcount',maxpoolcount])
                paralist.append(['modelpara','convlayercount',convlayercount])
                '''
                print(paralist)
                #config_file_path,paralist
                ini.wirteIniAll(config_file_path=fullpath,paralist=paralist)
                
            #time.sleep(120)   
            score = model.evaluate(X_validate, Y_validate,batch_size=batch_size, show_accuracy=True, verbose=1)
            
            
            if isSaveModel==1 or isSaveModel==2: 
                print('saving...')
                ftxt=codecs.open(fullpath+'info.txt','w','utf8')
                ftxt.write('CNN has Validation score:'+ str(score[0])+'\n')
                ftxt.write('CNN has Validation accuracy:'+ str(score[1])+'\n')
                ftxt.write('Remark:'+remarktext+paratext)
                ftxt.close()
                content=[]
                if  os.path.exists(savepath+classesname+'/'+'infolist.csv'):
                    fcsv=open(savepath+classesname+'/'+'infolist.csv','a') 
                else:
                    fcsv=open(savepath+classesname+'/'+'infolist.csv','w')
                    content=['tid','datetime','score','accuracy','modelpath','pypath','dro','inner1','convx','hr','maxP','maxpoolcount','convlayercount','gen_rot_data','gen_rot_angle_max','lr','methodName']  
                writer=csv.writer(fcsv)    
                if len(content)>0:
                    writer.writerow(content)        
                datetime=time.strftime('%Y-%m-%d %H:%M.%S',time.localtime())
            
                
                #content=[tid,datetime,score[0],score[1],fullpath+savename+tid,fullpath+savename+tid+'.py',dro,inner1,convx,hr,maxP,maxpoolcount,convlayercount,gen_rot_data,gen_rot_angle_max,lr,methodname]    
                content=[tid,datetime,score[0],score[1],fullpath+savename+tid,fullpath+savename+tid+'.py',None,None,None,None,None,None,None,None,None,lr,methodname]    
                writer.writerow(content)           
                fcsv.close() 
                
            #----------------------------------------------------------------------------    
            
            if verbose:
                print('CNN has Validation score:', score[0])
                print('CNN has Validation accuracy:', score[1])
                
            del model    
            #time.sleep(60*20)
            '''
            if isSaveModel==1:
                if forCount==80 or forCount==160:
                    time.sleep(60*60)        
            '''
            from matplotlib import font_manager
            if isSaveModel!=1 and isSaveModel!=2:
                fontP=font_manager.FontProperties(fname='/home/tuixiangbeijingtest0/Desktop/ftp/simhei.ttf',size=6)
                filecount=0
                imagespath='/home/tuixiangbeijingtest0/Desktop/workfolder/ttest2/'
                predicted_col=model.predict_classes(X_validate)
                print(predicted_col.shape)
                print(np.mean(Y_validate_keep))
                if not os.path.exists(imagespath+classesname):
                    os.mkdir(imagespath+classesname)
                for checki in range(Y_validate_keep.shape[0]):
                    predicted_val=predicted_col[checki,]
                    actual_val=Y_validate_keep[checki]
                    
                    if predicted_val==actual_val:
                        if filecount>=10:
                            break
                        filecount+=1
                        foldername=imagespath+classesname+'/'+str(checki)+'.png'
                        plt.clf()
                        plt.title('predicted='+str(predicted_val)+'  actual='+str(actual_val))
                        #plt.xlabel(text_validate[checki],fontproperties=fontP)
                        plt.imshow(X_validate[checki,0,:,:],cmap=plt.cm.Greys_r)
                        plt.savefig(foldername)        
                        filename=imagespath+classesname+'/'+str(checki)+'_actual_val'+str(actual_val)+'.npy'
                        np.save(filename,X_validate[checki,0,:,:])
                np.save(imagespath+classesname+'/x_val.npy',X_validate)
                np.save(imagespath+classesname+'/y_val.npy',Y_validate_keep)
'''
from matplotlib import font_manager

fontP=font_manager.FontProperties(fname='/home/tuixiang/Desktop/ftp/simhei.ttf',size=6)
filecount=0
filecount2=0
predicted_col=model.predict_classes(X_validate)
print(predicted_col.shape)
print(np.mean(Y_validate_keep))
for checki in range(Y_validate_keep.shape[0]):
    predicted_val=predicted_col[checki,]
    actual_val=Y_validate_keep[checki]
    if predicted_val==actual_val:
        filecount+=1
        if filecount<10:
            foldername='/home/tuixiang/Desktop/workfolder/ttest2/chestWater/'+str(checki)+'.png'
            plt.clf()
            plt.title('predicted='+str(predicted_val)+'  actual='+str(actual_val))
            #plt.xlabel(text_validate[checki],fontproperties=fontP)
            plt.imshow(X_validate[checki,0,:,:],cmap=plt.cm.Greys_r)
            plt.savefig(foldername)        
            filename='/home/tuixiang/Desktop/workfolder/ttest2/chestWater/'+str(checki)+'_actual_val'+str(actual_val)+'.npy'
            np.save(filename,X_validate[checki,0,:,:])
    else:
        filecount2+=1
        if filecount2<10:
            foldername='/home/tuixiang/Desktop/workfolder/ttest2/chestWater/'+str(checki)+'.png'
            plt.clf()
            plt.title('predicted='+str(predicted_val)+'  actual='+str(actual_val))
            #plt.xlabel(text_validate[checki],fontproperties=fontP)
            plt.imshow(X_validate[checki,0,:,:],cmap=plt.cm.Greys_r)
            plt.savefig(foldername)        
            filename='/home/tuixiang/Desktop/workfolder/ttest2/chestWater/'+str(checki)+'_actual_val'+str(actual_val)+'.npy'
            np.save(filename,X_validate[checki,0,:,:])
np.save('/home/tuixiang/Desktop/workfolder/ttest2/chestWater/x_val.npy',X_validate)
np.save('/home/tuixiang/Desktop/workfolder/ttest2/chestWater/y_val.npy',Y_validate_keep)

'''
'''       
tmp = sio.loadmat('/home/tuixiang/Desktop/Testing/Data/cr/1.2.392.200001.1001.2.3.5.1001.20 (50th copy).110104061114000041/1.2.840.113564.10.1.27927077213875318535139193122152178125178195/matlabOut.mat')
# extract data and filename
imageData = tmp['jpegData']
imageName = tmp['fileName']
imgplot=plt.imshow(imageData,cmap = cm.Greys_r)

### resizing the image
factor=40
ratio=[5,4]
ratio=[x*factor for x in ratio]
resizedImg=misc.imresize(imageData,ratio)
plt.imshow(resizedImg,cmap = cm.Greys_r)
'''

