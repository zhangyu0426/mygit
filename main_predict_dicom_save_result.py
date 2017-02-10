#!/usr/bin/env python

"""
This is the function used to predict X-ray dicom files.

Beijing: THEANO_FLAGS=mode=FAST_RUN,device=gpu1,cuda.root=/usr/local/cuda,floatX=float32,optimizer_including=conv_meta predict.py'

Wuhan: THEANO_FLAGS=mode=FAST_RUN,device=gpu0,cuda.root=/usr/local/cuda,floatX=float32,optimizer_including=cudnn python predict.py 

five params, five predict_fn, input:dicom,label  output:recall

"""
from __future__ import print_function

import sys
import os
import time
import random
#sys.path.insert(0,'/home/tuixiang1/anaconda2/lib/python2.7/site-packages')
#import pkg_resources
#pkg_resources.require("scipy==0.16.0")
sys.path.append('../lib')
sys.path.append('../model')
import numpy as np
#np.set_printoptions(threshold=np.inf)
import theano
import theano.tensor as T
import cPickle as pickle

import lasagne
import dicom
from sklearn import preprocessing
import scipy.misc as misc 

import scipy
print (scipy.__version__)
print (scipy.__file__)
import cv2

import xlrd
import xlwt
#from googlenet_model_las import build_model, iterate_minibatches
from vgg16model_las_sigmoid import build_model, iterate_minibatches
from generic_utils import get_acc

prob_dict={}
prob_list=[]
predict_fn_list=[]
params_path=[]

class_list=['guanggu_tijian']

params_path = ['/home/tuixiangbeijingtest0/Desktop/params_5k_5k/5k_5k_SavedModels_Epoch_9.7_5.4_18_dro_0.9_lr_0.00015_ts_0.2.params']

dicom_dir = '/media/disk_sda/srv/cr_test_set_2k'

xls_path = ['/media/disk_sda/srv/cr_test_set_2k/2k_sick_and_health_test.xls']


num_of_predict = 20
dicom2jpg = True

def get_img_info(dicom_dir,xls_path):
    ###########################   
    risno_list=[]   
    foldercol=[]
    foldercol1=[]
    
    label_list=[]
    
    count=0
    
    imgpath_list = []
    describle_list=[]
    impression_list=[]
    patient_loc_id_list=[]
    
    label_list = []
    patient_loc_id=[]
    describle=[]
    impression=[]
    
    for sh_count in xls_path:
        wb=xlrd.open_workbook(sh_count)
        sh=wb.sheet_by_index(0)
        
        risno_list+=sh.col_values(0)[1:]
        foldercol+=sh.col_values(5)[1:]
        foldercol1+=sh.col_values(4)[1:]
        label_list+=sh.col_values(6)[1:]
        patient_loc_id+=sh.col_values(1)[1:]
        describle+=sh.col_values(2)[1:]
        impression+=sh.col_values(3)[1:]

    print(len(imgpath_list),len(label_list),len(risno_list))
    
    for folderi in range(sh.nrows-1):
        imgpath = dicom_dir+foldercol1[folderi]+foldercol[folderi]
#        print('imgpath: ',imgpath)
        if os.path.isfile(imgpath):
            # print('imgpath: ',imgpath)
            imgpath_list.append(imgpath)
            describle_list.append(describle[folderi])
            impression_list.append(impression[folderi])    
            patient_loc_id_list.append(patient_loc_id[folderi])
            count += 1
    
    print(len(imgpath_list),len(label_list),len(risno_list))
    print ('total number of image in xls is: ',count)
    assert len(imgpath_list) == len(label_list) == len(risno_list), 'length of label and imgpath is not equel!'

    return risno_list,imgpath_list,label_list,patient_loc_id_list,describle_list,impression_list


def load_model(params_path, keyclass,dro=0.9):
    input_var = T.tensor4('input')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...   ",keyclass)
    network = build_model(input_var=input_var,dro=dro)
 
    print('start model loading...')
    with open(params_path, 'r') as f:
        data = pickle.load(f)
        lasagne.layers.set_all_param_values(network['fc8'], data)
        #lasagne.layers.set_all_param_values(network['loss3/classifier'], data)
    print('pretrained model loaded')

    start_time = time.time()
    networkout=network['prob']
    prediction = lasagne.layers.get_output(networkout)

    test_prediction = lasagne.layers.get_output(networkout, deterministic=True)

    predict_fn = theano.function([input_var], test_prediction)
    return predict_fn

def predict_one_dicom(filepath, predict_fn, newsize = [299,299], dicom2jpg = False, jpg_path = './jpg/'):
    temp_dir='./temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    filepath=filepath.replace(' ','\ ')
    prob = 0.0
    cmd='gdcmconv -w '+filepath+' '+temp_dir+'tmp_out.dicom' #gdcm2pnm
    os.system(cmd)
    

    ds = dicom.read_file(temp_dir+'tmp_out.dicom')
    pix = ds.pixel_array
    

    pix = misc.imresize(pix,newsize)

    pix_normalized = pix-np.mean(pix)
#    print (pix_normalized)
    pix_normalized = preprocessing.scale(pix_normalized)

    pix_normalized=pix_normalized.reshape(1,1,pix.shape[0],pix.shape[1])

    pix_normalized=np.repeat(pix_normalized,3,axis=1)
    pix_normalized=pix_normalized.astype('float32')
    
#    print (pix_normalized)
#    print (pix_normalized)
    prob=predict_fn(pix_normalized)
    color_type = [(0,0,255), (0,204,255), (0,255,0)]
#    color_type = [(0,0,255),(0,255,0)]
    if prob>=0.8:
        pos = 0      #red
    elif prob<=0.7 and prob>0.2:
        pos = 1      #yellow
    else:
        pos = 2      #green

    if dicom2jpg == True:
        pix = cv2.cvtColor(pix,cv2.COLOR_GRAY2RGB)
        cv2.rectangle(pix, (2,2), (220,220), color_type[pos], 8)
        cv2.imwrite(jpg_path,pix)
    os.system('rm -rf '+temp_dir)
    return prob

def predict(imgpath_list,label_list,patient_loc_id_list,describle_list,impression_list, params_path, class_list, num_of_predict = 33, newsize = [224,224], dicom2jpg = False):
    
    excelcache = xlwt.Workbook(encoding= 'utf-8')
    sheetcache = excelcache.add_sheet('sheet1')
    
    #T=[0.6,0.5,0.4,0.3,0.2] #threshold list
    #T=[0.2,0.18,0.16,0.14,0.1]
    #T=[0.02,0.04,0.06,0.08,0.09]
    T=[0.95]

    for iiter in range(len(T)):
        sheetcache.write(0,iiter+6,'RESULT:'+str(T[iiter]))
    sheetcache.write(0,0,'patient_loc_id')
    sheetcache.write(0,1,'describle'+str(T[iiter]))
    sheetcache.write(0,2,'impression'+str(T[iiter]))
    sheetcache.write(0,3,'path')
    sheetcache.write(0,4,'flag')
    sheetcache.write(0,5,'prob')
    
    outfilename='result.txt'
    outfile=open(outfilename,'w')
    ##########################################
    for p in range(len(params_path)):    
        predict_fn_list.append(load_model(params_path[p],class_list[p]))
        print(class_list[p],predict_fn_list[p])
        
    print('start predicting........')    
    ###########
    cnt = 0
    pred_result=[]
    label_result=[]

    for filei in range(len(imgpath_list)):
        filepath = imgpath_list[filei]
        label = label_list[filei]
        
        if os.path.isfile(filepath):
            if cnt >= num_of_predict:
                break
#            try:
            if 1==1:
                jpg_dir = './jpg/'
                if not os.path.exists(jpg_dir):
                    os.makedirs(jpg_dir)
                prob_dict[filepath]=[]
                for i in range(len(predict_fn_list)):
                    jpg_path = jpg_dir + str(cnt) + class_list[i] + '.jpg'
#                    print(filepath)
                    prob = predict_one_dicom(filepath, predict_fn_list[i], newsize, dicom2jpg, jpg_path)
                    prob_dict[filepath].append(prob[0][0])

                prob_pred=max(prob_dict[filepath])
                    
                print('start writing to rusult.txt......')
                outfile.write("%s %s:\n" % (str(cnt),filepath))
                print(imgpath_list[filei])
                
                for i in range(len(predict_fn_list)):
                    outfile.write("%s %s\n" % (class_list[i],prob_dict[filepath][i]))
                    print ('predict: ',class_list[i],prob_dict[filepath][i])
                outfile.write("final prob %s:\n" % (str(prob_pred)))
                print('final prob: ',prob_pred)
                
                #add write to excel
                for iiter in range(len(T)):
                    if prob_pred > T[iiter]:
                        prob_in = 1
                    else:
                        prob_in = 0
                    sheetcache.write(cnt+1, iiter+6, prob_in)
#                sheetcache.write(cnt+1, iiter+4, prob_pred)
#                print(type(prob_pred))
                prob_round = round(prob_pred, 7)
                sheetcache.write(cnt+1, 0, patient_loc_id_list[filei])
                sheetcache.write(cnt+1, 1, describle_list[filei])
                sheetcache.write(cnt+1, 2, impression_list[filei])
                sheetcache.write(cnt+1, 3, filepath)
                sheetcache.write(cnt+1, 4, label)
                sheetcache.write(cnt+1, 5, float(prob_round))
                cnt += 1
                print ('cnt: ',cnt)
                
                pred_result.append(prob_pred)
                label_result.append(label)
                               
#            except:
#                pass
    excelcache.save('./result.xls')
    outfile.close()
    print(pred_result)
    print(label_result)
    
    pred_result = np.asarray(pred_result)
    label_result = np.asarray(label_result)
    print (pred_result.shape)
    print (label_result.shape)

    
    for t in T:
        train_acc3 = get_acc(label_result, pred_result, t)
        print('t: ',t)
        print ('Accuracy       : ' + str(train_acc3[0]))
        print ('Precision      : ' + str(train_acc3[3]))
        print ('Positive Recall: ' + str(train_acc3[1]))
        print ('Negative Recall: ' + str(train_acc3[2]))
                
    
    return 0

if __name__ == '__main__':
    
    risno_list,imgpath_list,label_list,patient_loc_id_list,describle_list,impression_list = get_img_info(dicom_dir,xls_path)
    
    predict(imgpath_list,label_list,patient_loc_id_list,describle_list,impression_list,params_path,class_list, num_of_predict, dicom2jpg = dicom2jpg)

