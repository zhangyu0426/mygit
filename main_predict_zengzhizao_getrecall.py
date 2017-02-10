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
###############################################
prob_dict={}
prob_list=[]
predict_fn_list=[]
params_path=[]

params_dir = '/srv2/params/googlenet_five_class_preprocessing/'
#class_list=['lgjd','xyzd','zdmjtc','xmzh','fgr']
class_list=['zengzhizao_health']
#params_path = [params_dir+'lgjd_pre_googlenet_SaveModels.params',params_dir+'xyzd_pre_googlenet_SaveModels.params',params_dir+'zdmjtc_pre_googlenet_SaveModels.params',params_dir+'xmzh_pre_googlenet_SaveModels.params',params_dir+'fgr_pre_googlenet_SaveModels.params']
params_path = ['/home/tuixiangbeijingtest0/workspace/zcw/workspace/Predict_dicom/params/1Wzengzhizao_90_58_SavedModels_Epoch_320_dro_0.95_lr_5e-05_ts_0.2.params']
#params_path = ['/srv1/wuhan_params/1W_p98_n46.params']
for p in params_path:
    print (p)

##################
#dicom_dir = '/home/tuixiang1/workspace/code_whf/find_health.git/prediction-dicom.git/test_select'
dicom_dir = '/media/disk_sda/srv/zengzhizao_health'
#xls_path = ['/srv1/lung_heart/5w_test/cr_all_health_3w_test.xls']
xls_path = ['/home/tuixiangbeijingtest0/workspace/xls/127all_zengzhizao.xls']


num_of_predict = 30000
dicom2jpg = True


def get_img_info(dicom_dir,xls_path):
    ###########################   
    risno_list=[]   
    foldercol=[]
    foldercol1=[]
    
    label_list=[]
    
    count=0
    
    imgpath_list = []
    label_list = []
    
    for sh_count in xls_path:
        wb=xlrd.open_workbook(sh_count)
        sh=wb.sheet_by_index(0)
        
        risno_list+=sh.col_values(0)[1:]
        foldercol+=sh.col_values(5)[1:]
        foldercol1+=sh.col_values(4)[1:]
        label_list+=sh.col_values(6)[1:]

    print(len(imgpath_list),len(label_list),len(risno_list))
    
    for folderi in range(sh.nrows-1):
        imgpath = dicom_dir+foldercol1[folderi]+foldercol[folderi]
        if os.path.isfile(imgpath):
#            print('imgpath: ',imgpath)
            imgpath_list.append(imgpath)            
            count += 1
    
    print(len(imgpath_list),len(label_list),len(risno_list))
    print ('total number of image in xls is: ',count)
    assert len(imgpath_list) == len(label_list) == len(risno_list), 'length of label and imgpath is not equel!'

    return risno_list,imgpath_list,label_list


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
    elif prob<=0.7 and prob>0.3:
        pos = 1      #yellow
    else:
        pos = 2      #green

    if dicom2jpg == True:
        pix = cv2.cvtColor(pix,cv2.COLOR_GRAY2RGB)
        cv2.rectangle(pix, (2,2), (220,220), color_type[pos], 8)
        cv2.imwrite(jpg_path,pix)
    os.system('rm -rf '+temp_dir)
    return prob

def predict(imgpath_list,label_list, params_path, class_list, num_of_predict = 3000, newsize = [224,224], dicom2jpg = False):

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
            try:
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
                
                cnt += 1
                print ('cnt: ',cnt)
                
                pred_result.append(prob_pred)
                label_result.append(label)
                               
            except:
                pass
            
    outfile.close()
    print(pred_result)
    print(label_result)
    
    pred_result = np.asarray(pred_result)
    label_result = np.asarray(label_result)
    print (pred_result.shape)
    print (label_result.shape)

    T=[0.7,0.6,0.5,0.4,0.3,0.2]
    for t in T:
        train_acc3 = get_acc(label_result, pred_result, t)
        print('t: ',t)
        print ('Accuracy       : ' + str(train_acc3[0]))
        print ('Precision      : ' + str(train_acc3[3]))
        print ('Positive Recall: ' + str(train_acc3[1]))
        print ('Negative Recall: ' + str(train_acc3[2]))
                
    
    return 0

if __name__ == '__main__':
    
    risno_list,imgpath_list,label_list = get_img_info(dicom_dir,xls_path)
    
    predict(imgpath_list,label_list, params_path,class_list, num_of_predict, dicom2jpg = dicom2jpg)

