#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import,print_function
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cv2
import os
import time
import Image
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc as misc
import scipy 
print (scipy.__version__)
#import matplotlib.pyplot as plt
import time
#from sklearn.datasets import make_classification
#from sklearn.cross_validation import train_test_split
import xlrd
import csv
import string
import glob
import dicom
import scipy.io as sio
import scipy.misc as misc
import random
import xlwt
random.seed(1337)

#sudo mount -t cifs //isilon.com/PACSS /mnt/PACSS -o username=tjh 
#需要调整的几个参数分别是xls文件的路径，excel文件名，下载图片保存位置，生成的验证集合excel的文件名，numpy保存位置，numpy保存名称
excelpath = '/home/tuixiangbeijingtest0/workspace/xls/'
excelname = ['2012_path.csv'
            ,'2013_path.csv'
            ,'2014_path.csv'
            ,'2015_path.csv'
            ,'2016_path.csv']  
savepath  = '/mnt/backup/PACSS'
val_excel = 'cr_backup_decompress_failed'

#image_dir = '/mnt/backup/PACSS/jpg/2016'


def back_up(newsize=[299,299],tvt=5,excelpath=excelpath,excelname=excelname):
    print('start.................................')
    #读取excel中的数据
    ypathlist=[]
    for i in range(len(excelname)):
        ypathlist.append(excelpath+excelname[i])

    PATIENT_LOCAL_IDcol = []
    UNC_PATHcol=[]
    FILENAMEcol = []

    #解压缩失败的保存到excel
    list_PATIENT_LOCAL_IDcol = []
    list_UNC_PATHcol = []
    list_FILENAMEcol = [] 

    decompress_error = 0
    error=0
    cnt = 0
    for path in ypathlist:
        reader = csv.reader(open(path,"rU"))    
        PATIENT_LOCAL_IDcol = [row[0] for row in reader]
        reader = csv.reader(open(path,"rU"))    
        UNC_PATHcol = [row[1] for row in reader]
        reader = csv.reader(open(path,"rU"))    
        FILENAMEcol = [row[2] for row in reader]

        #print(len(PATIENT_LOCAL_IDcol))
        print(path)
        time_start = time.time()
        randomlist=range(len(PATIENT_LOCAL_IDcol))
        #randomlist=range(5)
        random.shuffle(randomlist)
        for folderi in randomlist:
            folder=FILENAMEcol[folderi]
            if len(folder)==0:
                pass
            else:            
                filepath=UNC_PATHcol[folderi]+FILENAMEcol[folderi]
                filepath=filepath.replace(' ','\ ')
                copy_str = 'cp --parents '+filepath+' '+savepath
                os.system(copy_str)    #copy img to savepath
                decompress_str='gdcmconv -w '+savepath+filepath+'  '+savepath+filepath+'hh' #gdcm2pnm
                os.system(decompress_str)
                origin_path = savepath+UNC_PATHcol[folderi]+FILENAMEcol[folderi]
                newpath=savepath+UNC_PATHcol[folderi]+FILENAMEcol[folderi]+'hh'  #解压缩以后的文件
                #判断解压缩文件是否存在
                if os.path.exists(newpath):
                    print('decompress_file_exist,make_jpg...')
                    origin_size = os.path.getsize(origin_path)
                    new_size = os.path.getsize(newpath)
                    print('new_size:'+str(new_size))
                    print('origin_size:'+str(origin_size))
                    if new_size >= origin_size:
                        #生成JPG的保存路径
                        split_path =  FILENAMEcol[folderi]
                        split_list =  split_path.split("/")[:len(split_path.split("/"))-1]
                        connect_str = '/'
                        newFILENAMEcol = connect_str.join(split_list)
                        image_dir = savepath+UNC_PATHcol[folderi]+newFILENAMEcol
                        print(image_dir)
                        #生成JPG并且保存
                        try:
                            ds = dicom.read_file(newpath)
                            pix = ds.pixel_array
                            print(pix.shape)
                            patient_id = ds.PatientID
                            bodypart,viewposition=ds.BodyPartExamined,ds.ViewPosition
                            monochrome = ds.PhotometricInterpretation
                            if bodypart=='CHEST' and viewposition=='PA':
                                center = ds.WindowCenter
                                width = ds.WindowWidth
                                if isinstance(center,list):
                                    center=center[0]
                                if isinstance(width,list):
                                    width=width[0]
                                low=center-width/2
                                hig=center+width/2
                                pix_out=np.zeros(pix.shape)
                                w1=np.where(pix>low) and np.where(pix<hig)
                                pix_out[w1]=((pix[w1]-center+0.5)/(width-1)+0.5)*255
                                pix_out[np.where(pix<=low)]=pix[np.where(pix<=low)]=0
                                pix_out[np.where(pix>=hig)]=pix[np.where(pix>=hig)]=255
                                pix_out = misc.imresize(pix_out,[pix_out.shape[0], pix_out.shape[1]])
                                if monochrome == 'MONOCHROME1':
                                    pix_out = 255 - pix_out
                                jpg_path = os.path.join(image_dir, (PATIENT_LOCAL_IDcol[folderi] + '.jpg'))#使用影像号作为文件名（武汉）
                                pix_out = cv2.cvtColor(pix_out,cv2.COLOR_GRAY2RGB)
                                cv2.imwrite(jpg_path,pix_out)

                                #计算成功解压缩且保存成JPG的个数
                                cnt+=1
                                print(str(cnt)+' jpg images succeed.')
                                rm_str = 'rm '+savepath+filepath
                                os.system(rm_str)   #解压缩成功，生成jpg成功，则删除原始dicom文件
                        except:
                            #未能保存成JPG的个数
                            error+=1
                            print(str(error)+' images read_dicom_error.')
                            continue
                    else:
                        decompress_error+=1
                        #新文件小于原始dicom，说明解压缩生成的是0字节的文件，保存到excel
                        list_PATIENT_LOCAL_IDcol.append(PATIENT_LOCAL_IDcol[folderi])
                        list_UNC_PATHcol.append(UNC_PATHcol[folderi]) 
                        list_FILENAMEcol.append(FILENAMEcol[folderi])

    decompress_failed = xlwt.Workbook(encoding='utf-8')
    sheet1=decompress_failed.add_sheet('Sheet1')
    sheet1.write(0,0,'PATIENT_LOCAL_ID')
    sheet1.write(0,1,'UNC_PATH')
    sheet1.write(0,2,'FILENAME')
    colcount = 0
    for content in [list_PATIENT_LOCAL_IDcol, list_UNC_PATHcol, list_FILENAMEcol]:
        xlcount = 1
        for item in content:      
            sheet1.write(xlcount,colcount,item)
            xlcount+=1   
        colcount += 1
    decompress_failed.save(excelpath+val_excel+'_'+str(decompress_error)+'_'+'backup'+'.xls') 
        
    time_end = time.time()
    print('runing time is :'+str(time_end-time_start))
    print('cannot_read_dicom_info: '+str(error))
    print('0_file: '+str(decompress_error))



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
        back_up(**kwargs)














