#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')


import os
import os.path
import xlrd
import csv

#import gdcm


#运行代码前 运行cmd sudo mount -t cifs //isilon.com/PACSS /mnt/PACSS -o username=tjh 

sta=1
img_path = '/mnt/10.35.51.13/wuhan_x_ray/PACSS/mnt/PACSS'
savepath='/mnt/10.35.51.13/wuhan_x_ray/data_origin/tj_zhuyuan_pos_test'
ypathlist=['/home/tuixiangbeijingtest0/workspace/xls/detection_test_data_path.csv']


PATIENT_LOCAL_IDcol = []
UNC_PATHcol=[]
FILENAMEcol = []
for path in ypathlist:
    reader = csv.reader(open(path,"rU"))    
    PATIENT_LOCAL_IDcol = [row[0] for row in reader]
    reader = csv.reader(open(path,"rU"))    
    UNC_PATHcol =         [row[1] for row in reader]
    reader = csv.reader(open(path,"rU"))    
    FILENAMEcol =         [row[2] for row in reader]

    print(len(PATIENT_LOCAL_IDcol))
    for index in range(1,len(PATIENT_LOCAL_IDcol)):
        
        if len(UNC_PATHcol[index])==0 or len(FILENAMEcol[index])==0:
            continue

        split_path =  FILENAMEcol[index]
        split_list =  split_path.split("/")[:len(split_path.split("/"))-1]
        connect_str = '/'
        newFILENAMEcol = connect_str.join(split_list)
        image_dir = UNC_PATHcol[index]+newFILENAMEcol
        jpg_path = os.path.join(image_dir, (PATIENT_LOCAL_IDcol[index] + '.jpg'))
        #print(jpg_path)

        str1=jpg_path.replace(' ','\ ')
        print(str1)  
        cmdstr='cp  '+str1+' '+savepath
       
        print(cmdstr)
        print(index) 
        if sta==1:
            os.system(cmdstr)
                
       
        



            
            
            
        






            
            
            
        

