# -*- coding: utf-8 -*-


'''
统计一个人不止一套1.25mmCT的情况。经过测试，2040：2096人中，有25人有不止一套1.25mmCT
'''

import os
import os.path
import xlrd
import csv
import dicom
#import gdcm


#运行代码前 运行cmd sudo mount -t cifs //isilon.com/PACSS /mnt/PACSS -o username=tjh 

sta=1
savepath='/media/disk_sda/srv/CT2000_1012/'
xlspath=['./2000_health_path_80w.csv'
        ,'./2000_sick_path_80w.csv']


imgpath=[]
imgpath0=[]
imgpath1=[]
imgpath2=[]
patient_multi_series = []

for path in xlspath:
    reader = csv.reader(open(path,"rU"))    
    imgpath = [row[0] for row in reader]

    reader = csv.reader(open(path,"rU"))    
    imgpath1 = [row[1] for row in reader]
    reader = csv.reader(open(path,"rU"))    
    imgpath2 = [row[2] for row in reader]


    imgpath_multi = imgpath[1:]
    imgpath0 = set(imgpath_multi)
    imgpath0 = list(imgpath0)
    print(len(imgpath0))
    for pid in range(len(imgpath0)):
        print(pid)
        patient_local_id = imgpath0[pid]
        x = []
        for img in range(1,len(imgpath1)):
            if len(imgpath1[img])==0 or len(imgpath2[img])==0:
                continue
            if imgpath[img] == patient_local_id:
                try:
                    str1=imgpath1[img]+imgpath2[img]
                    graph = dicom.read_file(str1)
                    thick = graph.SliceThickness
                    if float(thick) ==1.25 :#or float(thick) == 1:
                        series = graph.SeriesNumber
                        x.append(series)
                        #print(img) 
                except:
                    #print('cannot read :'+str(str1))
                    continue
        y = set(x)
        if len(y)>1:
            patient_multi_series.append(patient_local_id)

print('the whole test set is :'+str(len(imgpath0)))
print('the people have 2 series at least:'+str(len(patient_multi_series)))
print('id of this people:',patient_multi_series)











