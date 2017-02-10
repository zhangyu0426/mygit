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

savepath='/media/disk_sda/srv/CT2000_1012/'
xlspath=['./2000_health_path_80w.csv'
        ,'./2000_sick_path_80w.csv']


delete_id = ['1515084015','1515089670','1514010322'
            ,'1514044612','1514012768','1514049088'
            ,'1514070177','1515085075','1516020323'
            ,'1513032055','1515004688','1515070553'
            ,'1515015019','1516021681','1515038096'
            ,'1514057002','1514033365','1514071088'
            ,'1514064075','1513027874','1514012773'
            ,'1515002950','1514004331','1514028033'
            ,'1513028609']

result_id = []
id_path=[]
imgpath1=[]
imgpath2=[]

for path in xlspath:
    reader = csv.reader(open(path,"rU"))    
    id_path = [row[0] for row in reader]

    reader = csv.reader(open(path,"rU"))    
    imgpath1 = [row[1] for row in reader]
    reader = csv.reader(open(path,"rU"))    
    imgpath2 = [row[2] for row in reader]



    print(len(delete_id))
    for pid in range(len(delete_id)):
        print(pid)
        patient_local_id = delete_id[pid]
        print(patient_local_id)
        for img in range(1,len(imgpath1)):
            if len(imgpath1[img])==0 or len(imgpath2[img])==0:
                continue
            if id_path[img] == patient_local_id:
                result_id.append(id_path[img])
                str1=imgpath1[img]+imgpath2[img]
                str1=str1.replace(' ','\ ')
                str1='/media/disk_sda/srv/CT2000_1012'+str1
                str1 = str1[0:str1.rfind('/')]
                #print(str1)
                os.system('rm -r '+str1)

result_id = set(result_id)
result_id = list(result_id)
print('delete id is :',result_id)









