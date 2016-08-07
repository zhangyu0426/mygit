# -*- coding: utf-8 -*-
"""
Created on Fri May  6 09:19:09 2016

@author: yangfan
"""
'''
功能 拷贝图片到本机 拷贝图片到本机并解压
使用方法
1.运行cmd sudo mount -t cifs //isilon.com/PACSS /mnt/PACSS -o username=tjh 
2.再运行 sudo python copyIMG.py 

参数 savepath                      本机保存图片的路径 一般在srv文件夹下面创建对应病种文件夹
     xlspath                       含有图片服务器路径的xls
     imgpath1+=table.col_values(5) 服务器路径分为两段 此为头部的xls索引
     imgpath2+=table.col_values(6) 服务器路径分为两段 此为身体的xls索引
     sta=0                         如果需要考取图片同时解压图片 设置sta=1
     table.nrows                   一次操作图片的个数 如果需要限制个数 修改此处 int类型

\\isilon.com\ris\usimage1  BAK  US_image
us_path = '\\isilon.com\ris\usimage1  BAK\'
     
返回值 执行每张图片的索引和错误报告
'''

import os
import os.path
import xlrd


#import gdcm


#运行代码前 运行cmd sudo mount -t cifs //isilon.com/PACSS /mnt/PACSS -o username=tjh 
#if us_image run: sudo mount -t cifs //isilon.com/ris/usimage1  BAK /mnt/US -o username=tjh 
#再运行 sudo python copyIMG.py  
#/home/tuixiangbeijingtest0/Desktop/workfolder/xls

sta=1
savepath='/media/disk_sda/srv/US/'
xlspath='/home/tuixiangbeijingtest0/Desktop/workfolder/xls/彩超-乳腺及淋巴结(双侧)_abnormal_3959.xls'
#savepath='/srv1/breastCR/'
#cmdstr='sudo mount -t cifs //isilon.com/PACSS /mnt/PACSS -o username=tjh'
#path1000='/home/tuixiangbeijingtest0/Desktop/workfolder/text1/img1000'

imgpath1=[]
imgpath2=[]
us_path = r'usimage1  BAK'  #us图片的路径 usimage1\ \ BAK
local_path = '/mnt/US/' #挂载到本地的路径
us_path_2 = '/'
dir_list = []
path_cnt = 0

data=xlrd.open_workbook(xlspath)
table=data.sheet_by_index(0)
imgpath1+=table.col_values(0)
imgpath2+=table.col_values(1)
print(len(imgpath1))
for index in range(1,table.nrows):
    if len(imgpath1[index])==0 or len(imgpath2[index])==0:
        continue
    str1=imgpath1[index]+us_path_2+imgpath2[index]
    str1=str1.replace(' ','\ ')
    str1 = local_path+us_path+us_path_2+str1
    #print(str1)
    
    if os.path.exists(str1):
        path_cnt = path_cnt+1
        dir_list = os.listdir(str1)
        str1 = str1.replace(r'usimage1  BAK','usimage1\ \ BAK')
        #print(dir_list[0])
        cmdstr='cp --parents '+str1+us_path_2+dir_list[0]+' '+savepath
        #print(cmdstr)
        
        #print(cmdstr)
        #print(cmdstr1)
        print(index) 
        if sta==1:
            cmdstr1='dcmdjpeg '+savepath+str1+us_path_2+dir_list[0]+' '+savepath+str1+us_path_2+dir_list[0]+'hh' #gdcm2pnm
            os.system(cmdstr1)
            os.system(cmdstr)

print(path_cnt)            
            
    








'''
path='/home/tuixiangbeijingtest0/Desktop/workfolder/text1/img/'
savepath='/home/tuixiangbeijingtest0/Desktop/workfolder/text1/img3/'
print(os.listdir(path))
imglist=os.listdir(path)
print(len(imglist))
for item in imglist:
    a=path+item
    b=savepath+item
    c='gdcmconv -raw '+a+' '+b
    #print(c)
    os.system(c)






xlspath='/home/tuixiangbeijingtest0/Desktop/workfolder/xls/1W_data.xls'
savepath='/home/tuixiangbeijingtest0/Desktop/workfolder/text1/'
head='smb:'
tstr='gdcmconv —raw '
count=0

data=xlrd.open_workbook(xlspath)
table=data.sheet_by_index(0)
nrows=table.nrows
for i in range(1,nrows):
    imgpathhead=table.cell(i,5).value
    #print(imgpath)
    imgpath=table.cell(i,6).value
    #print(imgpathhead)
    ipath=head+str(imgpathhead)+str(imgpath)
    #print(ipath)
    tstr1=tstr+str(ipath)+' '+savepath+str(imgpath)
    #print(tstr1)
    dirpath=os.path.split(imgpath)
    dirpath1=savepath+dirpath[0]+'/'
    print(tstr1)
    #os.system('gdcmconv —raw /home/tuixiangbeijingtest0/Desktop/compress-test/hw1 /home/tuixiangbeijingtest0/Desktop/compress-test/hwtest[%i]')
            
    if os.path.exists(dirpath1):
        #os.system(tstr1)
        shutil.copyfile(str(ipath),savepath+str(imgpath))
        count+=1
    else:
        os.makedirs(dirpath1)
        #os.system(tstr1)
        shutil.copyfile(str(ipath),savepath+str(imgpath))
        count+=1
    print(count)
    if count>10:
        break
 '''
            
                
            
            
            
            
        

