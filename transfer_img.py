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
     
返回值 执行每张图片的索引和错误报告
'''

import os
import os.path
import xlrd


#import gdcm


#运行代码前 运行cmd sudo mount -t cifs //isilon.com/PACSS /mnt/PACSS -o username=tjh 
#再运行 sudo python copyIMG.py 

sta=1
savepath='/media/tuixiangbeijingtest0/f989fbc9-24a9-42ce-802e-902125737abd/srv/ruxian2'
xlspath='/home/tuixiangbeijingtest0/Desktop/workfolder/xls/RX_DATA_rx_all_img.xls'
#savepath='/srv1/breastCR/'
#cmdstr='sudo mount -t cifs //isilon.com/PACSS /mnt/PACSS -o username=tjh'
#path1000='/home/tuixiangbeijingtest0/Desktop/workfolder/text1/img1000'

imgpath1=[]
imgpath2=[]
data=xlrd.open_workbook(xlspath)
table=data.sheet_by_index(0)
imgpath1+=table.col_values(5)
imgpath2+=table.col_values(6)
print(len(imgpath1))
for index in range(1,table.nrows):
    if len(imgpath1[index])==0 or len(imgpath2[index])==0:
        continue
    str1=imgpath1[index]+imgpath2[index]
    str1=str1.replace(' ','\ ')
    #print(str1)
    cmdstr='cp --parents '+str1+' '+savepath
   
    #print(cmdstr)
    #print(cmdstr1)
    print(index) 
    if sta==1:
        cmdstr1='gdcmconv -w '+savepath+str1+' '+savepath+str1+'hh' #gdcm2pnm
        os.system(cmdstr1)
    os.system(cmdstr)
   
    








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
            
                
            
            
            
            
        

