# -*- coding: utf-8 -*-

'''
功能 拷贝图片到本机 拷贝图片到本机并解压
使用方法
1.挂载服务器到本机：sudo mount -t cifs //isilon.com/ris/  /mnt/US -o username=tjh 
2.再运行 sudo python copyIMG.py 

需要注意的是
1，这个地址里面有空格，os.exists的时候和cmd执行的时候要匹配两种格式，中间历经两次格式转换。



'''

import os
import os.path
import xlrd
import xlwt


sta=1
savepath='/media/disk_sda/srv/US'
xlspath='/home/tuixiangbeijingtest0/Desktop/workfolder/xls/彩超-乳腺及淋巴结(双侧)_abnormal_3959.xls'

imgpath1=[]
imgpath2=[]
imgpath3=[]
imgpath4=[]
imgpath5=[]

out_imgpath1=[]
out_imgpath2=[]
out_imgpath3=[]
out_imgpath4=[]
out_imgpath5=[]
out_imgpath6=[]
out_imgpath7=[]

us_path = r'usimage1  BAK'  #us图片的路径 usimage1\ \ BAK
local_path = '/mnt/US/' #挂载到本地的路径
us_path_2 = '/'
dir_list = []
path_cnt = 0

data=xlrd.open_workbook(xlspath)
table=data.sheet_by_index(0)
imgpath1+=table.col_values(0)
imgpath2+=table.col_values(1)
imgpath3+=table.col_values(2)
imgpath4+=table.col_values(3)
imgpath5+=table.col_values(4)
print(len(imgpath1))
for index in range(1,100):#table.nrows):
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

        print(index) 
        if sta==1:
            cmdstr1='dcmdjpeg '+savepath+str1+us_path_2+dir_list[0]+' '+savepath+str1+us_path_2+dir_list[0]+'hh' #gdcm2pnm
            os.system(cmdstr1)
            os.system(cmdstr)
            if os.path.isfile(savepath+str1+us_path_2+dir_list[0]+'hh'): #如果成功解压，就append到list中去，最后输出病人报告和路径的excel
                out_imgpath1.append(imgpath1[index])
                out_imgpath2.append(imgpath2[index])
                out_imgpath3.append(imgpath3[index])
                out_imgpath4.append(imgpath4[index])
                out_imgpath5.append(imgpath5[index])
                out_imgpath6.append(dir_list[0])  
                out_imgpath7.append(1)                  


wb_output = xlwt.Workbook(encoding='utf-8')
sheet_output = wb_output.add_sheet('US_Breast') 
sheet_output.write(0,0,'LASTSAVETIME')   
sheet_output.write(0,1,'PATIENT_ID')
sheet_output.write(0,2,'EXAM_ITEMSSTR')
sheet_output.write(0,3,'DESCRIPTION')
sheet_output.write(0,4,'IMPRESSION')
sheet_output.write(0,5,'filename')
sheet_output.write(0,6,'tag')

colcount = 0
for content in [out_imgpath1,out_imgpath2,out_imgpath3,out_imgpath4,out_imgpath5,out_imgpath6,out_imgpath7]:
    xlcount = 1
    for item in content:
        sheet_output.write(xlcount,colcount,item)
        xlcount = xlcount+1
    colcount = colcount+1
wb_output.save('/home/tuixiangbeijingtest0/Desktop/workfolder/xls/彩超-乳腺及淋巴结(双侧)_abnormal_0808.xls')


print(path_cnt)            
            
    







            
            
            
        

