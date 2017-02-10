# -*- coding: utf-8 -*-
import sys
import os
import os.path
import xlrd
import csv
import dicom
#import gdcm

#运行前 把文件路径挂载到本机，cmd sudo mount -t cifs //isilon.com/PACSS /mnt/PACSS -o username=tjh 

sta=1
ct_thick = 1.25
savepath='/media/disk_sda/srv/dicom_dir'
xlspath=['/home/tuixiangbeijingtest0/workspace/xls/CT_path_after_may_cancer_pathology_category.csv']
#        ,'./sick.csv']

def ct_data_clean(sta=sta,savepath=savepath,xlspath=xlspath,ct_thick=ct_thick):
    imgpath1=[]
    imgpath2=[]
    imgpath3=[]
    img_cnt = 0
    for path in xlspath:
        reader = csv.reader(open(path,"rU"))    
        imgpath1 = [row[1] for row in reader]
        reader = csv.reader(open(path,"rU"))    
        imgpath2 = [row[2] for row in reader]
        reader = csv.reader(open(path,"rU"))    
        imgpath3 = [row[0] for row in reader]

        for index in range(1,len(imgpath1)):
            if len(imgpath1[index])==0 or len(imgpath2[index])==0:
                continue
            filepath=imgpath1[index]+imgpath2[index]
            str_copy=filepath.replace(' ','\ ')
            cmdstr='cp --parents '+str_copy+' '+savepath
            cmdstr1='gdcmconv -w '+savepath+str_copy+' '+savepath+str_copy+'hh' #gdcm2pnm
            cmd_rm = 'rm '+savepath+str_copy
            local_path = savepath+filepath  #下载以后本机路径,
            os.system(cmdstr)#下载到本地。
            try:
                graph = dicom.read_file(local_path)
                thick = graph.SliceThickness
                if float(thick) == ct_thick:
                    print('this is 1.25 dicom image.')
                    img_cnt+=1
                    print(img_cnt) 
                    if sta==1:
                        cmdstr1='gdcmconv -w '+savepath+str_copy+' '+savepath+str_copy+'hh' #gdcm2pnm
                        cmd_rm = 'rm '+savepath+str_copy
                        os.system(cmdstr1)  
                        print(cmdstr1)
                        #os.system(cmd_rm)  
                else:
                    print('not 1.25 img.delete')
                    os.system(cmd_rm) 

            except:
                print('this photo cannot read dicom info:' +str(filepath))
                continue


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("you need modify savepath,xlspath")
    else:
        kwargs = {}
        ct_data_clean(**kwargs)


