# -*- coding: utf-8 -*-
import sys
import os
import os.path
import xlrd
import csv
import dicom
#import gdcm



#运行代码前 把文件路径挂载到本机，cmd sudo mount -t cifs //isilon.com/PACSS /mnt/PACSS -o username=tjh 

sta=1
sick_location = 'HEAD'
ct_thick = 5
savepath='/media/disk_sda/srv/CT_naogeng_chuxue_2k'
xlspath=['./chuxue_path.csv'
        ,'./naogeng_path.csv']

def ct_head_data_clean(sta=sta,savepath=savepath,xlspath=xlspath,ct_thick=ct_thick,sick_location=sick_location):
    imgpath1=[]
    imgpath2=[]
    imgpath3=[]
    thick5 = 0
    thick6 = 0
    thick7 = 0
    img_cnt = 0
    for path in xlspath:
        reader = csv.reader(open(path,"rU"))    
        imgpath1 = [row[1] for row in reader]
        reader = csv.reader(open(path,"rU"))    
        imgpath2 = [row[2] for row in reader]
        reader = csv.reader(open(path,"rU"))    
        imgpath3 = [row[0] for row in reader]

        patient_local_id_set = set(imgpath3)
        patient_local_id_set = list(patient_local_id_set)

        for p_id in range(len(patient_local_id_set)):
            patient_local_id = patient_local_id_set[p_id]
            for index in range(1,len(imgpath1)):        
                if len(imgpath1[index])==0 or len(imgpath2[index])==0:
                    continue
                if imgpath3[index] == patient_local_id:
                    filepath=imgpath1[index]+imgpath2[index]
                    str_copy=filepath.replace(' ','\ ')
                    cmdstr='cp --parents '+str_copy+' '+savepath
                    cmd_rm = 'rm '+savepath+str_copy
                    local_path = savepath+str_copy  #下载以后本机路径
                    os.system(cmdstr)#下载到本地。
                    try:
                        graph = dicom.read_file(local_path)
                        thick = graph.SliceThickness
                        location = graph.StudyDescription
                        studyid = graph.StudyID
                        if float(thick) >= ct_thick and sick_location in location:
                            series_number = graph.SeriesNumber
                            print(series_number)
                            print('studyid:'+studyid)
                            print(thick)
                            print(location)
                            print('patient_local_id:'+patient_local_id)
                            print(p_id)
                            print('filepath:'+filepath)
                            if float(thick) == 5:  #统计各个毫米数的分布，无其他意义。
                                thick5 = thick5+1
                            if float(thick) == 6:
                                thick6 = thick6+1
                            if float(thick) ==7.5:
                                thick7 = thick7+1
                            break#确定这个病人的id，系列号后，退出循环，下一个循环是下载这个人的这个系列的图。
                        else:
                            os.system(cmd_rm) #不符合要求，删除这张图，寻找下一张可以确定病人信息的图

                    except:
                        continue

            for index in range(1,len(imgpath1)):
                if len(imgpath1[index])==0 or len(imgpath2[index])==0:
                    continue
                if imgpath3[index] == patient_local_id:
                    filepath=imgpath1[index]+imgpath2[index]
                    str_copy=filepath.replace(' ','\ ')
                    cmdstr='cp --parents '+str_copy+' '+savepath
                    cmdstr1='gdcmconv -w '+savepath+str_copy+' '+savepath+str_copy+'hh' #gdcm2pnm
                    cmd_rm = 'rm '+savepath+str_copy
                    local_path = savepath+str_copy  #下载以后本机路径
                    os.system(cmdstr)#下载到本地。
                    try:#读取下载后的dicom信息
                        graph = dicom.read_file(local_path)
                        series = graph.SeriesNumber
                        StudyDescription = graph.StudyDescription
                        studyid_img = graph.StudyID
                        if series==series_number and sick_location in StudyDescription and  studyid_img == studyid:
                            #判断是否是同一套，是则解压后删除源文件，否则直接删除文件
                            print('this is same series dicom image.')
                            img_cnt+=1
                            print(img_cnt) 
                            os.system(cmdstr1)  
                            os.system(cmd_rm)  
                        else:
                            print('useless img , delete it !')
                            os.system(cmd_rm)

                    except:
                        print('Error! this photo cannot read dicom info:' +str(filepath))
                        continue


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("you need modify savepath,xlspath")
    else:
        kwargs = {}
        ct_head_data_clean(**kwargs)


