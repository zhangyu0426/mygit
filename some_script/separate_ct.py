# -*- coding: utf-8 -*-
import sys
import os
import os.path
import xlrd
import csv
#import gdcm



sta=1
savepath='/media/disk_sda/srv/tupuan_lujing'
xlspath='./lung_cancer_pathology_path.csv'

excelpath = ['./1490_cancer_4.xls'
            ,'./1490_cancer_5.xls'
            ,'./1490_cancer_6.xls']

PATIENT_LOCAL_ID=[]
imgpath1=[]
imgpath2=[]

PATIENT_LOCAL_IDcol_1=[]
PATIENT_LOCAL_IDcol_2=[]
PATIENT_LOCAL_IDcol_3=[]


reader = csv.reader(open(xlspath,"rU"))    
PATIENT_LOCAL_ID = [row[0] for row in reader]
reader = csv.reader(open(xlspath,"rU"))    
imgpath1 = [row[1] for row in reader]
reader = csv.reader(open(xlspath,"rU"))    
imgpath2 = [row[2] for row in reader]


wb=xlrd.open_workbook(excelpath[0])
sh=wb.sheet_by_index(0)
PATIENT_LOCAL_IDcol_1+=sh.col_values(1)[1:]

wb=xlrd.open_workbook(excelpath[1])
sh=wb.sheet_by_index(0)
PATIENT_LOCAL_IDcol_2+=sh.col_values(1)[1:]

wb=xlrd.open_workbook(excelpath[2])
sh=wb.sheet_by_index(0)
PATIENT_LOCAL_IDcol_3+=sh.col_values(1)[1:]




for index in range(1,len(imgpath1)):
    if len(imgpath1[index])==0 or len(imgpath2[index])==0:
        continue
    filepath=imgpath1[index]+imgpath2[index]
    filepath = filepath.replace(' ','\ ')
    if PATIENT_LOCAL_ID[index] in PATIENT_LOCAL_IDcol_1:
        direct = '/media/disk_sda/srv/ct_biaoji_1490/4/'+PATIENT_LOCAL_ID[index]
        mkdir = 'mkdir /media/disk_sda/srv/ct_biaoji_1490/4/'+PATIENT_LOCAL_ID[index]
        if not os.path.isdir(direct):
            os.system(mkdir)
        cmdstr = 'cp '+savepath+filepath+' '+direct
        os.system(cmdstr)
        print('copy_to_file_4')

    if PATIENT_LOCAL_ID[index] in PATIENT_LOCAL_IDcol_2:
        direct = '/media/disk_sda/srv/ct_biaoji_1490/5/'+PATIENT_LOCAL_ID[index]
        mkdir = 'mkdir /media/disk_sda/srv/ct_biaoji_1490/5/'+PATIENT_LOCAL_ID[index]
        if not os.path.isdir(direct):
            os.system(mkdir)
        cmdstr = 'cp '+savepath+filepath+' '+direct
        os.system(cmdstr)
        print('copy_to_file_5')

    if PATIENT_LOCAL_ID[index] in PATIENT_LOCAL_IDcol_3:
        direct = '/media/disk_sda/srv/ct_biaoji_1490/6/'+PATIENT_LOCAL_ID[index]
        mkdir = 'mkdir /media/disk_sda/srv/ct_biaoji_1490/6/'+PATIENT_LOCAL_ID[index]
        if not os.path.isdir(direct):
            os.system(mkdir)
        cmdstr = 'cp '+savepath+filepath+' '+direct
        os.system(cmdstr)
        print('copy_to_file_6')






