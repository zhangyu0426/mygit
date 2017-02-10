# -*- coding: utf-8 -*-

import os
import xlrd
import xlwt

dirs = '/Users/zhangyu/Desktop/detection_label_1202/detection_result/pos'
excelpath = '/Users/zhangyu/Desktop/detection_label_1202/'
val_excel = 'pos_100_label'

countUcompress = 0
filename_col = []
for direct,folders,filenames in os.walk(dirs):
    for filename in filenames:
        cmdstr = os.path.join(direct, filename)
        cmdstr = os.path.join(direct, filename)
        print(filename.replace('.jpg',''))
        filename_col.append(filename.replace('.jpg',''))
        countUcompress +=1
print('the filename is:', countUcompress)
print(filename_col)



excel_item = xlwt.Workbook(encoding='utf-8')
sheet1=excel_item.add_sheet('Sheet1')
sheet1.write(0,0,'PATIENT_ID')
sheet1.write(0,1,'LABEL_SITUATION')


colcount = 0
for content in [filename_col]:
    xlcount = 1
    for item in content:      
        sheet1.write(xlcount,colcount,item)
        xlcount+=1   
    colcount += 1
excel_item.save(excelpath+val_excel+'.xls') 