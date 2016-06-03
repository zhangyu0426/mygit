# -*- coding: utf-8 -*-

import xlrd
import os
import sys
import xlwt

excelpath = '/Users/zhangyu/Desktop/test_code.xls'
'''
桌面上的excel文件
'''
#写入excel，需要初始化workbook对象,编码为utf-8
wbw = xlwt.Workbook(encoding='utf-8')
sheet = wbw.add_sheet('what the hell')
sheet2 = wbw.add_sheet('superstar')

superstar = ['whos the superstar?','zhangyu','mr.zhang','angela-zhangyu']


sheet.write(0,0,'i am fucking high !')

a= 0
b= 0
for flag in superstar:
	sheet2.write(a,b,flag)
	a = a+1




wbw.save('/Users/zhangyu/Desktop/high2.xls')