# -*- coding: utf-8 -*-

import xlrd
import os
import sys
import xlwt

excelpath = '/Users/zhangyu/Desktop/test_code.xls'
'''
桌面上的excel文件
'''

wb = xlrd.open_workbook(excelpath)
#获取sheet页的名
wb_sheet_name = wb.sheet_names()
#获取第一个sheet
wb_sheet_by_index = wb.sheet_by_index(0)
#获取行数
rownumber = wb_sheet_by_index.nrows  #3
#获取第一列内容
first_col = wb_sheet_by_index.col_values(0)
#print(first_col)


#  str和repr的差异 --打印sheet_name
'''
print(wb_sheet_name)
print(u','.join(wb_sheet_name))
for w in wb_sheet_name:
	print w
'''

#循环打印每一行的信息
'''
for rownum in range(wb_sheet_by_index.nrows):
	print(u','.join(wb_sheet_by_index.row_values(rownum)))
'''

#通过索引读数据
cell_one = wb_sheet_by_index.cell(1,0).value
#print(cell_one)







