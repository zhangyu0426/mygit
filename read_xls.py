#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import xlrd
import sys
import xlwt


workbook = xlrd.open_workbook('/Users/zhangyu/Desktop/guanjianci_chest_zy_0704_new.xls')
sheet = workbook.sheet_names()
sheet_name= ('').join(sheet)
print(sheet_name)

sheet_name1 = workbook.sheet_by_index(0)
sheet_name2 = workbook.sheet_by_index(1)

print(sheet_name2)
