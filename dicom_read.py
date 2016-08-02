# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:23:16 2016

@author: tuixiangbeijingtest0
"""
import dicom 

import sys

reload(sys)  
sys.setdefaultencoding('utf8')


info = dicom.read_file('/Users/zhangyu/Desktop/12345')

a = 'Private tag data'

print info
#print info.KVP
#print info.'Private tag data'