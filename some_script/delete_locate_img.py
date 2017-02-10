# -*- coding: utf-8 -*-

import sys 
import os
import dicom


def delete_locate_imag(dirs='dir'):
    countdel = 0
    for direct,folders,filenames in os.walk(dirs):
        for filename in filenames:
            cmdstr = os.path.join(direct, filename)
            print(cmdstr)
            try:
                graph = dicom.read_file(cmdstr)
                thick = graph.SliceThickness
                if float(thick) <1 or float(thick)>1.25:
                    countdel +=1
                    print(cmdstr)
                    os.remove(cmdstr)
            except:
                continue
    print('the del pics number is:', countdel)



if __name__=='__main__':
    if len(sys.argv)<2 or len(sys.argv)>2 :
        print('typing filename and path')
    if len(sys.argv) ==2:
        dirs = sys.argv[1]
        delete_locate_imag(dirs=dirs)

