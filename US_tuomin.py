# -*- coding: utf-8 -*-


#20160809  US图片格式是US.1.*  CT图片格式是1.* 使用时需要根据情况更改前缀
import os 
import sys
import magic

def create_logging(filepath='info.log'):
    import logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=filepath,
                        filemode='w')                   
    print('create log success')     
    return logging

def desensitization(gdcmIDlist,filepath,savepath,accessionno=None ):
    import os    
    rmstr=''
    firststr=''
    try:
            for x,y in gdcmIDlist:
                rmstr+='--remove  %s,%s ' % (x,y)  
            os.system("gdcmanon --dumb %s %s %s" % (rmstr,filepath,savepath))
             
            if accessionno==None:
               firststr='File:%s' %(savepath.replace('\\','/').split('/')[-1])  
            else:
               firststr='accessionno:%s' %(str(accessionno))              
            print("%s\nDesensitization execution success :Patient's Name,Institution Name moved" % (firststr))
            #firststr="%s Desensitization execution success :Patient's Name,Institution Name,date_time_verified moved" % (firststr)
            firststr="Desensitization execution success :Patient's Name,Institution Name moved\n"   
            logging_hyp.info(savepath)            
            logging_hyp.info(firststr)
        
    except Exception as e:
        print(e)
        
if __name__=='__main__':
    if len(sys.argv)==1:
        print 'Please input folder path (e.g.):python tuomin.py /home/user/folderpath'
    else:
       if len(sys.argv)>2:
           print 'error:Enter too much,or path contains spaces'
       else:
           folderpath=sys.argv[1]
           if os.path.exists(folderpath):
               print('folder path:'+folderpath)
               gidlist=[(10,10),(8,80),(8,81)] 
               logging_hyp = create_logging(str(folderpath)+'/tuomin.log')
               print ('log path:'+folderpath+'/tuomin.log')
               
               logging_hyp.info('folder path:'+folderpath+'\n')
               dicomlist=os.popen('find '+'\''+str(folderpath)+'\''+' -name \'US.1.*\'')
               count = 0
               for dicomfile in dicomlist:
               	   dicomfile = dicomfile.replace('\n','')
               	   print(dicomfile)
               	   if os.path.isfile(dicomfile):
                       print('1')
               	       label = magic.from_file(dicomfile,mime = True)
               	       print(label)
               	       if (label == 'application/dicom'):
               	       	   count = count +1
                           print ('------------------------------------------------------------')
                           #dicomfile = dicomfile.replace('\n','')
                           print dicomfile
                           dicomfile = dicomfile.replace(' ','\ ')
                           print dicomfile
                           desensitization(gidlist,dicomfile,dicomfile)
               print(count)


           else:
               print 'This is not a floder path,please check and try again'
        
        
        
        
