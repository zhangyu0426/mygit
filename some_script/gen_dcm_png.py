import dicom
import os
import cv2
import scipy.misc as misc
import numpy as np

dicom_dir ='/media/tx-eva/e74a794d-2dd4-458d-a90e-21c333efcad0/mxnet_test_data/test_1k_cr_jiejie/mnt/PACSS/'
image_dir ='/media/tx-eva/e74a794d-2dd4-458d-a90e-21c333efcad0/mxnet_test_data/test_1k_cr_jiejie/patient_jpg'
cnt = 0

#python dcm2jpg_cw.py --dicom_dir /media/tx-eva/e74a794d-2dd4-458d-
#a90e-21c333efcad0/mxnet_test_data/test_1k_cr_jiejie/mnt/PACSS/ --image_dir /media/tx-eva/e74a794d-2dd4-458d-a90e-21c333efcad0/mxnet_test_data/test_1k_cr_jiejie/patient_jpg




unzip_path = './temp.out'
print(dicom_dir)
try:
for dirpath,dirnames,filenames in os.walk(dicom_dir):
    print(filenames)
    for filename in filenames:
        print(filename)
        filepath = os.path.join(dirpath,filename)
        filepath=filepath.replace(' ','\ ')
        print('Processing:' + filepath)
        cmd='gdcmconv -w '+filepath+' '+unzip_path #gdcm2pnm
        os.system(cmd)

    ds = dicom.read_file(unzip_path)
    pix = ds.pixel_array
    print(pix.shape)
    patient_id = ds.PatientID
    bodypart,viewposition=ds.BodyPartExamined,ds.ViewPosition
    monochrome = ds.PhotometricInterpretation

    if bodypart=='CHEST' and viewposition=='PA':
        center = ds.WindowCenter
        width = ds.WindowWidth

        if isinstance(center,list):
            center=center[0]
        if isinstance(width,list):
            width=width[0]

        low=center-width/2
        hig=center+width/2
        pix_out=np.zeros(pix.shape)

        w1=np.where(pix>low) and np.where(pix<hig)

        pix_out[w1]=((pix[w1]-center+0.5)/(width-1)+0.5)*255
        pix_out[np.where(pix<=low)]=pix[np.where(pix<=low)]=0
        pix_out[np.where(pix>=hig)]=pix[np.where(pix>=hig)]=255

        pix_out = misc.imresize(pix_out,[pix_out.shape[0], pix_out.shape[1]])
        if monochrome == 'MONOCHROME1':
            pix_out = 255 - pix_out

        jpg_path = os.path.join(image_dir, (patient_id + '.jpg'))

        pix_out = cv2.cvtColor(pix_out,cv2.COLOR_GRAY2RGB)

        cv2.imwrite(jpg_path,pix_out)

 
 
except:
    print('pass')     
 