#######################
#first make sure the img size in date_load is right
#########################
import os
import random


minibatch=[20,40,60,80]
num_epochs=[30,50]
dropouts=[0.5,0.6,0.7,0.8,0.9]
lrs=[0.00006,0.0001,0.0005,0.001,0.01]

groups=[]
for i in range(len(minibatch)):
    for j in range(len(num_epochs)):
        for k in range(len(dropouts)):
            for s in range(len(lrs)):
                onegroup=[]
                onegroup.append(minibatch[i])
                onegroup.append(num_epochs[j])
                onegroup.append(dropouts[k]) 
                onegroup.append(lrs[s])
                groups.append(onegroup)    
print groups
print len(groups)

num_try=len(minibatch)*len(num_epochs)*len(dropouts)*len(lrs)
print ("Totally try %d times!\n",num_try)
num_gpu=4
num_loop=num_try/num_gpu
print ("There are %d loops!\n", num_loop)

for idx in range(num_loop):
    print ("%d turn is running now\n"% idx)
    
    print "using the GPU-0"
    str0='THEANO_FLAGS=mode=FAST_RUN,device=gpu0,cuda.root=/usr/local/cuda,floatX=float32,optimizer_including=cudnn python main_las.py google '
    str0=str0+str(groups[idx*4][0])+' '+str(groups[idx*4][1])+' '+str(groups[idx*4][2])+' '+str(groups[idx*4][3])+' '
    str0=str0+'2>&1 >log0.txt &'
    print str0
    os.system(str0) 
    
    print "using the GPU-1"
    str1='THEANO_FLAGS=mode=FAST_RUN,device=gpu1,cuda.root=/usr/local/cuda,floatX=float32,optimizer_including=cudnn python main_las.py google '
    str1=str1+str(groups[idx*4+1][0])+' '+str(groups[idx*4+1][1])+' '+str(groups[idx*4+1][2])+' '+str(groups[idx*4+1][3])+' '
    str1=str1+'2>&1 >log0.txt &'
    print str1
    os.system(str1)
 
    print "using the GPU-2"
    str2='THEANO_FLAGS=mode=FAST_RUN,device=gpu2,cuda.root=/usr/local/cuda,floatX=float32,optimizer_including=cudnn python main_las.py google '
    str2=str2+str(groups[idx*4+2][0])+' '+str(groups[idx*4+2][1])+' '+str(groups[idx*4+2][2])+' '+str(groups[idx*4+2][3])+' '
    str2=str2+'2>&1 >log0.txt &'
    print str2
    os.system(str2)

    print "using the GPU-3"
    str3='THEANO_FLAGS=mode=FAST_RUN,device=gpu3,cuda.root=/usr/local/cuda,floatX=float32,optimizer_including=cudnn python main_las.py google '
    str3=str3+str(groups[idx*4+3][0])+' '+str(groups[idx*4+3][1])+' '+str(groups[idx*4+3][2])+' '+str(groups[idx*4+3][3])+' '
    print str3
    os.system(str3)
