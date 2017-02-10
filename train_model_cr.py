#!/usr/bin/env python

"""
This is the code to find if a person is healthy.

TODO: Change the parameters above mainmodel() to fit your situation. 

Beijing: THEANO_FLAGS=mode=FAST_RUN,device=gpu1,cuda.root=/usr/local/cuda,floatX=float32,optimizer_including=conv_meta python run.py'

Wuhan: THEANO_FLAGS=mode=FAST_RUN,device=gpu0,cuda.root=/usr/local/cuda,floatX=float32,optimizer_including=cudnn python run.py 

if encounter errors do
sudo ldconfig /usr/local/cuda/lib64 

Author: zchaowei@infervision.com
"""

from __future__ import print_function

import sys
import time
sys.path.append('../lib')
sys.path.append('../model')
from generic_utils import Progbar, get_acc, fas_neg_entrop, load_data

import numpy as np
import theano
import theano.tensor as T

import lasagne

import random

from sklearn import preprocessing 

modeltype = 'vgg'

if modeltype == 'vgg':
    from vgg16model_las_sigmoid import build_model, iterate_minibatches, read_model_param, write_model_param
    preload = 'vgg16.pkl'
else:
    from googlenet_model_las import build_model, iterate_minibatches, read_model_param, write_model_param
    preload = 'blvc_googlenet.pkl'

# TODO: Modify the content below to fit your situation

modelsavedir = '/media/disk_sdb/params_1w_zzz_jjy_qt/'
modelloaddir = '../../../model/'
dataloaddir = '/media/disk_sdb/numpy/1w_zzz_jjy_qt_0921/'

print("Loading data...")
X_train=np.load(dataloaddir + '1w_zzz_jjy_qt16792_224_X_train.npy')
Y_train=np.load(dataloaddir + '1w_zzz_jjy_qt16792_224_Y_train.npy')
X_val=np.load(dataloaddir + '1w_zzz_jjy_qt4198_224X_val.npy')
Y_val=np.load(dataloaddir + '1w_zzz_jjy_qt4198_224Y_val.npy')

Y_train = Y_train.reshape(Y_train.shape[0])
Y_val = Y_val.reshape(Y_val.shape[0])
#X_train,Y_train,X_val,Y_val = load_data()

dropout = 0.95
learningrate = 0.00005
threshold = 0.2 #0.2
fn_weights = 2.0 #2.0

# TODO: Modify the content above to fit your situation

def mainmodel(dro=0.4, lr=0.00005, threshold = 0.5, fn_weights = 1.0, model=modeltype, mini_batch=20, num_epochs=500, preload=preload):
    print("model:%s minibatch:%d num_epochs:%d dropout:%f learingrate:%f threshold:%f fn_weights:%f\n" 
          %(model,mini_batch,num_epochs,dro,lr,threshold,fn_weights)) 

    print('Preprocessing data...')
    for i in range(X_train.shape[0]):
        X_train[i,0] = preprocessing.scale(X_train[i,0])
        X_train[i,1] = preprocessing.scale(X_train[i,1])
        X_train[i,2] = preprocessing.scale(X_train[i,2])

    for i in range(X_val.shape[0]):
        X_val[i,0] = preprocessing.scale(X_val[i,0])
        X_val[i,1] = preprocessing.scale(X_val[i,1])
        X_val[i,2] = preprocessing.scale(X_val[i,2])

    print ('The shape of training data is' + str(X_train.shape))
    print ('The shape of test data is' + str(X_val.shape))
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('input')
    target_var = T.matrix('targets')
    # target_var = T.matrix('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    # add input_var when needed in transfer learning model
    network = build_model(input_var=input_var,dro=dro)
 
    if preload:
        if model=='vgg':
            read_model_param(network['fc7'],modelloaddir, preload)
        else:
            read_model_param(network['pool5/7x7_s1'],modelloaddir, preload)
        print('pretrained model loaded')
    start_time = time.time()
    networkout = network['prob']

    prediction = lasagne.layers.get_output(networkout)
    # disable dropout
    test_prediction = lasagne.layers.get_output(networkout, deterministic=True)

    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = fas_neg_entrop(prediction, target_var, fn_weights)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(networkout, trainable=True)
    updates = lasagne.updates.adagrad(
            loss, params, learning_rate=lr, epsilon=1e-06)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    getpred_fn = theano.function([input_var], prediction)

    getpred_test_fn = theano.function([input_var], test_prediction)

    print("Starting training...")
    time.time()-start_time
    # We iterate over epochs:
    # log the accuracy of each epoch
    fout = open('../log/xin_log_dro_' + str(dro) +'_lr_'+ str(lr) +'_ts_'+ str(threshold) + '.log', 'w+')
    
    list_sort = []#postive,negtive,epoch

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        fout.write('Epoch'+str(epoch)+'\n')
        print('Epoch',epoch)

        train_batches = 0
        progbarcount = 0
        progbar=Progbar(30)

        # log the probabilities of prediction in each epoch
        output_path = '../log/prediction_epoch_' + str(epoch) + '.out'
        output_path_train = '../log/train_epoch_' + str(epoch) + '.out'

        with file(output_path_train, 'wb') as outfile_train:
            for batch in iterate_minibatches(X_train, Y_train, mini_batch, shuffle=True):
                inputs, targets = batch
                # reshape to fit the loss function calculation
                targets = targets.reshape((targets.shape[0],1))
                progbarcount = progbarcount + np.float(len(targets))/len(Y_train)*25

                batch_pred_train = getpred_fn(inputs)

                if train_batches == 0:
                    pred_train = batch_pred_train
                    y_label_train = targets
                    train_batches += 1
                else:
                    pred_train = np.concatenate((pred_train,batch_pred_train))
                    y_label_train = np.concatenate((y_label_train,targets))
                    train_batches += 1

                np.savetxt(outfile_train, batch_pred_train, fmt='%-7.2f')
                


                err =  train_fn(inputs, targets)
        
        outfile_train.close()
        progbar.update(progbarcount,values=[('train_batches',train_batches)])              

        # And a full pass over the validation data:

        val_batches = 0


        with file(output_path, 'wb') as outfile:
            pred_val = []
            for batch in iterate_minibatches(X_val, Y_val, np.min([mini_batch,len(Y_val)]), shuffle=True):
                inputs, targets = batch
                batch_pred_val = getpred_test_fn(inputs)
                if val_batches == 0:
                    pred_val = batch_pred_val
                    y_label_val = targets
                else:
                    pred_val = np.concatenate((pred_val,batch_pred_val))
                    y_label_val = np.concatenate((y_label_val,targets))

                np.savetxt(outfile, batch_pred_val, fmt='%-7.2f')
                val_batches += 1
                
        outfile.close()
        progbar.update(30,values=[('val_batches',val_batches)])

        ## For softmax
        # train_acc = get_acc(y_label_train, pred_train[:,1], threshold)
        # val_acc = get_acc(y_label_val, pred_val[:,1], threshold)

        # For sigmoid
        train_acc = get_acc(y_label_train, pred_train, threshold)
        val_acc = get_acc(y_label_val, pred_val, threshold)

        # img_save_path = '../log/curve_epoch_' + str(epoch) + '.png'
        # plot_recall_curve(y_label_val,pred_val,img_save_path)

        # all_param_values = lasagne.layers.get_all_param_values(networkout)

#        print ('Accuracy       : {:.2f} %'.format(100*train_acc[0][0]))
#        print ('Precision      : {:.2f} %'.format(100*train_acc[0][3]))
#        print ('Positive Recall: {:.2f} %'.format(100*train_acc[0][1]))
#        print ('Negtive Recall : {:.2f} %'.format(100*train_acc[0][2]))
#
#        print ('Val Accuracy       : {:.2f} %'.format(100*val_acc[0]))
#        print ('Val Precision      : {:.2f} %'.format(100*val_acc[3]))
#        print ('Val Positive Recall: {:.2f} %'.format(100*val_acc[1]))
#        print ('Val Negtive Recall : {:.2f} %'.format(100*val_acc[2]))

        print ('Accuracy       : {:.2f}%  {:.2f}%'.format(100*train_acc[0][0], 100*val_acc[0]))
        print ('Precision      : {:.2f}%  {:.2f}%'.format(100*train_acc[0][3], 100*val_acc[3]))
        print ('Positive Recall: {:.2f}%  {:.2f}%'.format(100*train_acc[0][1], 100*val_acc[1]))
        print ('Negtive Recall : {:.2f}%  {:.2f}%'.format(100*train_acc[0][2], 100*val_acc[2]))

        if 0 == epoch:
            list_sort.append([val_acc[0],val_acc[3],val_acc[1],val_acc[2],epoch])
        saveflag = True
        i = 0
        while i< len(list_sort) and i < 30:
            if 1.0 == val_acc[1] and val_acc[2] < 0.1:
                saveflag = False
                break
            elif val_acc[1] > list_sort[i][2] and abs(train_acc[0][0]-val_acc[0])> 0.05 and train_acc[0][0] < 0.99:
                list_sort.insert(i, [val_acc[0],val_acc[3],val_acc[1],val_acc[2],epoch])
                break
            elif val_acc[1] == list_sort[i][2] and val_acc[2] > list_sort[i][3] and \
            abs(train_acc[0][0]-val_acc[0])> 0.05 and train_acc[0][0] < 0.99:
                list_sort.insert(i, [val_acc[0],val_acc[3],val_acc[1],val_acc[2],epoch])
                break
            else:
                i +=1
        if 30 == i:
            saveflag = False

            
        fout.write('Accuracy       : {:.2f} %'.format(100*train_acc[0][0]))
        fout.write('Precision      : {:.2f} %'.format(100*train_acc[0][3]))
        fout.write('Positive Recall: {:.2f} %'.format(100*train_acc[0][1]))
        fout.write('Negtive Recall : {:.2f} %'.format(100*train_acc[0][2]))

        fout.write('Val Accuracy       : {:.2f} %'.format(100*val_acc[0]))
        fout.write('Val Precision      : {:.2f} %'.format(100*val_acc[3]))
        fout.write('Val Positive Recall: {:.2f} %'.format(100*val_acc[1]))
        fout.write('Val Negtive Recall : {:.2f} %'.format(100*val_acc[2]))
        
        if True == saveflag:
            saveto = ('1w_zzz_jjy_qt_SavedModels_Epoch_' + str(epoch) +
                     '_dro_' + str(dro) +'_lr_'+ str(lr) +'_ts_'+ str(threshold)+ '.params')
            write_model_param(networkout,modelsavedir,saveto)
                     
        

    # After training, we compute and print the test error:
    for one in list_sort[:30]:
        print(one)
    fout.close()
    print ("Training Completed!")
    return 0

if __name__ == '__main__':

    # drop_out = [0.3,0.4,0.5,0.6,0.7,0.6,0.5,0.4]
    # learning_rate = [0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.0001, 0.0001, 0.0001]
    # for i in range(len(drop_out)):
    #     dropout = drop_out[i]
    #     learningrate = learning_rate[i]
    #     mainmodel(dropout, learningrate)
    #     time.sleep(660)

    mainmodel(dropout, learningrate,threshold, fn_weights)
