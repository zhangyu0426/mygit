#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes

sudo THEANO_FLAGS=mode=FAST_RUN,device=gpu,cuda.root=/usr/local/cuda,floatX=float32,optimizer_including=conv_meta python cr_cnn_lasagne.py 'cnn' 100 500 'trained_model' 

if encounter errors do
sudo ldconfig /usr/local/cuda/lib64 
"""

from __future__ import print_function

import sys
import os
import time
#sys.path.append('/home/tx-wuhan-eva0/workspace/HipJoint/Lasagne/examples')
from generic_utils import Progbar


import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle

import lasagne

import scipy.io as sio
import scipy.misc as misc
#import matplotlib.pyplot as plt
import time
#from sklearn.datasets import make_classification

#from sklearn.cross_validation import train_test_split

import xlrd
import string
import glob

modeltype=sys.argv[1]

if modeltype=='vgg':
    from dataload_vgg import load_data
    from vgg16model_las import build_model, iterate_minibatches, read_model_param, write_model_param
    preload='vgg16.pkl'
else:
    from dataload_googlenet import load_data
    from googlenet_model_las import build_model, iterate_minibatches, read_model_param, write_model_param
    preload='blvc_googlenet.pkl'

saveto='SaveModels.param'
modelsavedir='/home/tx-wuhan-eva0/workspace/HipJoint/Lasagne/sk/'
modelloaddir='/home/tx-wuhan-eva0/workspace/HipJoint/Lasagne/ModelZoo/'



def mainmodel(model=modeltype, mini_batch=20,num_epochs=30, dro=0.7,lr=0.0001, preload=preload,saveto=saveto):
    print("model:%s minibatch:%d num_epochs:%d dropout:%f learingrate:%f\n" % (model,mini_batch,num_epochs,dro,lr)) 
   
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('input')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

        ## add input_var when needed in transfer learning model
    network = build_model(input_var=input_var,dro=dro)
 
    if preload:
#        with open(preload, 'r') as f:
#            data = pickle.load(f)
#        print(data)
#        lasagne.layers.set_all_param_values(model, data)
        if model=='vgg':
            read_model_param(network['fc7'],modelloaddir, preload)
        else:
            read_model_param(network['pool5/7x7_s1'],modelloaddir, preload)
        print('pretrained model loaded')
    start_time = time.time()
    networkout=network['prob']
    #networkout=network
    prediction = lasagne.layers.get_output(networkout)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(networkout, trainable=True)
    updates = lasagne.updates.adagrad(
            loss, params, learning_rate=lr, epsilon=1e-06)

    test_prediction = lasagne.layers.get_output(networkout, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], [loss,acc], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    time.time()-start_time
    # We iterate over epochs:
    train_accuracy=[]
    train_losses=[]
    val_accuracy=[]
    val_losses=[]
    time_id=str(int(time.time()))
    #outfilename='./benchmark/'+model+'_'+str(mini_batch)+'_'+str(num_epochs)+'_'+str(dro)+'_'+str(lr)+'_'+time_id+'.txt'
    outfilename='./benchmark/'+model+'_'+str(mini_batch)+'_'+str(num_epochs)+'_'+str(dro)+'_'+str(lr)+'.txt'
    print (outfilename)
    outfile=open(outfilename,'w')
    #outfile.write("model:%s mini_batch:%s num_epochs:%s dro:%s learningrate:%s\n" % (model,mini_batch,num_epochs,dro,lr))
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        print('Epoch',epoch)
        train_err = 0
        train_acc = 0
        train_batches = 0
        progbarcount = 0
        progbar=Progbar(30)
        for batch in iterate_minibatches(X_train, y_train, mini_batch, shuffle=True):
            inputs, targets = batch
            progbarcount = progbarcount + np.float(len(targets))/len(y_train)*25
            err, acc = train_fn(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1
            #print(train_batches)
            progbar.update(progbarcount,values=[('acc',round(train_acc/train_batches,3)),
                                                ('loss',round(train_err/train_batches,3))])
            
        train_accuracy.append(round(train_acc/train_batches,3))
        train_losses.append(round(train_err/train_batches,3))
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, np.min([mini_batch,len(y_val)]), shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        #print(val_acc,val_err,val_batches)
        progbar.update(30,values=[('val_acc',round(val_acc/val_batches,3)),
                                            ('val_loss',round(val_err/val_batches,3))])
        val_accuracy.append(round(val_acc/val_batches,3))
        val_losses.append(round(val_err/val_batches,3))

    # After training, we compute and print the test error:
    print (val_losses)
    '''
    outfile.write('\n')
    outfile.write('train_acc: ')
    for it in train_accuracy:
        outfile.write('%s,' % it)
    outfile.write('\n')
    outfile.write('train_loss: ')
    for it in train_losses:
        outfile.write('%s,' % it)
    outfile.write('\n')
    outfile.write('val_acc: ')
    for it in val_accuracy:
        outfile.write('%s,' % it)
    outfile.write('\n')
    outfile.write('vall_loss: ')
    for it in val_losses:
        outfile.write('%s,' % it)
    outfile.write('\n')
    '''
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_val, y_val, np.min([mini_batch,len(y_val)]), shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("")
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    #outfile.write('\n')
    #outfile.write("test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
    outfile.write("test accuracy:\t{:.2f}".format(test_acc / test_batches))
    outfile.close()
    write_model_param(networkout,modelsavedir,saveto)
    return train_err/train_batches, train_acc/train_batches * 100, 
    val_err/val_batches, val_acc/val_batches * 100


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print("model=modeltype, mini_batch,num_epochs, dro")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['mini_batch'] = int(sys.argv[2])
            kwargs['num_epochs'] = int(sys.argv[3])
            kwargs['dro']=float(sys.argv[4])
            kwargs['lr']=float(sys.argv[5])
        mainmodel(**kwargs)
