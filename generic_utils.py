from __future__ import absolute_import,print_function
import numpy as np
import time
import sys

import theano
# from theano.tensor import basic as tensor, subtensor, opt, elemwise
import theano.tensor as T

import matplotlib.pyplot as plt

def load_data():

    #X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    X_normal = np.load('/media/disk_sdb/numpy/ZengZhiZao_zy/CR_1w_chest10565_224_X_train0.npy')
    Y_normal = np.load('/media/disk_sdb/numpy/ZengZhiZao_zy/CR_1w_chest10565_224_Y_train0.npy')
    X_abnormal=np.load('/media/disk_sdb/numpy/ZengZhiZao_zy/CR_1w_chest10376_224_X_train1.npy')
    Y_abnormal=np.load('/media/disk_sdb/numpy/ZengZhiZao_zy/CR_1w_chest10376_224_Y_train1.npy')

    print (Y_normal[0])
    print (Y_abnormal[0])

    X_train_normal = X_normal[0:7000]
    Y_train_normal = Y_normal[0:7000]

    
    X_train_abnormal = X_abnormal[0:7000]
    Y_train_abnormal = Y_abnormal[0:7000]

    
    X_train = np.concatenate((X_train_normal,X_train_abnormal))

    Y_train = np.concatenate((Y_train_normal,Y_train_abnormal))

    
    X_val_normal = X_normal[7000:10000]
    Y_val_normal = Y_normal[7000:10000]

    
    X_val_abnormal = X_abnormal[7000:10000]
    Y_val_abnormal = Y_abnormal[7000:10000]
    
    X_val = np.concatenate((X_val_normal,X_val_abnormal))
    Y_val = np.concatenate((Y_val_normal,Y_val_abnormal))
    
    print("shape of X_train: ",X_train.shape)
    print("shape of Y_train: ",Y_train.shape)
    print("shape of X_val: ",X_val.shape)
    print("shape of Y_val: ",Y_val.shape) 
    


    return X_train,Y_train,X_val,Y_val 


def fas_neg_entrop(output, target, K):
    ''' this function compute a new loss function that takes into account false negativity 
    K = constant that modifies the weight on the false negativity loss term
    '''
    one = np.float32(1.0)
    output = T.clip(output, 0.0001, 0.9999)  # don't piss off the log
    cost = -(K*target * T.log(output) + (one - target) * T.log(one - output)-0.0*target*T.log(one - output))

    return cost


def cross_entropy(output, target):

    ''' this function compute a new loss function that takes into account false negativity 
    K = constant that modifies the weight on the false negativity loss term
    '''

    one = np.float32(1.0)
    output = T.clip(output, 0.0001, 0.9999)  # don't piss off the log
    cost = -(1.0*target * T.log(output) + (one - target) * T.log(one - output))

    return cost

def cross_entropy_train(output, target, weights):

    ''' this function compute a new loss function that takes into account false negativity 
    K = constant that modifies the weight on the false negativity loss term
    '''
    one = np.float32(1.0)
    output = T.clip(output, 0.0001, 0.9999)  # don't piss off the log
    weights2 = [1.0/(1.0-1.0/x) for x in weights]

    cost = -T.sum(weights * target * T.log(output) + weights2 * (one - target) * T.log(one - output), axis=1) # Sum over all labels
    cost = T.mean(cost, axis=0) # Compute mean over n_samples
    return cost

def gen_confusion_matrix(targets, prob_pred, threshold):
    """ Calculate the confusion matrix according to provided data.

    Parameters
    ----------
    targets : array, shape = [n_samples]
        True targets of binary classification in range {0, 1}.

    prob_pred : array, shape = [n_samples]
        Estimated probabilities.

    threshold : float
        Threshold to determine whether a probability jumps into 0 or 1

    """
    assert(len(targets) == len(prob_pred))

    matrix = np.zeros((2, 2),dtype=np.int)

    new_prob = np.copy(prob_pred)

    new_prob[new_prob > threshold] = 1
    new_prob[new_prob <= threshold] = 0


    for i in range(new_prob.shape[0]):
        if new_prob[i] == 1 and targets[i] == 1: # True positive
            matrix[0][0] += 1
        if new_prob[i] == 0 and targets[i] == 0: # True negative
            matrix[1][1] += 1
        if new_prob[i] == 1 and targets[i]== 0: # False positive
            matrix[1][0] += 1
        if new_prob[i] == 0 and targets[i] == 1: # False negative
            matrix[0][1] += 1

    return matrix


def cal_acc(matrix):
    """ Calculate the accuracy, positive_recall, negative recall, precision and f1_score

    Parameters
    ----------
    matrix : 2-D array
        The confusion matrix

    ACC = (tp+tn)/(tp+tn+fp+fn)

    positive_recall = tp/(tp+fn)

    negative_recall = tn/(tn+fp)

    precision = tp/(tp+fp)

    f1_score = 2*positive_recall*precision/(positive_recall+precision)

    """
    acc = 1.0*(matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[1][1]+matrix[1][0]+matrix[0][1])
    if (matrix[0][0]+matrix[0][1]) == 0:
        positive_recall = 0.00
    else:
        positive_recall = 1.0*matrix[0][0]/(matrix[0][0]+matrix[0][1])
    if (matrix[1][1]+matrix[1][0]) == 0:
        negative_recall = 0.00
    else:
        negative_recall = 1.0*matrix[1][1]/(matrix[1][1]+matrix[1][0])
    if (matrix[0][0]+matrix[1][0]) == 0:
        precision = 0.00
    else:
        precision = 1.0*matrix[0][0]/(matrix[0][0]+matrix[1][0])
    if (positive_recall+precision) == 0:
        f1_score = 0.00
    else:
        f1_score = 2*positive_recall*precision/(positive_recall+precision)

    return acc, positive_recall, negative_recall, precision, f1_score

def get_acc(targets, prob_pred, threshold = 0.5):
    """ Calculate the accuracy, positive_recall, negative recall, precision and f1_score
        For multiple labels
    Parameters
    ----------
    targets : array or ndarray, shape = [n_samples,n_labels]
        True targets of binary classification in range {0, 1}.

    prob_pred : array or ndarray, shape = [n_samples,n_labels]
        Estimated probabilities.

    threshold : float
        Threshold to determine whether a probability jumps into 0 or 1

    Return
    ----------
    result: 2-D array
        Each row is an instance, while columns are [acc, positive_recall, negative_recall, precision, f1_score]
    """
    if len(targets.shape) == 1:
        num_of_labels = 1
        result = np.zeros(5,dtype=np.float)
        matrix = gen_confusion_matrix(targets, prob_pred, threshold)
        acc, positive_recall, negative_recall, precision, f1_score = cal_acc(matrix)
        result[0] = acc
        result[1] = positive_recall
        result[2] = negative_recall
        result[3] = precision
        result[4] = f1_score

    else:
        num_of_labels = targets.shape[1]
        result = np.zeros((num_of_labels, 5),dtype=np.float)

        for i in range(num_of_labels):
            matrix = gen_confusion_matrix(targets[:,i], prob_pred[:,i], threshold)
            acc, positive_recall, negative_recall, precision, f1_score = cal_acc(matrix)
            result[i][0] = acc
            result[i][1] = positive_recall
            result[i][2] = negative_recall
            result[i][3] = precision
            result[i][4] = f1_score

    return result

def plot_recall_curve(targets, prob_pred, savepath):
    threshold = [0.1*i for i in range(11)]
    num_of_labels = targets.shape[1]
    positive_recall = np.zeros((num_of_labels, 11),dtype=np.float)
    negative_recall = np.zeros((num_of_labels, 11),dtype=np.float)
    for i in range(11):
        acc = get_acc(targets, prob_pred, threshold[i])

        positive_recall[:,i] = acc[:,1]
        negative_recall[:,i] = acc[:,2]

    type_of_line = ['ro--','bs--','g^--','y--','k-.']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Disease Detect Rate')
    plt.ylabel('Healthy Cleared Rate')

    for i in range(num_of_labels):
        plt.plot(positive_recall[i],negative_recall[i],type_of_line[i], label = ('Class'+str(i)))
        for j in range(11):
            ax.annotate(threshold[j], (positive_recall[i][j],negative_recall[i][j]))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig(savepath)
    plt.close(fig)


def display_info(train_acc, val_acc, label_names):

    assert(len(label_names) == train_acc.shape[0])
    for i in range (len(label_names)):
        print ('Accuracy of ' + label_names[i] + ': {:.2f} %'.format(100*train_acc[i][0]), end = ' ' )
        print ('Precision of ' + label_names[i] + ': {:.2f} %'.format(100*train_acc[i][3]), end='  ')
        print ('Recall of   ' + label_names[i] + ': {:.2f} %'.format(100*train_acc[i][1]), end='  ')
        print ('\n')
    for i in range (len(label_names)):
        print ('Val Accuracy of ' + label_names[i] + ': {:.2f} %'.format(100*val_acc[i][0]), end='  ')
        print ('Val Precision of ' + label_names[i] + ': {:.2f} %'.format(100*val_acc[i][3]), end='  ')
        print ('Val Recall of   ' + label_names[i] + ': {:.2f} %'.format(100*val_acc[i][1]), end='  ')
        print ('\n')

def display_info2(train_acc, val_acc, label_names,index):

    print ('Accuracy of ' + label_names[index] + ': {:.2f} %'.format(100*train_acc[0]), end = ' ' )
    print ('Precision of ' + label_names[index] + ': {:.2f} %'.format(100*train_acc[3]), end='  ')
    print ('Recall of   ' + label_names[index] + ': {:.2f} %'.format(100*train_acc[1]), end='  ')
    print ('\n')
    for i in range (len(label_names)):
        print ('Val Accuracy of ' + label_names[i] + ': {:.2f} %'.format(100*val_acc[i][0]), end='  ')
        print ('Val Precision of ' + label_names[i] + ': {:.2f} %'.format(100*val_acc[i][3]), end='  ')
        print ('Val Recall of   ' + label_names[i] + ': {:.2f} %'.format(100*val_acc[i][1]), end='  ')
        print ('\n')


class Progbar(object):
    def __init__(self, target, width=30, verbose=1):
        '''
            @param target: total number of steps expected
        '''
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[]):
        '''
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if type(self.sum_values[k]) is list:
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)
''' write a function to reshape a single channel numpy array to a multi-channel numpy array such that 
the shape fits the pretrained models, by subsampling pixels around the grids 
nparray: is of shape (height, width), 
gridsize: is an integer denoting the small grid size, 
subsampcount: is the number of pixels to subsample from the grid 

return: an nparray of shape (subsampcount, height/gridsize, width/gridsize)'''

def subchannel(nparray, gridsize):
    if len(nparray.shape)!=2:
        raise ValueError('dimension of nparray not 2')
    nparrayall=[]
    for ii in range(gridsize):
        for jj in range(gridsize):
            nparrayij=nparray[ii::gridsize,jj::gridsize]
            nparrayall.append(nparrayij)
    return np.array(nparrayall)

        
if __name__ == '__main__':
    add()

