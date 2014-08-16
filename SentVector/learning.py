## learning.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/15/2014 23:15:25>

import numpy
from datastructure import WordCode, Instance
from sentvector import SentVector

class SGDLearn(object):
    def __init__(self, model, trndata, learning_rate=1e-4):
        """ Initialize the parameters related to learning

        :type model: instance of class SentVector
        :param model: SentVector model

        :type trndata: list of Instance
        :param trndata: Training data

        :type learning_rate: float
        :param learning_rate: initial learning rate
        """
        self.model = model
        self.trndata = trndata
        self.learning_rate = learning_rate

    def sgd_one_word(self, index):
        """ Update parameters with one training sample

        :type index: int
        :param index: index of training sample
        """
        param_grads = self.model.gradient(self.trndata[index])
        print "Before update:", self.model.hierarchical_softmax(self.trndata[index])
        self.model.grad_update(param_grads, self.learning_rate)
        print "After update:", self.model.hierarchical_softmax(self.trndata[index])
            
        
    def sgd_per_word(self):
        """ Read one word, using SGD to update related parameters

        :type index: int
        :param index: index of training instance
        """
        n_instance = len(self.trndata)
        Index = shuffle(range(n_instance))
        for index in Index:
            print 'index = {}'.format(index)
            self.train_function(index)

    def sgd_minibatch(self, anything_here):
        """
        """
        pass
