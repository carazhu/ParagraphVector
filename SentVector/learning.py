## learning.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/14/2014 18:57:17>

import theano
import theano.tensor as T
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
        ind = T.iscalar()
        cost = self.model.negative_log_likelihood(ind)
        # Gradient
        gparams = []
        for param in self.model.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)
        # Update rules
        updates = []
        for (param, gparam) in zip(self.model.params, gparams):
            updates.append((param, param - self.learning_rate * gparam))
        self.train_function = theano.function(inputs = [index],
                                                outputs = cost,
                                                updates = updates,
                                                givens = {ind:index})


    def sgd_one_word(self, index):
        self.train_function(index)
        
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
