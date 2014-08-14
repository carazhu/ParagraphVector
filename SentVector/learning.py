## learning.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/13/2014 20:00:41>

import theano
import theano.tensor as T
import numpy
from datastructure import WordCode, Instance
from sentvector import SentVector

class Learning(object):
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
        index = T.lscalar()
        cost = self.model.negative_log_likelihood(self.trndata[index])
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
                                                givens = {index:index})

            
    def sgd_one_word(self, index):
        """ Read one word, using SGD to update related parameters

        :type index: int
        :param index: index of training instance
        """        
        n_instance = len(self.trndata)
        Index = shuffle(range(n_instance))
        for index in Index:
            print 'index = {}'.format(index)
            # Print out NLL for sanity check
            print self.model.negative_log_likelihood(self.trndata[index])
            self.train_function(index)
            print self.model.negative_log_likelihood(self.trndata[index])
            
            

    def sgd_minibatch(self, anything here):
        """
        """
        pass
