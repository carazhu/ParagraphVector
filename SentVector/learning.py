## learning.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/16/2014 21:55:50>

import numpy, time
import pp
from datastructure import WordCode, Instance
from sentvector import SentVector
from random import shuffle


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

    def sgd_oneword(self, index):
        """ Update parameters with one training sample

        :type index: int
        :param index: index of training sample
        """
        param_grads = self.model.gradient(self.trndata[index])
        print "Before update:", self.model.hierarchical_softmax(self.trndata[index])
        self.model.grad_update(param_grads, self.learning_rate)
        print "After update:", self.model.hierarchical_softmax(self.trndata[index])
            
        
    def sgd_perword(self):
        """ Read one word, using SGD to update related parameters
        """
        Index = range(len(self.trndata))
        shuffle(Index)
        for (n, index) in enumerate(Index):
            if (n+1) % 100 == 0:
                print 'Update with {} samples'.format(n+1)
            param_grads = self.model.gradient(self.trndata[index])
            if (n+1) % 100 == 0:
                print "Before update:", self.model.hierarchical_softmax(self.trndata[index])
            if param_grads is not None:
                self.model.grad_update(param_grads, self.learning_rate)
            if (n+1) % 100 == 0:
                print "After update:", self.model.hierarchical_softmax(self.trndata[index])

            
    def sgd_minibatch(self):
        """ Update parameters with one batch of training examples

        :type batch_size: int
        :type batch_size: size of one mini-batch
        """
        ppservers = ()
        ncpus = 2
        job_server = pp.Server(ncpus, ppservers=ppservers)
        print "Starting parallel computing with {} workers".format(job_server.get_ncpus())
        inputs = range(ncpus)
        start_time = time.time()
        jobs = [(input, job_server.submit(self.model.gradient, (self.trndata[input],), (), ())) for input in inputs]
        for input, job in jobs:
            print "Sum of primes below", input, "is", job()
        print "Time elapsed: {} s".format(time.time() - start_time)
        job_server.print_stats()
