## sentvector.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/13/2014 20:56:15>

import theano
import theano.tensor as T
import numpy
from huffman import *
from datastructure import WordCode, Instance

class SentVector(object):
    def __init__(self, n_word, n_sent, n_feat, n_dim):
        """ Initialize the parameters of the SentVector model

        :type input: one instance of WordCode
        :param input: 
        
        :type n_word: int
        :param n_word: number of words

        :type n_sent: int
        :param n_sent: number of sentences

        :type n_feat: int
        :param n_feat: number of feature vectors, used for
                       hierarchical softmax
        
        :type n_dim: int
        :param n_dim: number of latent dimension
        """
        self.Word = theano.shared(value=numpy.zeros((n_dim, n_word),
                                                    dtype=theano.config.floatX),
                                                    name='Word')
        self.Sent = theano.shared(value=numpy.zeros((n_dim, n_sent),
                                                    dtype=theano.config.floatX),
                                                    name='Sent')
        self.Feat = theano.shared(value=numpy.zeros((n_dim, n_feat+1),
                                                    dtype=theano.config.floatX),
                                                    name='Feat')
        self.W = theano.shared(value=numpy.zeros((n_dim, n_dim),
                                                 dtype=theano.config.floatX),
                                                 name='W')
        self.b = theano.shared(value=numpy.zeros((n_feat+1,),
                                                 dtype=theano.config.floatX),
                                                 name='b')
        self.params = [self.Word, self.Sent, self.Feat, self.W, self.b]


    def hierarchical_softmax(self, input):
        """ Compute the hierarchical softmax for a given word
        Simple average, without involving any parameter - YJ

        :type input: Instance
        :param input: an instance of class Instance
        """
        word_idx = input.windex
        sent_idx = input.sindex
        context_list = input.clist
        code = input.code
        # 
        nWords = self.Word.shape[1]
        nCode = len(code)
        # Average context words and sentence vector
        r_hat = numpy.zeros((self.n_dim,))
        for idx in cont_list:
            r_hat += self.Word[:,idx]
        r_hat += self.Sent[:,sent_idx]
        r_hat = r_hat / (len(cont_list) + 1) # Average
        # Word vector (row vector)
        word_vec = self.Word[:,word_idx].transpose()
        logprob = 0.0
        for idx in range(nCode):
            # Along the code to compute hierarchical softmax
            if idx == 0:
                code_idx = -1
                label = code[0]
            else:
                code_idx = int(code[:idx], 2)
                label = code[idx]
            prob_idx_label_1 = T.nnet.sigmoid(T.dot(word_vec, r_hat) + self.b[code_idx])
            if label == '1':
                prob_idx = prob_idx_label_1
            elif label == '0':
                prob_idx = 1 - prob_idx_label_1
            logprob += T.log(prob_idx)
        return T.exp(logprob)

    def negative_log_likelihood(self, input):
        """ Return the mean of the hierarchical softmax for a given word
        """
        return -T.log(self.hierarchical_softmax(input))

    def save_model(self, fname):
        """ Save the shared variables into files

        :type fname: string
        :param fname: file name
        """
        pass

    def load_model(self, fname):
        """
        """
        pass

