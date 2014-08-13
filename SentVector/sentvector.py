## sentvector.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/12/2014 22:19:42>

import theano
import theano.tensor as T
import numpy

class SentVector(object):
    def __init__(self, n_word, n_sent, n_feat, n_dim):
        """ Initialize the parameters of the SentVector model

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

    def hierarchical_softmax(self, word_idx, sent_idx, cont_list, code):
        """ Compute the hierarchical softmax for a given word
        Simple average, without involving any parameter - YJ

        :type word_idx: int
        :param word_idx: word index

        :type sent_idx: int
        :param sent_idx: sentence index

        :type context_list: int list
        :param context_list: a list of context indices for the given word

        :type code: binary string
        :param code: a list of '1' and '0' string
        """
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

    def negative_log_likelihood(self, word_idx, sent_idx, cont_list, code):
        """ Return the mean of the hierarchical softmax for a given word
        """
        return -T.log(self.hierarchical_softmax(word_idx, sent_idx, cont_list, code))

    def save_model(self, fname):
        """ Save the shared variables into files

        :param fname: output file name
        """
        pass

    def load_model(self, fname):
        """
        """
        pass

