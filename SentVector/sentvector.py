## sentvector.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/14/2014 21:54:01>

import numpy
from huffman import *
from datastructure import WordCode, Instance

rng = numpy.random.RandomState(1234)

def sigmoid(x):
    return 1 / (1 + numpy.exp(x))

def get_codeindex(code):
    idx = len(code)
    return (2**(idx-1)) + int(code[:idx],2)

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
        # self.Word = numpy.asarray(rng.uniform(low=0, high=1.0,
        #                                     size=(n_dim, n_word)))
        self.Sent = numpy.asarray(rng.uniform(low=0, high=1.0,
                                            size=(n_dim, n_sent)))
        self.Feat = numpy.asarray(rng.uniform(low=0, high=1.0,
                                            size=(n_dim, n_feat+1)))
        self.b = numpy.zeros((n_feat+1,))
        W_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_dim + n_dim)),
            high=-numpy.sqrt(6. / (n_dim + n_dim)),
            size=(n_dim, n_dim)))
        self.W = W_values * 4
        # self.params = [self.Word, self.Sent, self.Feat, self.W, self.b]
        self.n_dim = n_dim
        self.nWords = n_word

    def hierarchical_softmax(self, word_idx, sent_idx, cont_list, code):
        """ Compute the hierarchical softmax for a given word
        Simple average, without involving any parameter - YJ

        Refer to __log_prob_path for parameters explanation
        """
        logprob = self.__log_prob_path(word_idx, sent_idx, cont_list, code)
        return numpy.exp(logprob.sum())

    def __log_prob_path(self, word_idx, sent_idx, cont_list, code):
        """ Following the huffman tree to compute the log probability
        """
        nWords = self.nWords
        # Average context words and sentence vector
        r_hat = numpy.zeros((self.n_dim,))
        for idx in cont_list:
            r_hat += self.Word[:,idx]
        r_hat = r_hat / len(cont_list) # Average
        # Add sentence vector
        r_hat += self.Sent[:,sent_idx]
        # Word vector (row vector)
        # word_vec = self.Word[:,word_idx].transpose()
        nCode = len(code)
        log_prob_path = numpy.zeros((nCode,))
        for idx in range(1, nCode+1):
            code_idx = get_codeindex(code[:idx])
            label = code[idx-1]
            vec = self.Feat[:,code_idx].T
            prob_idx_label_1 = sigmoid(numpy.dot(vec, r_hat) + self.b[code_idx])
            if label == '1':
                prob_idx = prob_idx_label_1
            elif label == '0':
                prob_idx = 1 - prob_idx_label_1
            print 'idx = {}, code = {}, code_idx={}, prob_{} = {}'.format(idx, code[:idx], code_idx, idx, prob_idx)
            log_prob_path[idx-1] = numpy.log(prob_idx)
        return log_prob_path

    def negative_log_likelihood(self, word_idx, sent_idx, cont_list, code):
        """ Return the mean of the hierarchical softmax for a given word
        """
        logprob = self.__log_prob_path(word_idx, sent_idx, cont_list, code)
        return -1.0 * logprob.sum()

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

