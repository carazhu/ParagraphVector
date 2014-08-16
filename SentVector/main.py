## main.py
## Author: Yangfeng Ji
## Date: 08-13-2014
## Time-stamp: <yangfeng 08/15/2014 23:16:21>

from sentvector import SentVector
from cPickle import load
from datastructure import Instance
from learning import SGDLearn
import gzip

def main():
    n_word = 3731
    n_sent = 24414
    n_feat = 2**16
    n_dim = 100
    print 'Load data ...'
    trndata = load(gzip.open("../Debtates/data-sample.pickle.gz"))
    print 'Create a SentVector model ...'
    sv = SentVector(n_word, n_sent, n_feat, n_dim)
    print 'Create a SGDLearn instance ...'
    learner = SGDLearn(sv, trndata)
    print 'Update parameters with one instance ...'
    learner.sgd_one_word(1)


if __name__ == '__main__':
    main()
