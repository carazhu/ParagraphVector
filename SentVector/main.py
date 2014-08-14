## main.py
## Author: Yangfeng Ji
## Date: 08-13-2014
## Time-stamp: <yangfeng 08/14/2014 18:45:04>

from sentvector import SentVector
from cPickle import load
from datastructure import Instance
from learning import SGDLearn
import gzip

def main():
    n_word = 3731
    n_sent = 24414
    n_feat = 12
    n_dim = 100
    print 'Create a SentVector model ...'
    sv = SentVector(n_word, n_sent, n_feat, n_dim)
    print 'Load data ...'
    trndata = load(gzip.open("../Debtates/data-sample.pickle.gz"))
    print 'Create a SGDLearn instance ...'
    learner = SGDLearn(sv, trndata)
    print 'Update parameters with one instance ...'
    learner.sgd_one_word(0)


if __name__ == '__main__':
    main()
