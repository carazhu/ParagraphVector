## main.py
## Author: Yangfeng Ji
## Date: 08-13-2014
## Time-stamp: <yangfeng 08/16/2014 22:13:02>

from sentvector import SentVector
from cPickle import load
from datastructure import Instance
from learning import SGDLearn
import gzip

def main():
    n_word, n_sent = 3731, 24415
    n_feat, n_dim = 2**16, 50
    print 'Load data ...'
    trndata = load(gzip.open("../Debtates/data-sample.pickle.gz"))
    print 'Create a SentVector model ...'
    sv = SentVector(n_word, n_sent, n_feat, n_dim)
    print 'Create a SGDLearn instance ...'
    learner = SGDLearn(sv, trndata)
    # print 'Update parameters with one instance ...'
    # learner.sgd_one_word(1)
    print 'Update parameters with entire dataset (one pass) ...'
    learner.sgd_perword()
    # learner.sgd_minibatch()


if __name__ == '__main__':
    main()
