## preprocess.py
## Author: Cara Zhu
## Date: 05-19-2015
## Time-stamp: <Cara 05/19/2015 14:42:16>

import string
from huffman import HuffmanCode
from datastructure import WordCode
from collections import defaultdict
from sentvector import SentVector

def main():
    n_word, n_sent = 5158, 22431
    n_feat, n_dim = 2**17, 50
    print ('Load model ...')
    sv = SentVector(n_word, n_sent, n_feat, n_dim)    
    #word,sent,u,v,feat,b = sv.load_model("model.pickle.gz")
    sv.load_model("model.pickle.gz")
    print("Word Vector \n{}").format(sv.Word)
    print("Sent Vector \n{}").format(sv.Sent)
        


if __name__ == '__main__':
    main()
