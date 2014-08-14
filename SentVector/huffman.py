## huffman.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/13/2014 12:23:33>

from huffmancoding import *
from datastructure import WordCode

class HuffmanCode(object):
    def __init__(self, anything_here):
        """
        """
        self.code = {}
        self.max_length = 0

    def load(self, fname):
        """
        """
        fin = open(fname, 'r')
        for line in fin:
            items = line.strip().split("\t")
            wc = WordCode(int(items[0]), items[1], items[2],
                          float(items[3]))
            self.code[items[1]] = wc
        fin.close()
        return self.code

    def save(self, fname):
        """

        Data format
        word \t code \t word-freq
        """
        fout = open(fname, 'r')
        for (key, wc) in self.code.iteritems():
            fout.write(str(wc.index) + "\t" + wc.word + "\t" + wc.code + "\t" + str(wc.freq) + "\n")
        fout.close()

    def coding(self, word_freq):
        """
        word_freq: {word:freq}
        1, Regular word index starts from 1, index 0 for all
           low-frequency words
        """
        word_list = []
        prob_list = []
        for (word, prob) in word_freq.iteritems():
            word_list.append(word)
            prob_list.append(prob)
        code_list = huffman(prob_list)
        for (idx, code) in enumerate(code_list):
            wc = WordCode(idx, word_list[idx], code, freq_list[idx])
            if self.max_length < len(code):
                self.max_length = len(code)
            self.code[word] = wc
        return self.code
        

    
