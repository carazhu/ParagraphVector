## huffman.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/10/2014 15:04:51>

class WordCode(object):
    def __init__(self, index, word, code, freq):
        self.word = word
        self.index = index
        self.code = code
        self.freq = freq

class HuffmanCode(object):
    def __init__(self, anything_here):
        """
        """
        self.code = {}

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

        Regular word index starts from 1, index 0 for all
        low-frequency words
        """
        
        

    
