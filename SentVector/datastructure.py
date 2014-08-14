## datastructure.py
## Author: Yangfeng Ji
## Date: 08-13-2014
## Time-stamp: <yangfeng 08/13/2014 22:16:22>

class WordCode(object):
    def __init__(self, index, word, code, freq):
        """ Data structure for collecting Huffman code for word

        :type index: int
        :param index: index of word

        :type word: string
        :param word: word itself

        :type code: string
        :param code: binary code string for word

        :freq code: float
        :param code: word frequency in data
        """
        self.word = word
        self.index = index
        self.code = code
        self.freq = freq


class Instance(object):
    def __init__(self, windex, sindex, clist, code):
        """ Data structure for training/test instance

        :type windex: int
        :param windex: index of word

        :type sindex: int
        :param sindex: index of sentence

        :type clist: list
        :param clist: a list of word index as context

        :type code: string
        :param code: a binary string as the huffman code of word
        """
        self.windex = windex
        self.sindex = sindex
        self.clist = clist
        self.code = code
