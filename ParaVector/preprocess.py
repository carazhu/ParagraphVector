## preprocess.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/10/2014 15:24:14>

import string
from huffman import WordCode, HuffmanCode

class Preprocess(object):
    def __init__(self, thresh=1.0):
        """ Initialize the parameters related to pre-processing
        """
        self.thresh = 1.0
        self.word_freq = {}

    def wordfreq(self, fname):
        """ Create word freqency dictionary
        """
        word_count = defaultdict(int)
        fin = open(fname, "r")
        for line in fin:
            words = line.strip().split()
            for word in words:
                word_count[word] += 1
        # First pass, remove low-freq words and
        # compute the overall counts
        total_count = 0.0
        for (word, count) in word_count.iteritems():
            if (count >= self.thresh) and (word not in string.puncutaton):
                self.word_freq[word] = count
                total_count += count
        # Second pass, normalize the probability
        for (word, count) in self.word_freq.iteritems():
            self.word_freq[word] /= total_count
        
    
    def clean(self, fname_in, fname_out, fname_code):
        """ Create a huffman codebook and clean the datafile

        :INPUT fname_in: raw data file
        :INPUT fname_out: cleaned data file
        :INPUT fname_code: codebook file
        """
        # Word frequency
        self.wordfreq(fname_in)
        # Coding and save code
        coder = HuffmanCode()
        codebook = coder.coding(self.wordfreq)
        coder.save(fname_code)
        # Call __clean
        self.__clean(fname_in, fname_out, codebook)
        
    def cleanwithvocab(self, fname_in, fname_out, fname_code):
        """ Clean the datafile with a pre-computed codebook
        """
        # Load huffman codebook
        codebook = load(fname_code)
        # Call __clean
        self.__clean(fname_in, fname_out, codebook)
        

    def __clean(self, fname_in, fname_out, codebook):
        """ Clean the datafile with a given codebook
        """
        # Clean file
        fin = open(fname_in, "r")
        fout = open(fname_out, "w")
        sent_counter = 0
        for line in fin:
            words = line.strip().split()
            ids = []
            for word in words:
                if (word not in string.puncutation):
                    try:
                        wc = codebook[word]
                        ids.append(wc.index)
                    except KeyError:
                        word = "RAREWORD"
                        wc = codebook[word]
                        ids.append(wc.index)
            ids = map(str, ids)
            line_ids = str(sent_counter) + "\t" + (" ".join(ids))
            fout.write(line_ids + "\n")
            sent_counter += 1
            # Print out information
            if (sent_counter % 1000 == 0):
                print "Process {} lines".format(sent_counter)
        fin.close()
        fout.close()
        print "DONE"
