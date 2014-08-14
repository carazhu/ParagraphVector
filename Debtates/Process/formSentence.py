## formSentence.py
## Author: Yangfeng Ji
## Date: 08-13-2014
## Time-stamp: <yangfeng 08/14/2014 12:00:57>

import os, nltk
from nltk.tokenize import word_tokenize

def process_line(line):
    try:
        line = line.strip().split("|")[1]
    except IndexError:
        return []
    line = line.replace("<p>","").replace("[<i>Applause</i>]","").lower()
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(line)
    return sents


def main(path, fname_out, fname_index):
    fout = open(fname_out, 'w')
    findex = open(fname_index, 'w')
    filelist = os.listdir(path)
    for fname in filelist:
        counter = 0
        fin = open(os.path.join(path, fname), 'r')
        for line in fin:
            sents = process_line(line)
            for sent in sents:
                words = word_tokenize(sent)
                fout.write((" ".join(words)) + '\n')
                findex.write(fname + "\t" + str(counter) + "\n")
                counter += 1
    fout.close()
    findex.close()
    print 'Done'


if __name__ == '__main__':
    path = "../Data/"
    fname_out = "../debtates-sent.txt"
    fname_index = "../debtates-sent-index.txt"
    main(path, fname_out, fname_index)
