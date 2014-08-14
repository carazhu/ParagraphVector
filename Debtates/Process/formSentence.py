## formSentence.py
## Author: Yangfeng Ji
## Date: 08-13-2014
## Time-stamp: <yangfeng 08/13/2014 21:40:25>

import os, nltk
from nltk.tokenize.punkt import PunktWordTokenizer

def process_line(line):
    try:
        line = line.strip().split("|")[1]
    except IndexError:
        return []
    line = line.replace("<p>","").replace("[<i>Applause</i>]","").lower()
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(line)
    return sents


def main(path, fname_out):
    wordpunct = PunktWordTokenizer()
    fout = open(fname_out, 'w')
    filelist = os.listdir(path)
    for fname in filelist:
        fin = open(os.path.join(path, fname), 'r')
        for line in fin:
            sents = process_line(line)
            for sent in sents:
                words = wordpunct.tokenize(sent)
                fout.write((" ".join(words)) + '\n')
    print 'Done'


if __name__ == '__main__':
    path = "../Data/"
    fname_out = "../debtates-sent.txt"
    main(path, fname_out)
