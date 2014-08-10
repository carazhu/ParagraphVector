## learning.py
## Author: Yangfeng Ji
## Date: 08-10-2014
## Time-stamp: <yangfeng 08/10/2014 14:01:12>

class Learning(object):
    def __init__(self, model, learning_rate, update_word=True):
        """ Initialize the parameters related to learning
        """
        self.model = model
        self.learning_rate = learning_rate
        self.update_word = update_word
        # Anything else here

    def sgd_one_word(self, word_id, context_id, sent_id):
        """ Read one word, using SGD to update related parameters
        """
        pass

    def sgd_minibatch(self, anything here):
        """
        """
        pass
