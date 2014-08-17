#Paragraph/Sentence Vector Model#

##Code Structure##

- Preprocess class: which could 1) creates the dataset --- each row is one sentence 2) replace the low-frequency words with a special tag 3) then, replace each word and sentence their indices 4) build the haffman tree with word frequency 5) save the word-index and haffman code into file for future purpose
- HuffmanCode class: which builds a Huffman tree according to word frequencies
- GenSample class: which generates all training example, one example per word
- ParaVector class: includes every thing we need from the paragraph vector model 1) Initlization 2) Hiarchical Softmax 3) Negative Loglikelihood 4) Save model (mainly the shared variables) 5) Load model
- Learning class: includes 1) One word SGD --- read one word and update related parameters 2) Per-word SGD --- one pass updating with all training examples (words)
- Main function: create an instance of ParaVector class, then run the model 

## Hyper-parameter ##

- Word frequency threshold for pre-processing (one word will be removed from the vocab, if it shows up less than the given threshold)
- Initial learning rate (default: 1e-4)
- With or without AdaGrad updating
- Size of latent dimension
- Number of updating pass (default: 30)

## Reference ##

- Q. Le and T. Mikolov, *Distributed Representation of Sentences and Documents*, 2014 (For the basic model framework, with minor changes)
- A. Mnih and G. Hinton, *A Scalable Hierarchical Distributed Language Model*, 2009 (For the technical detail of the hierarchical softmax method)
- J. Duchi and E. Hazan and Y. Singer, *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*, 2010 (For AdaGrad algorithm)