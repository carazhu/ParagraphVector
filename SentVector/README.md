#Paragraph/Sentence Vector Model#

##Code Structure##

- ParaVector class: includes every thing we need from the paragraph vector model 1) Initlization 2) Hiarchical Softmax 3) Negative Loglikelihood 4) Save model (mainly the shared variables) 5) Load model
- Learning class: includes 1) One word SGD --- read one word and update related parameters 2) One-sentence SGD --- read one sentence and update related parameters
- Preprocess class: which could 1) creates the dataset --- each row is one sentence 2) replace the low-frequency words with a special tag 3) then, replace each word and sentence their indices 4) build the haffman tree with word frequency 5) save the word-index and haffman code into file for future purpose
- Main function: create an instance of ParaVector class, then run the model 

## Reference ##

- A. Mnih and G. Hinton, *A Scalable Hierarchical Distributed Language Model*, 2009
- 