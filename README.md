# Transformer
I built and trained a transformer model completely from scratch. It was able to achieve >25% on Engligh-German translation, within 5% of the author's original results.

*Index*
1. Multihead Attention
2. Positional Encoding
3. Multidimensional Matrix Math
4. Optimizer

## Multihead Attention
Multihead attention modules are the heart of transformers. They are what allows the model to effectively attend to multiple parts of the input at the same time.

I implemented multihead attention using the scaled dot-product attention equation. This equation takes in three matrices: the query matrix, the key matrix, and the value matrix. The query and key matrices are multiplied together, and then this product is divided by the square root of the dimensionality of the key matrix. This is done to keep the dot product from getting too large. Finally, a softmax is applied to the result to get the attention weights which are then used to combine the value matrix and produce the final output.

In self-attention, the query, key, and value matrices are all the same. This is used to understand relationships between words in the same document. In encoder-decoder attention, the query is the output from the previous iteration of the decoder. By mapping the decoder output to the original input sentence, the model is able to map relationships between the output and input document, which is very useful for translation.

`CODE`

## Positional Encoding
Unlike previous natural language models such as RNNs, transformers take in a sequence of words as input, but do not keep track of the order of the words in the sequence. This is because the transformer is based on the attention mechanism, which is not order-dependent.

In order to add some sense of order back into the model, positional encoding is used. This is a vector of values which are added to the word embeddings. My implementation of positional encoding followed the paper directly. The authors use an algorithm where the positional encoding of even tokens are a sinusoid and the positional encoding of odd tokens are a cosinusoid.

`CODE`

## Multidimensional Matrix Math
In order to implement the attention mechanism and positional encoding, I had to become quite proficient in matrix math. This included taking the dot product of two matrices, matrix multiplication, and transposing matrices.

I found that the best way to understand these operations was to think of them in terms of lists. For example, taking the dot product of two matrices is just taking the dot product of the two lists of numbers which make up the matrices.

Multidimensional matrix multiplication is a bit more complicated, but can be thought of as a series of dot products. First, the columns of the first matrix are dotted with the rows of the second matrix. This process can be expanded so that you may choose any 2 arbitrary dimensions to multiply together. My implementation is heavily based on a 2010 paper written by Ashu that I found on page 2 of Google[1].

`CODE`

## Optimizer
I used the Adam optimizer[2] with a learning rate of 0.0001 and a batch size of 64. I found that these hyperparameters worked well on my training data. I trained my model for 10 epochs and found that it converged after about 5 epochs.

`CODE`

## Results
I was able to achieve a translation accuracy of >25% on the English-German translation task, within 5% of the author's original results.

`CODE`

## References
[1] Ashu, M. (2010). Matrix multiplication from a higher perspective. Retrieved from http://www.cse.iitk.ac.in/users/ashu/cs365/l6.pdf

[2] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.