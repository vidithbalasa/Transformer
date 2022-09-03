# Transformer
```
Paper
-------
[Attention is All You Need](https://arxiv.org/abs/1706.03762)

Authors
-------
Asish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uzkoreit, 
Llion Jones, Adian N. Gomez, Lukasz Kaiser, Illia Polosukhin
```
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

```python
# model.py
class MultiHeadAttention:
    def __init__(self, d_model = 512, n_head = 8, max_seq_len = 512):
        '''
        Params
        ------
        @d_model: embedding size
        @n_head: number of heads        
        d_k: embedding size after paritioning the embedding vector by n_head

        Layers
        ------
        W_Q, W_K, W_V: Linear layers for the queries, keys, and values
        W_O1, W_O2: Linear layers for the output
        '''
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        in_out = (self.d_model * max_seq_len,) * 2 
        self.W_Q = Linear(*in_out)
        self.W_K = Linear(*in_out)
        self.W_V = Linear(*in_out)

        self.W_O1 = Linear(*in_out)
        self.W_O2 = Linear(*in_out, nonlin=False)
    
    def __call__(self, Q, K, V) -> list[list]:
        '''
        Forward pass
        Assume Q, K, and V are all word embeddings of shape (batch_size, sequence_length, embedding_size)

        Steps
        -----
        1. Run each sequence (Q, K, V) through the corresponding linear layers
           - Spits out 3 matrices of shape (batch_size, sequence_length, embedding_size) 
        2. Transpose the matrix from 
            (batch_size, sequence_length, embedding_size) -> (batch_size, n_heads, sequence_length, d_k)
        3. Get attention score per head
        4. Concatenate attention scores
            - (b, h, l, d) -> (b, l, h, d) -> (b, l, h*d)
        5. Run attention scores through the output layers
        6. Add the residual connection
        '''
        input = Q

        Q = [self.W_Q(q) for q in Q]
        K = [self.W_K(k) for k in K]
        V = [self.W_V(v) for v in V]

        Q = self.view_by_head(Q)
        K = self.view_by_head(K)
        V = self.view_by_head(V)

        x = self.ScaledDotProductAttention(Q, K, V)

        x = [list(map(list, zip(*seq))) for seq in x] # (b,h,l,d) -> (b,l,h,d)
        # (b,l,h,d) -> (b,l,h*d)
        for idx in range(len(x)):
            x[idx] = [flatten_nd(seq) for seq in x[idx]]

        x = [self.W_O1(i) for i in x]
        x = [self.W_O2(i) for i in x]

        return add_tensors(input, x)       

    def ScaledDotProductAttention(self, Q, K, V, mask=None) -> list:
        '''
        Assume Q,K,V are all matrices of shape (batch_size, n_heads, sequence_length, d_k) == (b,h,l,d)

        Equation
        --------
        Attention(Q,K,V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
        '''
        K_T = zeros_from(shape(K)) # K -> K^T || (b,h,l,d) -> (b,h,d,l)
        for i in range(len(K)):
            for j in range(len(K[0])):
                K_T[i][j] = transpose(K[i][j])

        x = matmul_4d(Q, K_T) # Q * K^T = x || shape(x) = (b,h,l,l)

        scale = math.sqrt(len(K))
        x = apply(x, lambda i: i / scale) # x / sqrt(d_k)

        x = softmax(x)

        if mask is not None:
            x += mask

        x = matmul_4d(x, V) # x • V
        
        return x
    
    def view_by_head(self, l: list) -> list:
        '''
        Assume l is a matrix of shape (batch_size, sequence_length, d_model)

        Change the matrix to view in shape (batch_size, sequence_length, n_heads, d_k)

        Return a matrix of shape (batch_size, n_heads, sequence_length, d_k)
        '''
        matrix = []
        for seq in l:
            new_seq = []
            for token in seq:
                x = [token[x:x+self.d_k] for x in range(0, len(token), self.d_k)]
                new_seq.append(x)
            matrix.append(new_seq)
    
        return [list(map(list, zip(*seq))) for seq in matrix]
```

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
