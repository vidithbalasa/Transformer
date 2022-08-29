from micrograd.engine import Value
from linear import Linear
# import numpy as np
import math
from typing import List

from tensor import flatten_2d

class MultiHeadAttention:
    def __init__(self, d_k = 64, n_head = 8):
        '''
        Params
        ------
        @d_k: dimensionality of the keys (i.e. embedding size)
        @n_head: number of heads        
        d_model: dimensionality of the model

        Layers
        ------
        W_Q, W_K, W_V: Linear layers for the queries, keys, and values
        W_O1, W_O2: Linear layers for the output
        '''
        self.d_k = d_k
        self.n_head = n_head
        self.d_model = d_k * n_head

        in_out = (self.d_model, self.d_model) 
        self.W_Q = Linear(*in_out)
        self.W_K = Linear(*in_out)
        self.W_V = Linear(*in_out)

        self.W_O1 = Linear(*in_out)
        self.W_O2 = Linear(*in_out, nonlin=False)
    
    def __call__(self, Q, K, V) -> List[Value]:
        '''
        Forward pass
        Assume Q, K, and V are all word embeddings of shape (batch_size, sequence_length, embedding_size)

        1. Run each matrix (Q, K, V) through the corresponding linear layers
        2. Run each head through ScaledDotProductAttention
        3. Concatenate the heads and run through the output layer
        4. Run the output of all the heads through a linear layer
        '''
        Q_out = self.W_Q(Q)
        K_out = self.W_K(K)
        V_out = self.W_V(V)

        return

    def ScaledDotProductAttention(self, Q, K, V, mask=None) -> list:
        '''
        Attention(Q,K,V) = softmax( (Q • K^T) / sqrt(d_k) ) • V
            where Q,K,V are matrices of shape [batch_size, seq_len, d_k]
            where d_k is the dimensionality of the keys (i.e. d_model)
        A single scaled dot product head.
        '''
        # change Q from [B,s,d_k] to []
        # K_T = np.asarray(K).transpose()
        K_T = K

        x = [qi * xi for qi, xi in zip(Q, K_T)] # Q • K
        x = [i/math.sqrt(len(K)) for i in x] # x / sqrt(d_k)

        scale = sum([math.exp(i.data) for i in x])
        if scale:
            x = [Value(math.exp(i.data)) / scale for i in x] # softmax(x)

        if mask is not None:
            x = x * mask

        x = [xi * vi for xi, vi in zip(x, V)] # x • V
        
        return x