from micrograd.engine import Value
from linear import Linear
import math
from typing import List

from tensor import view_by_head

class MultiHeadAttention:
    def __init__(self, d_model = 512, n_head = 8):
        '''
        Params
        ------
        @d_model: embedding size
        @n_head: number of heads        
        d_k: embedding size after paritioning the output by n_head

        Layers
        ------
        W_Q, W_K, W_V: Linear layers for the queries, keys, and values
        W_O1, W_O2: Linear layers for the output
        '''
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

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

        1. Run each sequence (Q, K, V) through the corresponding linear layers
           - Spits out 3 matrices of shape (batch_size, sequence_length, embedding_size) 
        2. Transpose the matrix from 
            (batch_size, sequence_length, embedding_size) -> (batch_size, n_heads, sequence_length, d_k)
        3. Run matrix through ScaledDotProductAttention
        4. Run the output of all the heads through output layers
        '''
        Q = [self.W_Q(q) for q in Q]
        K = [self.W_K(k) for k in K]
        V = [self.W_V(v) for v in V]

        Q = view_by_head(Q, self.d_k)
        K = view_by_head(K, self.d_k)
        V = view_by_head(V, self.d_k)

        x = self.ScaledDotProductAttention(Q, K, V)

        x = [self.W_O1(i) for i in x]
        x = [self.W_O2(i) for i in x]

        return x

    def ScaledDotProductAttention(self, Q, K, V, mask=None) -> list:
        '''
        Assume Q,K,V are all matrices of shape (batch_size, n_heads, sequence_length, d_k)

        Attention(Q,K,V) = softmax( (Q • K^T) / sqrt(d_k) ) • V
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