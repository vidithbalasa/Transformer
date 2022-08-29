from micrograd.engine import Value
from linear import Linear
import numpy as np
import math
from typing import List

from tensor import flatten_2d

class MultiHeadAttention:
    def __init__(self, n_in, n_out, n_head=8):
        self.n_in = n_in
        self.n_out = n_out
        self.n_head = n_head

        # for each head, create 3 linear layers for Q, K, & V
        self.Q_l = [Linear(n_in, n_in) for _ in range(n_head)]
        self.K_l = [Linear(n_in, n_in) for _ in range(n_head)]
        self.V_l = [Linear(n_in, n_in) for _ in range(n_head)]

        self.out_l = Linear(n_in * n_head, n_out) # output layer
    
    def __call__(self, Q, K, V) -> List[Value]:
        '''
        Forward pass
        1. Run each head (Q, K, V) through the corresponding linear layers
        2. Run each head through ScaledDotProductAttention
        3. Concatenate the heads and run through the output layer
        4. Run the output of all the heads through a linear layer
        '''
        Q_out: List[Value] = [l(Q) for l in self.Q_l] # step 1
        K_out = [l(K) for l in self.K_l]
        V_out = [l(V) for l in self.V_l]

        scaled_out = [self.ScaledDotProductAttention(Q_out[i], K_out[i], V_out[i]) for i in range(self.n_head)] # step 2

        heads_out = flatten_2d(scaled_out) # step 3 // concatenating a list of lists is the same as flattening it
        
        return self.out_l(heads_out) # step 4
    
    def ScaledDotProductAttention(self, Q, K, V, mask=None) -> list:
        '''
        Attention(Q,K,V) = softmax( (Q • K^T) / sqrt(d_k) ) • V
        '''
        K_T = np.asarray(K).transpose()

        x = [qi * xi for qi, xi in zip(Q, K_T)] # Q • K
        x = [i/math.sqrt(len(K)) for i in x] # x / sqrt(d_k)

        scale = sum([math.exp(i.data) for i in x])
        if scale:
            x = [Value(math.exp(i.data)) / scale for i in x] # softmax(x)

        if mask is not None:
            x = x * mask

        x = [xi * vi for xi, vi in zip(x, V)] # x • V
        
        return x