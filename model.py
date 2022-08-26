from micrograd.nn import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out)
        self.n_in = n_in
        self.n_out = n_out
    
    def __call__(self, x):
        '''
        Forward pass.
        '''
        if x.data.shape[0] > 1:
            x.data = x.data.flatten()
        return super().__call__(x.data)
    
    def __repr__(self):
        return f"Linear(n_in={self.n_in}, n_out={self.n_out})"

def ScaledDotProductAttention(Q, K, V, mask=None):
        # set x as an array of type float
        x = np.array(Q * K.T, dtype=np.float)
        x /= np.sqrt(K.shape[-1])
        if mask is not None:
            x = x * mask
        return x * V

class MultiHeadAttention:
    def __init__(self, n_in, n_out, n_head=8):
        self.n_in = n_in
        self.n_out = n_out
        self.n_head = n_head
        # for each head, create 3 linear layers for Q, K, & V
        self.Q_l = [Linear(n_in, n_out) for _ in range(n_head)]
        self.K_l = [Linear(n_in, n_out) for _ in range(n_head)]
        self.V_l = [Linear(n_in, n_out) for _ in range(n_head)]
        # output layer
        self.out_l = Linear(n_in * n_head, n_out)
    
    def __call__(self, Q, K, V):
        '''
        Forward pass
        '''
        # get the outputs from each head
        Q_out = [l(Q) for l in self.Q_l]
        K_out = [l(K) for l in self.K_l]
        V_out = [l(V) for l in self.V_l]
        # run scaled dot product attention on each head
        scaled_out = [ScaledDotProductAttention(Q_out[i], K_out[i], V_out[i]) for i in range(self.n_head)]
        # concatenate the outputs from each head
        heads_out = np.concatenate(scaled_out, axis=1)
        # take the dot of out and the output layer
        return self.out_l(heads_out)