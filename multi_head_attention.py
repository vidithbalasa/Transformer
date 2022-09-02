import math
from linear import Linear
from tensor import add_tensors, apply, flatten_nd, matmul_4d, shape, softmax, transpose, zeros_from

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