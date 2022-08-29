from micrograd.nn import Layer
from tensor import flatten_2d, is_valid_tensor

class Linear(Layer):
    '''
    A single dense layer.
    '''
    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out)
        self.n_in = n_in
        self.n_out = n_out
    
    def __call__(self, x:list):
        '''
        Forward pass.
        '''
        assert is_valid_tensor(x), "Input must be a valid 1D or 2D tensor."

        if isinstance(x[0], list):
            x = flatten_2d(x) # flatten if 2D

        return super().__call__(x)
    
    def __repr__(self):
        return f"Linear(n_in={self.n_in}, n_out={self.n_out})"