import numpy as np

class HypothesisFunction:
    def __init__(self, l:int, m:int, k:int)-> tuple[np.ndarray, np.ndarray]:
        self.l:int = l
        self.m:int = m
        self.k:int = k
        
        self.Wh:np.ndarray = np.random.normal(size=(m,l))
        self.Wo:np.ndarray = np.random.normal(size=(k,m))
        
        self.bo:np.ndarray = np.zeros((k, 1))
        self.bh:np.ndarray = np.zeros((m,1))

    def forward(self, x):
        assert x.shape[0] == self.l, f"Your input must be consistent the value l={self.l}"
        
        a:np.ndarray = np.tanh(np.dot(self.Wh,x) + self.bh)
        
        y :np.ndarray= np.dot(self.Wo,a)+self.bo
        
        y: np.ndarray = np.maximum(0,y)
        return y, a

    def double_forward(self, x1:np.ndarray, x2:np.ndarray) -> np.ndarray:
        y1, _ = self.forward(x1)
        y1: np.ndarray
        y2, _ = self.forward(x2)
        y2:np.ndarray
        
        z:np.ndarray = np.concatenate((y1,y2))
        
        z_bar:np.ndarray = (z - np.mean(z)) / np.std(z)
        
        return z_bar
    
        
    def count_params(self):
        num_params = lambda z: np.prod(z.shape)
        total_params = num_params(self.Wh) + num_params(self.Wo) + num_params(self.bh) + num_params(self.bo)
        
        return total_params