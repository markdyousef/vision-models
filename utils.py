import threading
import numpy as np
import random

class BatchIndices(object):
    def __init__(self, n, batch_size, shuffle=False):
        self.n, self.batch_size, self.shuffle = n, batch_size, shuffle
        self.lock = threading.Lock()
        self.reset()
    
    def reset(self):
        self.idxs = (np.random.permutation(self.n) if self.shuffle else np.arange(0, self.n))
        self.current = 0
    
    def __next__(self):
        with self.lock:
            if self.current >= self.n:
                self.reset()
            ni = min(self.batch_size, self.n-self.current)
            res = self.idxs[self.current: self.current+ni]
            self.current += ni
            return res

class segment_generator(object):
    def __init__(self, x, y, batch_size=64, out_size=(224, 224), train=True):
        self.x, self.y, self.batch_size, self.train = x, y, batch_size, train
        self.n, self.row, self.column, _ = x.shape
        self.idx_gen = BatchIndices(self.n, batch_size, train)
        self.height, self.width = out_size
        self.channels = self.y.shape[-1] if len(y.shape) == 4 else 1

    def get_slice(self, i, o):
        start = random.randint(0, i-o) if self.train else (i-o)
        return slice(start, start+o)

    def get_item(self, idx):
        slice_row = self.get_slice(self.row, self.height)
        slice_column = self.get_slice(self.column, self.width)
        x = self.x[idx, slice_row, slice_column]
        y = self.y[idx, slice_row, slice_column]
        
        if self.train and (random.random() > 0.5):
            y = y[:, ::-1]
            x = x[:, ::-1]
        return x, y

    def __next__(self):
        idxs = next(self.idx_gen)
        items = (self.get_item(idx) for idx in idxs)
        xs, ys = zip(*items)
        return np.stack(xs), np.stack(ys).reshape(len(ys), -1, self.channels)