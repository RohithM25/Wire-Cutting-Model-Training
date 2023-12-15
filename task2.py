import numpy as np


def softmax(vector):
    vector = np.exp(vector)
    return vector/np.sum(vector)

class Task2_Model:
    def __init__(self, data: np.ndarray, weights: np.ndarray, labels: np.ndarray):
        self.data = data # Should be a m x n matrix
        self.w = weights # This should be a n x out vector
        self.labels = labels # Should be a m x out vector

    def __call__(self, d = np.array([])):
        if not d.any():
            lin = np.dot(self.data, self.w)
        else:
            lin = np.dot(d, self.w)
        return softmax(lin)
    
    def cross_entropy_loss(self, epsilon=1e-10):
        preds = self()
        actual = self.labels
        losses = []

        for p, a in zip(preds, actual):
            logs = np.log(p + epsilon)
            logs *= a
            cross_entropy = - np.sum(logs)
            if not np.isnan(cross_entropy) and not np.isinf(cross_entropy): losses.append(cross_entropy)
        avg = sum(losses)/actual.shape[0]
        return avg
    
if __name__ == '__main__':
    tdata = np.load('OHE_Task2_Trainset2000/data.npy')
    tlabels = np.load('OHE_Task2_Trainset2000/labels.npy')
    m,n = tdata.shape
    m,o = tlabels.shape
    model = Task2_Model(data=tdata, weights=np.random.rand(n, o), labels=tlabels)

    

