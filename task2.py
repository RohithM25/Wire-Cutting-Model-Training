import numpy as np
from regression import add_polynomial_features


def softmax(matrix):
    exp_matrix = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    return exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)

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
    
    def cross_entropy_loss(self, preds, actual, epsilon=1e-10):
        preds = np.clip(preds, epsilon, 1 - epsilon)
        m = actual.shape[0]
        loss = -np.sum(actual * np.log(preds + epsilon)) / m
        return loss
    
    def accuracy(self, preds, actual):
        predicted_labels = np.argmax(preds, axis=1)
        true_labels = np.argmax(actual, axis=1)
        correct_predictions = np.sum(predicted_labels == true_labels)
        total_samples = actual.shape[0]
        acc = correct_predictions / total_samples
        return acc
    
    def train(self, iters, lr = 0.05, batch=None, reset_weights = True, reg_lambda = 0.01):
        if not batch:
            batch = self.data.shape[0]
        if reset_weights:
            self.w = np.zeros((self.data.shape[1], self.labels.shape[1]))
        for i in range(iters):
            ri = np.random.permutation(len(self.data))
            data = self.data[ri][:batch]
            labels = self.labels[ri][:batch]
            pred = self(data)
            loss = self.cross_entropy_loss(pred, labels)
            # Add L2 regularization penalty
            loss += reg_lambda * 0.5 * np.sum(self.w ** 2)
            grad = np.dot(data.T, pred-labels)/batch + reg_lambda * self.w
            self.w -= lr*grad
        print(f"Training final loss: {loss}, Training final accuracy {model.accuracy(model(), model.labels)}")

        
        
def one_hot_encode(label):
    new_labels = []
    indices = []
    for i in range(label.shape[0]):
        if label[i] != 0:
            indices.append(i)
            arr = np.array([0]*5)
            arr[label[i]] = 1
            new_labels.append(arr)
    return np.array(new_labels), np.array(indices)


    
if __name__ == '__main__':
    tdata = np.load('OHE_Task2_Trainset2000/data.npy')
    tlabels = np.load('OHE_Task2_Trainset2000/labels.npy')
    # tlabels, ind = one_hot_encode(tlabels[:,1])
    # tdata = tdata[ind]
    # # tdata = add_polynomial_features(tdata, 2)
    np.column_stack((tdata, np.ones((tdata.shape[0],1))))

    m,n = tdata.shape
    m,o = tlabels.shape

    model = Task2_Model(data=tdata, weights=np.zeros((n, o)), labels=tlabels)
    print("init loss", model.cross_entropy_loss(model(), model.labels), "init acc", model.accuracy(model(), model.labels))
    model.train(5000, batch=300)


    
    

