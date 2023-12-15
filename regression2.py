import numpy as np
from regression import add_neighborhood_feature, add_polynomial_features

# We can think of data as m x n matrix. m -> number of data points/samples, n -> number of features per datapoint
# We can multiply m x n matrix with weights vector n x 1, then apply sigmoid to all of these to get m results
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class GD_Model:
    def __init__(self, data: np.ndarray, weights: np.ndarray, labels: np.ndarray):
        self.data = data # Should be a m x n matrix
        self.w = weights # This should be a n x 1 vector
        self.labels = labels # Should be a m x 1 vector
    
    # Applies fn to all features and concatenates them together
    # data matrix goes from m x n to m x 2n

    def add_feature_fn(self, fn):
        feature_data = fn(self.data + 1e-20)
        if feature_data.shape[0]==self.data.shape[0]:
            self.data = np.concatenate((self.data,feature_data), axis=1)
        else:
            raise ValueError(f"shapes where mismatched. New features have {feature_data.shape[0]} rows, original data has {self.data.shape[0]} rows")
        self.w = np.ones((self.data.shape[1],1)) # reset weights
    
    def __call__(self, d = np.array([])):
        if not d.any():
            lin = np.dot(self.data, self.w)
        else:
            lin = np.dot(d, self.w)
        return sigmoid(lin)
    
    # We will call the model and compare results to labels
    def LCE(self, f, y, epsilon=1e-15):
        # print(f.shape, y.shape)
        f = np.clip(f, epsilon, 1-epsilon)
        assert f.shape[0] == y.shape[0]
        return - np.mean(y*np.log(f) + (np.ones_like(y) - y)*np.log(np.ones_like(f) - f))
    
    def accuracy(self, threshhold = 0.5):
        pred = self()
        bin = pred >= threshhold
        count = sum(1 if p == y else 0 for p,y in zip(bin, self.labels))
        return count/len(self.labels)


    def train(self, iters, lr = 0.05, batch=None, reset_weights = True, reg_lambda = 0.01):
        if not batch:
            batch = self.data.shape[0]
        if reset_weights:
            self.w = np.ones((self.data.shape[1],1))
        for i in range(iters):
            ri = np.random.permutation(len(self.data))
            data = self.data[ri][:batch]
            labels = self.labels[ri][:batch]
            pred = self(data)
            loss = self.LCE(pred, labels)
            # Add L2 regularization penalty
            loss += reg_lambda * 0.5 * np.sum(self.w ** 2)

            grad = np.matmul(data.T, pred-labels[:,np.newaxis]) + reg_lambda * self.w

            self.w -= lr*(1/batch)*grad
        print(f"Training final loss: {loss}, Training final accuracy: {self.accuracy()}")

# For a given datapoint get the sum of each color
def get_sum_colors(data):
    all = []
    for i in range(data.shape[0]):
        last_color = int(np.max(data[i])+1)
        colors = [0]*last_color
        for j in range(last_color):
            colors[j] = np.sum(data[i] == i)
        all.append(np.array(colors))
    return np.array(all)

if __name__ == '__main__':
    data = np.load('5k_data/data.npy')
    labels = np.load('5k_data/labels.npy')
    data = add_neighborhood_feature(data, window_size=5)
    regression = GD_Model(data, np.ones((data.shape[1],1)), labels[:,0])
    # regression.add_feature_fn(np.log)
    print(f'Current loss: {regression.LCE(f=regression(), y=regression.labels)}, Current accuracy: {regression.accuracy()}')
    regression.train(8000, batch = 400)
    ww = regression.w

    test_data = np.load('Task1_Testset500/data.npy')
    test_labels = np.load('Task1_Testset500/labels.npy')[:,0]
    test_data = add_neighborhood_feature(test_data, window_size=5)
    regression.w = np.ones_like(regression.w)
    regression.data = test_data
    regression.labels = test_labels
    # regression.add_feature_fn(np.log)
    print(f'Initial Testing loss: {regression.LCE(f=regression(), y=regression.labels)}, Initial Testing accuracy: {regression.accuracy()}')
    regression.w = ww
    print(f'Final Testing loss: {regression.LCE(f=regression(), y=regression.labels)}, Final Testing accuracy: {regression.accuracy()}')
    # test_regr = GD_Model(test_data, regression.w, test_labels)
    # print(f'Testing loss: {test_regr.LCE(f=test_regr(), y=test_regr.labels)}, Current accuracy: {test_regr.accuracy()}')
    
