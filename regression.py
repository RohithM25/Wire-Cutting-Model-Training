import numpy as np
from numba import jit
import warnings
from numba import NumbaDeprecationWarning, NumbaWarning

# Suppress all Numba warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)

np.random.seed(42)

# @jit(nopython=False)
def SGD(w, x, y, alpha):
    # compute estimated output y'
    y_prime = sigmoid(float(np.dot(x,w)))
    # compute Loss of y' compared to y where y is the actual/correct output
    cur_loss = loss(y, y_prime) #+ (1e-10 * np.sum(np.abs(w))) # l2 reg can be changed for the model
    # SGD with new weights = old weights - alpha (sig(w * x) - y)xi) 
    w -= alpha * (y_prime - y) * x
    return cur_loss

# @jit(nopython=False)
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

# @jit(nopython=False)
def avg_loss(data, w, labels):
    losses = [loss(sigmoid(np.dot(x, w)), y) for x, y in zip(data, labels[:, 0])]
    return np.average(losses)

# @jit(nopython=False)
def loss(actual_y, y_prime):
    epsilon = 1e-15
    return -actual_y * np.log(y_prime+epsilon) - (1-actual_y) * np.log(1-y_prime+epsilon)

def add_polynomial_features(X, degree):
    """Add polynomial features to the input data."""
    features = [X]
    for d in range(2, degree + 1):
        features.append(X**d)
    return np.concatenate(features, axis=1)

# @jit(nopython=False)
def accuracy(data, labels, weights):
    outs = sigmoid(np.dot(data, weights))
    binary = np.int64((outs >= 0.5))
    samples = data.shape[0]
    correct = np.sum(np.equal(binary,labels[:,0]))
    return correct/samples

# @jit(nopython=False)
def feed_forward(data,labels, num_iters = 10000):
    # Adding in a bias term
    bias = np.ones((data.shape[0], 1))
    data = np.concatenate((data, bias), axis=1)
    print(data.shape)
    samples,features = data.shape
    w = np.random.rand(features)*2 - np.ones(shape=features) #[0,1] -> [-1,1]
    for i in range(num_iters):
        index = np.random.randint(0, 1000)
        x = data[index]
        y = labels[index][0]
        loss = SGD(w, x, y, alpha=0.01)
        
    print(f'Average Loss: {avg_loss(data,w,labels)}, Accuracy on training data: {accuracy(data, labels, w)}')
    return w

def compute_neighborhood_mean2(image, window_size):
    image = image.reshape((20,20))

    padded_image = np.pad(image, pad_width=window_size//2, mode='constant')

    windowed_view = np.lib.stride_tricks.sliding_window_view(padded_image, (window_size, window_size))

    neighborhood_means = np.mean(windowed_view, axis=(-2, -1))

    return neighborhood_means

def add_neighborhood_feature(data, window_size=3):
    new_data = [] # store all the new data points in here
    for i in range(data.shape[0]): #i.e for each data point
        means = compute_neighborhood_mean2(data[i], window_size)
        new_data.append(np.concatenate((data[i], means.flatten())))
    return np.array(new_data)

if __name__ == '__main__':
    training_data = np.load('1k_data/data.npy')
    training_labels = np.load('1k_data/labels.npy')
    testing_data = np.load('Task1_Testset500/data.npy')
    testing_labels = np.load('Task1_Testset500/labels.npy')

    data = add_neighborhood_feature(training_data)
    data = add_polynomial_features(data, degree=4)
    weights = feed_forward(data, training_labels, num_iters=10000000)

    tdata = add_neighborhood_feature(testing_data)
    tdata = add_polynomial_features(tdata, degree=4)
    tdata = np.concatenate((tdata, np.ones((tdata.shape[0], 1))), axis=1)
    print(f'Testing accuracy:{accuracy(tdata, testing_labels, weights)}. Testing loss:{avg_loss(tdata, weights, testing_labels)}')

