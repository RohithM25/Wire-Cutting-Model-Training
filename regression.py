import numpy as np

np.random.seed(42)

def SGD(w, x, y, alpha):
    # compute estimated output y'
    y_prime = sigmoid(float(np.dot(x,w)))
    # compute Loss of y' compared to y where y is the actual/correct output
    cur_loss = loss(y, y_prime) #+ (1e-10 * np.sum(np.abs(w))) # l2 reg can be changed for the model
    # SGD with new weights = old weights - alpha (sig(w * x) - y)xi) 
    w -= alpha * (y_prime - y) * x
    return cur_loss

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def avg_loss(data, w, labels):
    losses = [loss(sigmoid(np.dot(x, w)), y) for x, y in zip(data, labels[:, 0])]
    return np.average(losses)

def loss(actual_y, y_prime):
    epsilon = 1e-15
    return -actual_y * np.log(y_prime+epsilon) - (1-actual_y) * np.log(1-y_prime+epsilon)

def add_polynomial_features(X, degree):
    """Add polynomial features to the input data."""
    features = [X]
    for d in range(2, degree + 1):
        features.append(X**d)
    return np.column_stack(features)

def accuracy(data, labels, weights):
    outs = np.dot(data, weights)
    binary = (outs >= 0.5).astype(int)

    samples = data.shape[0]
    correct = sum(binary == labels[:,0])
    return correct/samples

def feed_forward(data,labels, num_iters = 10000):
    data = np.column_stack([data, np.ones(data.shape[0])]) # Adding in a bias term
    samples,features = data.shape
    w = np.random.rand(features)*2 - np.ones(shape=features) #[0,1] -> [-100,100]
    for i in range(num_iters):
        index = np.random.randint(0, 1000)
        x = data[index]
        y = labels[index][0]
        loss = SGD(w, x, y, alpha=0.01)
        
    print(f'Average Loss: {avg_loss(data,w,labels)}, Accuracy on testing set: {accuracy(data, labels, w)}')
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



