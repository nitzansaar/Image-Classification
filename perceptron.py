import numpy as np


def perceptron_learning_algorithm(data, labels, alpha=0.1, bias=-0.1, max_iterations=1000):
    weights = np.zeros(data.shape[1] + 1)
    weights[0] = bias

    data_with_bias = np.hstack((np.full((data.shape[0], 1), 1), data))

    for _ in range(max_iterations):
        for i in range(len(data)):
            weighted_sum = np.dot(data_with_bias[i], weights)
            output = 1 if weighted_sum >= 0 else 0
            error = labels[i] - output
            weights += alpha * error * data_with_bias[i]

    return weights


data = np.array([
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1],
    [0, 1, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0],
])

labels = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

final_weights = perceptron_learning_algorithm(data, labels)
print("Final weights:", final_weights)
