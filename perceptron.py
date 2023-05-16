import numpy as np


class Perceptron:
    def __init__(self, alpha=0.1, bias=-0.1, max_iterations=1000):
        self.alpha = alpha
        self.bias = bias
        self.max_iterations = max_iterations
        self.weights = None

    def train(self, data, labels):
        self.weights = np.zeros(data.shape[1] + 1)
        self.weights[0] = self.bias

        data_with_bias = np.hstack((np.full((data.shape[0], 1), 1), data))

        for _ in range(self.max_iterations):
            for i in range(len(data)):
                weighted_sum = np.dot(data_with_bias[i], self.weights)
                output = 1 if weighted_sum >= 0 else 0
                error = labels[i] - output
                self.weights += self.alpha * error * data_with_bias[i]

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if weighted_sum >= 0 else 0

    def validate(self, data, labels):
        correct_predictions = 0
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if prediction == labels[i]:
                correct_predictions += 1

        accuracy = correct_predictions / len(data)
        return accuracy


perceptron = Perceptron()

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

perceptron.train(data, labels)

accuracy = perceptron.validate(data, labels)
print("Accuracy on training data: ", accuracy)
