import numpy as np

class SGD_Classifier:
	def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
		self.learning_rate = learning_rate
		self.n_iterations = n_iterations
		self.batch_size = batch_size

	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0.0

		for _ in range(self.n_iterations):
			for i in range(0, n_samples, self.batch_size):
				X_batch = X[i:i+self.batch_size]
				y_batch = y[i:i+self.batch_size]

				# calculate required parameters
				y_prediction = np.dot(X_batch, self.weights) + self.bias
				prediction_error = (y_prediction - y_batch)
				weight_gradient = (2 / self.batch_size) * np.dot(
					X_batch.T,
					prediction_error,
				)
				bias_gradient = (2 / self.batch_size) * np.sum(prediction_error)

				# update
				self.weights = self.weights - self.learning_rate * weight_gradient
				self.bias = self.bias - self.learning_rate * bias_gradient

			if _ % 100 == 0:
				res = np.dot(X_batch, self.weights) + self.bias - y_batch
				print(f"Iteration {_}: Loss = {np.mean(res ** 2):.6f}")

	def predict(self, X):
		y_prediction = np.dot(X, self.weights) + self.bias
		return y_prediction
