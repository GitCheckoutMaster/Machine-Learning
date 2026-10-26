from sklearn.datasets import fetch_openml # type: ignore
import numpy as np # type: ignore
import SGD_Classifier
import matplotlib.pyplot as plt # type: ignore

def plot_image(image, label):
	plt.imshow(image.reshape(28, 28), cmap='gray')
	plt.title(f"Label: {label}")
	plt.axis('off')
	plt.show()

def load_mnist_data():
	mnist = fetch_openml("mnist_784", as_frame=False, parser="liac-arff")
	X, y = mnist.data, mnist.target
	return X, y

def preprocess_data(X, y):
	X_train, X_test, y_train, y_test = X[:10000], X[10000:10060], y[:10000], y[10000:10060]
	# plot_image(X_test[0], y_test[0])
	# plot_image(X_test[1], y_test[1])
	# plot_image(X_test[2], y_test[2])
	return X_train.astype(np.float64) / 255.0, X_test.astype(np.float64) / 255.0, y_train.astype(int), y_test.astype(int)

if __name__ == "__main__":
	X, y = load_mnist_data()
	X_train, X_test, y_train, y_test = preprocess_data(X, y)
	y_train = (y_train == 5).astype(int)
	y_test = (y_test == 5).astype(int)

	classifier = SGD_Classifier.SGD_Classifier(learning_rate=0.01, n_iterations=500, batch_size=5)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	print(f"Predictions: {y_pred.round()}")
	print(f"True Labels: {y_test}")
	accuracy = np.mean(y_pred.round() == y_test)
	print(f"Test Accuracy: {accuracy * 100:.2f}%")

