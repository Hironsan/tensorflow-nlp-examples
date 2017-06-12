import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split


def main():
    # Load Iris dataset
    iris = datasets.load_iris()
    # Extract features
    X = iris.data
    # Extract class label
    y = iris.target

    # Split data into training data and test data
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)

    # Specify that all features have real-value data
    features = [tf.contrib.layers.real_valued_column('', dimension=4)]

    # Build linear classifier
    classifier = tf.contrib.learn.LinearClassifier(feature_columns=features,
                                                   n_classes=3,
                                                   optimizer=tf.train.AdamOptimizer(learning_rate=0.01))

    # Define the train inputs
    def get_train_inputs():
        x = tf.constant(train_x)
        y = tf.constant(train_y)
        return x, y

    # Fit model
    classifier.fit(input_fn=get_train_inputs, steps=2000)
    accuracy_score = classifier.evaluate(input_fn=get_train_inputs, steps=1)['accuracy']
    print('\nTest Accuracy: {0:f}\n'.format(accuracy_score))  # 0.990476

    # Define the test inputs
    def get_test_inputs():
        x = tf.constant(test_x)
        y = tf.constant(test_y)
        return x, y

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)['accuracy']
    print('\nTest Accuracy: {0:f}\n'.format(accuracy_score))  # 0.977778

if __name__ == '__main__':
    main()
