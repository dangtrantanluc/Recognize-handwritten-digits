import tensorflow as tf

def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(-1, 784) / 255.0
    test_images = test_images.reshape(-1, 784) / 255.0

    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    training_data = list(zip(train_images, train_labels))
    test_data = list(zip(test_images, test_labels))

    return training_data, test_data
