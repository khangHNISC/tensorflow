import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.4:
            self.model.stop_training = True


if __name__ == '__main__':
    print(tf.__version__)
    callback = myCallback()
    fashion_mnist = tf.keras.datasets.fashion_mnist
    # 60000 vs 10000
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # examine data
    index = 42
    np.set_printoptions(linewidth=320)
    print(f'Label: {train_labels[index]}')
    print(f'Image pixel array: {train_images[index]}')
    plt.imshow(train_images[index], cmap='Greys')
    # plt.show()  # https://stackoverflow.com/questions/42812230/why-doesnt-plt-imshow-display-the-image

    # normalize data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # train
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ])

    model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, callbacks=[callback])

    # loss = 0.29xx = not great

    model.evaluate(test_images, test_labels)

    # loss = 0.345
    classifications = model.predict(test_images)

    print(classifications[0])
    print(test_labels[0])
