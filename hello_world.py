import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    print(tf.__version__)
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[1])])
    print(model.output_shape)
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    xs = np.array(list(range(0, 20)), dtype=float)
    ys = np.array([2 * x - 1 for x in range(0, 20)], dtype=float)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    xs_new = x_scaler.fit_transform(np.array(xs).reshape(-1, 1))
    ys_new = y_scaler.fit_transform(np.array(ys).reshape(-1, 1))

    model.fit(xs_new, ys_new, epochs=500, verbose=1)
    test_x = np.array([21]).reshape(-1, 1)
    x1 = xs / 20
    y1 = ys / 37
    # model.fit(xs, ys, epochs=2000)
    # print(y_scaler.inverse_transform(model.predict(x_scaler.transform(test_x))))

    print(model.predict([21]))
