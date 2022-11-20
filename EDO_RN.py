from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop

#Definimos nuestro dominio dotando los valores de las cotas:
minv = -1
maxv = 1

class ODE(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_t = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_t]

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size,1), minv, maxv)

        with tf.GradientTape() as Tape:
            with tf.GradientTape() as Tape2:
                Tape2.watch(x)
                y_pos = self(x, training = True)
            dy = Tape2.gradient(y_pos, x)
            x0 = tf.zeros((batch_size, 1))
            y0 = self(x0, training = True)
            eqd = dy + 2*x*y_pos
            ic = y0 - 1
            loss = keras.losses.mean_squared_error(0., eqd) + keras.losses.mean_squared_error(0., ic)

        #Gradientes
        grads = Tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
        self.loss_t.update_state(loss)
        return {"loss": self.loss_t.result()}

model = ODE()

model.add(Dense(20, activation='tanh', input_shape=(1,)))
model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(optimizer=Adam(), metrics=['loss'])

x = tf.linspace(-1,1,1000)

history = model.fit(x, epochs = 100, verbose = 1)

x_test = tf.linspace(-1, 1, 1000)
a = model.predict(x_test)
plt.plot(x_test, a)
plt.plot(x_test, np.exp(-x*x))
plt.show()
