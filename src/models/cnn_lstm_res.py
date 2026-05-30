import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Reshape, LSTM, add
from keras.models import Sequential

class CNN_LSTM_RES(keras.Model):
    def __init__(self, input_shape=(250, 1, 22)):
        super(CNN_LSTM_RES, self).__init__()

        # Conv 1
        self.b1 = Sequential([
            Conv2D(filters=25, kernel_size=(10, 1), padding='same', activation='elu', input_shape=input_shape),
            MaxPooling2D(pool_size=(3, 1), padding='same'),
            BatchNormalization(),
            Dropout(0.5)
        ])

        # Conv 2 
        self.b2 = Sequential([
            Conv2D(filters=50, kernel_size=(10, 1), padding='same', activation='elu'),
            MaxPooling2D(pool_size=(3, 1), padding='same'),
            BatchNormalization(),
            Dropout(0.5)
        ])
        self.sc2 = Sequential([Conv2D(50, (1, 1), padding='same'), MaxPooling2D((3, 1), padding='same')])

        # Conv 3 
        self.b3 = Sequential([
            Conv2D(filters=100, kernel_size=(10, 1), padding='same', activation='elu'),
            MaxPooling2D(pool_size=(3, 1), padding='same'),
            BatchNormalization(),
            Dropout(0.5)
        ])
        self.sc3 = Sequential([Conv2D(100, (1, 1), padding='same'), MaxPooling2D((3, 1), padding='same')])

        # Conv 4 
        self.b4 = Sequential([
            Conv2D(filters=200, kernel_size=(10, 1), padding='same', activation='elu'),
            MaxPooling2D(pool_size=(3, 1), padding='same'),
            BatchNormalization(),
            Dropout(0.5)
        ])
        self.sc4 = Sequential([Conv2D(200, (1, 1), padding='same'), MaxPooling2D((3, 1), padding='same')])

        # FC + LSTM + Output
        self.tail = Sequential([
            Flatten(),
            #Dense(100, 'gelu'),
            Dense(100),
            Reshape((100, 1)),
            LSTM(10, dropout=0.6, recurrent_dropout=0.1, return_sequences=False),
            Dense(4, activation='softmax')
        ])

    def call(self, x, training=False):
        x = self.b1(x, training=training)

        # Short residual passes
        x = add([self.b2(x, training=training), self.sc2(x)])
        x = add([self.b3(x, training=training), self.sc3(x)])
        x = add([self.b4(x, training=training), self.sc4(x)])

        return self.tail(x, training=training)

if __name__ == "__main__":
    model = CNN_LSTM_RES()
    model.build(input_shape=(None, 250, 1, 22))
    model.summary()
