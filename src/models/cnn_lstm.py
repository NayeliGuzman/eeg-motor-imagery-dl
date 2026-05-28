from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, LSTM, BatchNormalization, MaxPooling2D, Reshape
from tensorflow.keras.utils import to_categorical

class CNN_LSTM(Sequential):
  def __init__(self, input_shape=(250,1,22)):
    super(CNN_LSTM, self).__init__()

    # Conv. block 1
    self.add(Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=(250,1,22)))
    self.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    self.add(BatchNormalization())
    self.add(Dropout(0.5))

    # Conv. block 2
    self.add(Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu'))
    self.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    self.add(BatchNormalization())
    self.add(Dropout(0.5))

    # Conv. block 3
    self.add(Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu'))
    self.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    self.add(BatchNormalization())
    self.add(Dropout(0.5))

    # Conv. block 4
    self.add(Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu'))
    self.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    self.add(BatchNormalization())
    self.add(Dropout(0.5))

    # FC+LSTM layers
    self.add(Flatten()) # Adding a flattening operation to the output of CNN block
    self.add(Dense((100))) # FC layer with 100 units
    self.add(Reshape((100,1))) # Reshape my output of FC layer so that it's compatible
    self.add(LSTM(10, dropout=0.6, recurrent_dropout=0.1, input_shape=(100,1), return_sequences=False))


    # Output layer with Softmax activation
    self.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation





