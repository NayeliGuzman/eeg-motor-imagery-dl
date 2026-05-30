import keras
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Dropout, BatchNormalization,
    Reshape, Dense, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, DepthwiseConv1D,
    add, Dropout
)

class FeedForward(keras.layers.Layer):
    def __init__(self, d_model, ff_dim=256, dropout=0.1):
        super(FeedForward, self).__init__()
        self.dense1    = Dense(ff_dim, activation='gelu')
        self.dense2    = Dense(d_model)
        self.dropout   = Dropout(dropout)
        self.layernorm = LayerNormalization()

    def call(self, x, training=False):
        residual = x
        x = self.layernorm(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return residual + 0.5 * x

class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads, #128/4 = 32
            dropout=dropout
        )
        self.layernorm = LayerNormalization()
        self.dropout   = Dropout(dropout)

    def call(self, x, training=False):
        residual = x
        x = self.layernorm(x)
        x = self.attention(x, x, training=training) #self attention
        x = self.dropout(x, training=training)
        return residual + x

class ConformerConvModule(keras.layers.Layer):
    def __init__(self, d_model, kernel_size=31, dropout=0.1): # kernel size = 31 as in original paper
        super(ConformerConvModule, self).__init__()
        self.layernorm    = LayerNormalization()
        self.pointwise1   = Dense(d_model * 2) # expand
        self.glu          = Dense(d_model) # gate
        self.depthwise    = DepthwiseConv1D(kernel_size=kernel_size, padding='same')
        self.batchnorm    = BatchNormalization()
        self.pointwise2   = Dense(d_model) # project back
        self.dropout      = Dropout(dropout)

    def call(self, x, training=False):
        residual = x
        x = self.layernorm(x)
        x = self.pointwise1(x) # (batch, seq, d_model*2)
        x, gate = keras.ops.split(x, 2, axis=-1) # split into two (batch, seq, d_model)
        x = x * keras.activations.sigmoid(gate) # GLU
        x = self.depthwise(x, training=training)
        x = self.batchnorm(x, training=training)
        x = keras.activations.gelu(x)
        x = self.pointwise2(x)
        x = self.dropout(x, training=training)
        return residual + x

class ConformerBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim=256, kernel_size=31, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.ff1       = FeedForward(d_model, ff_dim, dropout)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv      = ConformerConvModule(d_model, kernel_size, dropout)
        self.ff2       = FeedForward(d_model, ff_dim, dropout)
        self.layernorm = LayerNormalization()

    def call(self, x, training=False):
        x = self.ff1(x, training=training)
        x = self.attention(x, training=training)
        x = self.conv(x, training=training)
        x = self.ff2(x, training=training)
        x = self.layernorm(x)
        return x

class CNN_CONFORMER(keras.Model):
    def __init__(self, input_shape=(250, 1, 22)):
        super(CNN_CONFORMER, self).__init__()

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

        self.reshape    = Reshape((4, 200))
        self.projection = Dense(128)
        self.conformer1 = ConformerBlock(d_model=128, num_heads=4)
        self.conformer2 = ConformerBlock(d_model=128, num_heads=4)
        self.gap        = GlobalAveragePooling1D()
        self.output_layer = Dense(4, activation='softmax')

    def call(self, x, training=False):
        x = self.b1(x, training=training)
        x = add([self.b2(x, training=training), self.sc2(x)])
        x = add([self.b3(x, training=training), self.sc3(x)])
        x = add([self.b4(x, training=training), self.sc4(x)])
        x = self.reshape(x)
        x = self.projection(x)
        x = self.conformer1(x, training=training)
        x = self.conformer2(x, training=training)
        x = self.gap(x)
        return self.output_layer(x)

if __name__ == "__main__":
    model = CNN_CONFORMER()
    model.build(input_shape=(None, 250, 1, 22))
    model.summary()
