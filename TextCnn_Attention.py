import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, MaxPooling2D, Conv2D, Flatten, Concatenate, Dropout, concatenate, TimeDistributed
from tensorflow.keras.constraints import max_norm
import tensorflow.keras as keras


class attention_cnn(Model):
    def __init__(self, vocab_size=10000, embedding_dim=100, max_len=64):
        super(attention_cnn, self).__init__()
        self.max_len = max_len
        self.embedding = Embedding(vocab_size+1, embedding_dim, input_length=max_len)
        self.W3 = TimeDistributed(Dense(max_len, activation='tanh'))
        self.f1 = Dense(1)  # 令embedding_dim维度 = 1
        self.conv1 = Conv2D(filters=100, kernel_size=(3, embedding_dim), padding='valid', kernel_constraint=max_norm(3, [0, 1, 2]))  # 1-D卷积
        self.conv2 = Conv2D(filters=100, kernel_size=(4, embedding_dim), padding='valid', kernel_constraint=max_norm(3, [0, 1, 2]))
        self.conv3 = Conv2D(filters=100, kernel_size=(5, embedding_dim), padding='valid', kernel_constraint=max_norm(3, [0, 1, 2]))
        self.pool1 = MaxPooling2D(pool_size=(max_len - 3 + 1, 1))
        self.pool2 = MaxPooling2D(pool_size=(max_len - 4 + 1, 1))
        self.pool3 = MaxPooling2D(pool_size=(max_len - 5 + 1, 1))
        self.drop = Dropout(rate=0.5)
        self.flatten = Flatten()
        self.dense = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x_input = inputs
        x = self.embedding(x_input)

        # ======================== attention =========================
        sentence_embedding = tf.reduce_mean(x, 1, keepdims=True)
        con_vector1 = concatenate([x, sentence_embedding], axis=1)
        score = self.f1(self.W3(con_vector1))  # (50, 65, 1)
        # ============================================================

        weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(weights * con_vector1, axis=1, keepdims=True)
        # ============================================================

        # ---------------- concatenate(or Concatenate) ----------------
        # con_vector1 = concatenate([x, context_vector], axis=1)
        con_vector2 = Concatenate(axis=1)([x, context_vector])
        # -------------------------------------------------------------

        con_vector2 = con_vector2[..., tf.newaxis]      # channel_last
        x = self.conv1(con_vector2)
        p1 = self.pool1(x)
        x = self.conv2(con_vector2)
        p2 = self.pool2(x)
        x = self.conv3(con_vector2)
        p3 = self.pool3(x)

        # ---------------- concatenate(or Concatenate) ----------------
        con_vector3 = concatenate([p1, p2, p3], axis=2)
        # con_vector3 = Concatenate(axis=2)([p1, p2, p3])
        # -------------------------------------------------------------

        x = self.drop(con_vector3)
        x = self.flatten(x)
        output = self.dense(x)

        return output


def load_imdb(num_words):
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
    return x_train, y_train, x_test, y_test


def pad_sentence(x_train, x_test, max_len=64):
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, value=0, padding='post', maxlen=max_len)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, value=0, padding='post', maxlen=max_len)
    return x_train, x_test, max_len


MAX_LEN = 64
batchsz = 64

x_train, y_train, x_test, y_test = load_imdb(10000)
x_train, x_test, max_len = pad_sentence(x_train, x_test, max_len=MAX_LEN)

model = attention_cnn(max_len=MAX_LEN)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=batchsz, validation_data=(x_test, y_test), verbose=1)
model.summary()
