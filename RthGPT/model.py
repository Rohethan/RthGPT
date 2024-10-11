import tensorflow as tf
from tensorflow.keras import layers as L

class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, model_dim):
        super(PositionalEncodingLayer, self).__init__()
        self.positional_encoding = self._get_positional_encoding(max_seq_len, model_dim)

    def _get_positional_encoding(self, max_seq_len, model_dim):
        pos = tf.range(max_seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(model_dim, dtype=tf.float32)[tf.newaxis, :]

        angle_rads = pos / tf.pow(10000, (2 * (i // 2)) / model_dim)

        # apply sin to even indices in the array; 2i
        sines = tf.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]


class EmbeddingSelector(L.Layer):
    def call(self, embeddings, prediction_index):
        batch_size = tf.shape(embeddings)[0]
        prediction_index_squeezed = tf.squeeze(prediction_index, axis=-1)  # [batch_size]
        index_range = tf.range(batch_size)  # [batch_size]
        indices = tf.stack([index_range, prediction_index_squeezed], axis=1)  # [batch_size, 2]

        predicted_embed = tf.gather_nd(embeddings, indices)  # [batch_size, embedding_size]
        return predicted_embed

class ProbabilitySelector(L.Layer):
    def __init__(self, embedding_layer, **kwargs):
        super(ProbabilitySelector, self).__init__(**kwargs)
        self.embedding_layer = embedding_layer

    def build(self, input_shape):
        self.embeddings = tf.reshape(self.embedding_layer.embeddings, (-1, self.embedding_layer.embeddings.shape[0], self.embedding_layer.embeddings.shape[1]))


    def call(self, inputs, **kwargs):
        return tf.linalg.matmul(inputs, self.embeddings, transpose_b=True)

def build_model(context_length:int, embedding_size:int, vocab_size:int, n_recurrent_blocks:int,recurrent_block_size:int):
    # Preparing the input to enter the transformer blocks
    token_input_layer = L.Input(shape=(context_length,))
    embedding_layer = L.Embedding(vocab_size, embedding_size)

    input_embeddings = embedding_layer(token_input_layer)
    embeddings = input_embeddings

    # passing through Recurrent block
    for i in range(n_recurrent_blocks):
        recurrent_layer = L.LSTM(recurrent_block_size, return_sequences=True)
        dense_layer = L.Dense(embedding_size)
        activation = L.Activation('relu')

        rnn_out = dense_layer(embeddings)
        dense_out = dense_layer(rnn_out)
        act_out = activation(dense_out)

        embeddings = act_out

    #token_selection_layer

    first_simplernn_layer = L.SimpleRNN(context_length)
    identification_dense_layer = L.Dense(vocab_size)

    tokens_summarization = first_simplernn_layer(embeddings)
    summarization_identification = identification_dense_layer(tokens_summarization)


    token_probability = L.Softmax()(summarization_identification)

    model = tf.keras.Model(inputs=token_input_layer, outputs=token_probability)
    return model

if __name__ == '__main__':
    model = build_model(2048, 256, 1024, 16, 384)
    model.summary()
    import numpy as np
    test_ctx = np.random.randint(1024, size=[1,2048])
    print("feeding random test data")
    model.predict(test_ctx)
