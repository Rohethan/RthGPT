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

def build_model(context_length:int, embedding_size:int, vocab_size:int, n_attention_blocks:int, attention_heads:int, after_attention_dense_ratio:int, dropout_rate=0.05):
    # Preparing the input to enter the transformer blocks
    token_input_layer = L.Input(shape=(context_length,))
    prediction_index = L.Input(shape=(1,), dtype=tf.int32)
    embedding_layer = L.Embedding(vocab_size, embedding_size)
    positional_encoding_layer = PositionalEncodingLayer(max_seq_len=context_length, model_dim=embedding_size)


    input_embeddings = embedding_layer(token_input_layer)
    positional_encodings = positional_encoding_layer(input_embeddings)
    embeddings = input_embeddings + positional_encodings

    # passing through transformer block
    for i in range(n_attention_blocks):
        attention_layer = L.MultiHeadAttention(attention_heads, embedding_size, name='Block_'+str(i)+'_attention_layer', dropout=dropout_rate)
        ratioed_dense_layer = L.Dense(embedding_size * after_attention_dense_ratio, name='Block_'+str(i)+'_dense_layer')
        final_block_dense_layer = L.Dense(embedding_size, name='Block_'+str(i)+'_final_dense_layer')
        dropout_layer = L.Dropout(dropout_rate, name='Block_' + str(i) + '_dropout')

        attention = attention_layer(embeddings, embeddings, use_causal_mask=True)
        ratioed_dense = ratioed_dense_layer(attention)
        ratioed_dense = dropout_layer(ratioed_dense)
        final_block_dense = final_block_dense_layer(ratioed_dense)
        final_block_dense = dropout_layer(final_block_dense)



        embeddings = final_block_dense

    # We select the vector of shape [batch_size, embedding_size] in embeddings of shape [batch_size, context_length, embedding_size] based on value in prediction_index
    selected_embedding_layer = EmbeddingSelector()
    predicted_embed = selected_embedding_layer(embeddings, prediction_index)

    prob_selector_layer = ProbabilitySelector(embedding_layer)
    token_probability = prob_selector_layer(predicted_embed)
    token_probability = L.Softmax()(token_probability)

    model = tf.keras.Model(inputs=[token_input_layer, prediction_index], outputs=[token_probability])
    return model

if __name__ == '__main__':
    model = build_model(2048, 256, 1024, 4, 2, 4)
    model.summary()
    import numpy as np
    test_ctx = np.random.randint(1024, size=[16, 2048])
    test_pred_indx = np.random.randint(2048, size=[16, 1])
    print("feeding random test data")
    model.predict([test_ctx, test_pred_indx])
