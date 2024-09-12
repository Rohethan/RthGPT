import tensorflow as tf
from tensorflow.keras import layers as L



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

def build_model(context_length:int, embedding_size:int, vocab_size:int, n_attention_blocks:int, attention_heads:int, after_attention_dense_ratio:int):
    # Preparing the input to enter the transformer blocks
    token_input_layer = L.Input(shape=(context_length,))
    prediction_index = L.Input(shape=(1,), dtype=tf.int32)
    embedding_layer = L.Embedding(vocab_size, embedding_size)
    positional_encodings = tf.Variable(tf.random.uniform(shape=(context_length, embedding_size), minval=0, maxval=1))

    input_embeddings = embedding_layer(token_input_layer)
    embeddings = input_embeddings + positional_encodings

    # passing through transformer block
    for i in range(n_attention_blocks):
        attention_layer = L.MultiHeadAttention(attention_heads,embedding_size, name='Block '+str(i)+' attention_layer')
        ratioed_dense_layer = L.Dense(embedding_size * after_attention_dense_ratio, name='Block '+str(i)+' dense_layer')
        final_block_dense_layer = L.Dense(embedding_size, name='Block '+str(i)+' final_dense_layer')

        attention = attention_layer(embeddings, embeddings, use_causal_mask=True)
        ratioed_dense = ratioed_dense_layer(attention)
        final_block_dense = final_block_dense_layer(ratioed_dense)

        embeddings = final_block_dense

        # We select the vector of shape [batch_size, embedding_size] in embeddings of shape [batch_size, context_length, embedding_size] based on value in prediction_index
        selected_embedding_layer = EmbeddingSelector()
        predicted_embed = selected_embedding_layer(embeddings, prediction_index)

        prob_selector_layer = ProbabilitySelector(embedding_layer)
        token_probability = prob_selector_layer(predicted_embed)

    model = tf.keras.Model(inputs=[token_input_layer], outputs=[token_probability])
    return model

if __name__ == '__main__':
    model = build_model(2048, 256, 4096, 32, 4, 4)
    model.summary()
