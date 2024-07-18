TOKENIZER_NAME = "Celestializer2048"
CONTEXT_SIZE = 2048
from datasets import load_dataset

import tokenizers
import tensorflow as tf

# Dataset generator and processing
# Assign speaker IDs
SYSTEM_ID = 1
USER_ID = 2
MODEL_ID = 3
PADDING_ID = 0

tokenizer = tokenizers.Tokenizer.from_file("Celestializer2048.json")
END_OF_TEXT_TOKEN = 1


def lenreg(list, size=1024, padding_idx=0):
 return (list + [padding_idx for _ in range(size)])[:size]


def create_speaker_context(seq_length, speaker_id, context_size, padding_idx=0):
 return [speaker_id if i < seq_length and i != padding_idx else padding_idx for i in range(context_size)]


def generator(context_size=CONTEXT_SIZE):  # Default context window size to 2048
 print("[GENERATOR] Loading dataset and tokenizer")
 dataset = load_dataset("Open-Orca/OpenOrca")
 print("[GENERATOR] dataset loaded")

 print("[GENERATOR] tokenizer loaded")
 tensorize = tf.convert_to_tensor

 for features in dataset["train"]:
  sp = tokenizer.encode(features["system_prompt"]).ids
  sp.append(END_OF_TEXT_TOKEN)
  up = tokenizer.encode(features["question"]).ids
  up.append(END_OF_TEXT_TOKEN)
  ma = tokenizer.encode(features["response"]).ids
  ma.append(END_OF_TEXT_TOKEN)
  # print(features["system_prompt"], features["question"], features["response"])
  token_context = sp + up
  speaker_context = [SYSTEM_ID] * len(sp) + [USER_ID] * len(up)

  for ma_token in ma:
   regulated_token_context = tensorize(lenreg(token_context, size=context_size), dtype=tf.int64)
   regulated_speaker_context = tensorize(lenreg(speaker_context, size=context_size, padding_idx=0), dtype=tf.int64)
   yield (regulated_token_context, regulated_speaker_context), ma_token
   token_context.append(ma_token)
   speaker_context.append(MODEL_ID)


output_signature = (
 (
  tf.TensorSpec(shape=(CONTEXT_SIZE), dtype=tf.int64),
  tf.TensorSpec(shape=(CONTEXT_SIZE), dtype=tf.int64)
 ),
 tf.TensorSpec(shape=(), dtype=tf.int64)
)
dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature, args=(CONTEXT_SIZE,))

import tensorflow as tf
from tensorflow.keras import layers as L


def build_llm_model(context_window, vocab_size, embedding_size, num_attention_blocks, num_heads, dense_attention_units,
                    dropout_rate=0.15, normalization_epsilon=1e-3):
 context_input_tokens = tf.keras.Input(shape=(context_window,), dtype=tf.int32)
 speaker_input_tokens = tf.keras.Input(shape=(context_window,), dtype=tf.int32)

 context_embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size)

 embedded_contexts = context_embedding_layer(context_input_tokens)
 embedded_speakers = tf.keras.layers.Embedding(4, embedding_size)(speaker_input_tokens)
 embeds = embedded_contexts + embedded_speakers

 for i in range(num_attention_blocks):
  attention = L.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_size)(embeds, embeds)
  attention = L.LayerNormalization(epsilon=normalization_epsilon)(attention)
  attention = L.Dropout(dropout_rate)(attention)

  dense1 = L.Dense(dense_attention_units, activation='sigmoid')(attention)
  dense1 = L.Dropout(dropout_rate)(dense1)
  dense1 = L.LayerNormalization(epsilon=normalization_epsilon)(dense1)

  dense2 = L.Dense(embedding_size, activation='sigmoid')(dense1)
  dense2 = L.Dropout(dropout_rate)(dense2)
  dense2 = L.LayerNormalization(epsilon=normalization_epsilon)(dense2)

  embeds = embeds + dense2

 final_embedding = embeds[:, -1]
 embeddings = tf.expand_dims(context_embedding_layer.embeddings, axis=0)
 embeddings = tf.tile(embeddings, [tf.shape(final_embedding)[0], 1, 1])

 final_output = tf.keras.layers.Dot(axes=-1)([final_embedding, embeddings])
 final_output = tf.keras.layers.Softmax()(final_output)

 model = tf.keras.Model(inputs=[context_input_tokens, speaker_input_tokens], outputs=final_output)
 optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
 loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
 model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
 return model


from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

#Small parameters numbers for testing purposes
model = build_llm_model(CONTEXT_SIZE,
                        tokenizer.get_vocab_size(),
                        10, # embedding size
                        2, # Number of blocks
                        2, # attention heads
                        20)# Dense units

model.summary()

from tensorflow.keras.callbacks import *
# TensorBoard callback
tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)

# Terminate training on nan. (Don't spend additional compute ressources)
nan_stop = TerminateOnNaN()

# ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath="./checkpoints/" + "ckpt_{epoch:03d}-{loss:.3f}.ckpt",
    save_weights_only=True,
    save_freq=4096,# Save the model every 2 epochs
    verbose=1)

dataset = dataset.batch(4).prefetch(2)
with tf.device('/CPU:0'):
    model.fit(dataset, epochs=100, steps_per_epoch=2048, verbose=1, batch_size=4, callbacks=[tensorboard_callback, nan_stop, checkpoint_callback])
    model.save("test")