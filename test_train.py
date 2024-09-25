import tensorflow as tf
import RthGPT

model = RthGPT.build_model(context_length=2304,
                           embedding_size=64,
                           vocab_size=4096,
                           n_attention_blocks=16,
                           attention_heads=2,
                           after_attention_dense_ratio=4)
model.summary()

dataset = RthGPT.PetitNicolasDataset()

print("model compiling start")
callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True),

]
optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'], callbacks=callbacks)
print("model compiled")
print("model fit start")
generator = dataset.tensorflow_dataset_generator()
batch_size = 16
print("estimated to have", 38857/batch_size, " iters per epoch")
generator = generator.repeat().batch(batch_size).prefetch(2)
model.fit(dataset.tensorflow_dataset_generator(), batch_size=batch_size, epochs=1000) #supposed to have 38857 elems per epoch, batched
model.save_weights('./weights/model')