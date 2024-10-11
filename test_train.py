import tensorflow as tf
import RthGPT

model = RthGPT.build_model(context_length=2304,
                           embedding_size=96,
                           vocab_size=4096,
                           n_recurrent_blocks=16,
                           recurrent_block_size=128)
model.summary()
input("Press enter to continue...")
dataset = RthGPT.PetitNicolasDataset()

print("model compiling start")
callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True),

]
optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("model compiled")
print("model fit start")
generator = dataset.tensorflow_dataset_generator()
batch_size = 128
print("estimated to have", 38857/batch_size, " iters per epoch")
generator = generator.batch(batch_size).prefetch(2)
model.fit(dataset.tensorflow_dataset_generator(), batch_size=batch_size, epochs=1000, steps_per_epoch=200 ,callbacks=callbacks) #supposed to have 38857 elems per epoch, batched
model.save_weights('./weights/model')