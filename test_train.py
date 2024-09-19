import tensorflow as tf
import RthGPT

model = RthGPT.build_model(context_length=2048,
                           embedding_size=128,
                           vocab_size=16384,
                           n_attention_blocks=16,
                           attention_heads=2,
                           after_attention_dense_ratio=2)
model.summary()

dataset = RthGPT.HuggingfaceWikipediaFr()

print("model compiling start")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("model compiled")
print("model fit start")
generator = dataset.tensorflow_dataset_generator()
generator = generator.batch(2).prefetch(2)
model.fit(dataset.tensorflow_dataset_generator(), batch_size=2, epochs=1)