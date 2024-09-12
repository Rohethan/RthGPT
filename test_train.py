import tensorflow as tf
import RthGPT

model = RthGPT.build_model(context_length=2048,
                           embedding_size=128,
                           vocab_size=16384,
                           n_attention_blocks=16,
                           attention_heads=4,
                           after_attention_dense_ratio=4)
model.summary()

dataset = RthGPT.HuggingfaceWikipediaFr()

print("model compiling start")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("model compiled")
print("model fit start")
generator = dataset.tensorflow_dataset_generator()
generator = generator.batch(4).prefetch(2)
model.fit(dataset.tensorflow_dataset_generator(), batch_size=4, epochs=1)