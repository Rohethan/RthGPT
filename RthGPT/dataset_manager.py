import glob

import tensorflow as tf
import numpy as np

import RthGPT


class PetitNicolasDataset():
    def __init__(self, ctx_lenght=2304):
        files = glob.glob('/home/rohethan/Projects/RthGPT/petit_nicolas/*.txt')
        txts = [open(file).read() for file in files]
        tokenizer = RthGPT.tokenizer.RthTokenizer("pn_4k.json")
        self.tokenized_texts = [tokenizer.tokenize(txt).ids for txt in txts]
        self.ctx_lenght = ctx_lenght

    def tokenized_data_generator(self):
        for element in self.tokenized_texts:
            increasing_context = [0]*self.ctx_lenght
            i = 0
            for token in element:

                ctx_arr = np.array(increasing_context).reshape(1, -1)
                guess_token = np.array(token).reshape(1, -1)

                yield ctx_arr, guess_token

                increasing_context[i] = token
                i += 1
    def tensorflow_dataset_generator(self):
        output_type = tf.int32, tf.int32
        output_shape = tf.TensorShape([None, self.ctx_lenght]), tf.TensorShape([None, 1])
        tfds = tf.data.Dataset.from_generator(self.tokenized_data_generator, output_types=output_type, output_shapes=output_shape)
        return tfds

class HuggingfaceWikipediaFr():
    def __init__(self, ctx_lenght=2048):
        from datasets import load_dataset
        self.ds = load_dataset("wikipedia", "20220301.fr")
        self.dst = self.ds["train"]
        self.ctx_lenght = ctx_lenght

    def raw_data_generator(self):
        for element in self.dst:
            yield element["text"]

    def tokenized_data_generator(self):
        from RthGPT.tokenizer import RthTokenizer
        tokenizer = RthTokenizer("wiki16k.json")
        for element in self.dst:
            tokenized_text = tokenizer.tokenizer.encode(element["text"])
            tokenized_text = tokenized_text.ids
            #print(tokenized_text)

            i = 0
            increasing_content = []
            for token in tokenized_text:
                ic_to_yield = increasing_content[:self.ctx_lenght] + [0]*(self.ctx_lenght-len(increasing_content))

                yield (np.array(ic_to_yield).reshape(1, len(ic_to_yield)), np.array(i).reshape(1,1)), np.array(token).reshape(1,1)
                increasing_content.append(token)
                i += 1




    def tensorflow_dataset_generator(self):
        output_type = (tf.int32, tf.int32), tf.int32
        output_shape = (tf.TensorShape([None, self.ctx_lenght]), tf.TensorShape([None, 1])), tf.TensorShape([None, 1])
        tfds = tf.data.Dataset.from_generator(self.tokenized_data_generator, output_types=output_type, output_shapes=output_shape)
        return tfds

if __name__ == '__main__':
    dataset = HuggingfaceWikipediaFr()
    generator = dataset.tokenized_data_generator()
    for element in generator:
        print(element)
        input()