import os

import tokenizers as t
from tokenizers import models as tm
from tokenizers import normalizers as tn
from tokenizers import pre_tokenizers as tpt
from tokenizers import decoders as tdc
from tokenizers import trainers as tr
class RthTokenizer:
    def __init__(self, filepath=None, vocab_size=16384) -> None:
        if filepath is not None:
            try :
                self.tokenizer = t.Tokenizer.from_file(filepath)
                self.vocab_size = self.tokenizer.get_vocab_size()
            except Exception as e :
                raise e
        else :
            self.tokenizer = t.Tokenizer(tm.BPE())
            self.tokenizer.normalizer = tn.NFKC()
            self.tokenizer.pre_tokenizer = tpt.ByteLevel()
            self.tokenizer.decoder = tdc.ByteLevel()
            self.vocab_size = vocab_size

    def train_tokenizer(self, generator, save_name:str):
        self.trainer = tr.BpeTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=tpt.ByteLevel.alphabet(),
            special_tokens=["<[PAD]>", "<[BOS]>", "<[EOS]>"]
        )
        self.tokenizer.train_from_iterator(generator, trainer=self.trainer)
        self.tokenizer.save(save_name)
