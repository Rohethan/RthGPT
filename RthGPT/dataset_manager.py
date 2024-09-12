

class HuggingfaceWikipediaFr():
    def __init__(self):
        from datasets import load_dataset
        self.ds = load_dataset("wikipedia", "20220301.fr")
        self.dst = self.ds["train"]

    def raw_data_iter(self):
        for element in self.dst:
            yield element["text"]
