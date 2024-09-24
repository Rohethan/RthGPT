from RthGPT import tokenizer

tk = tokenizer.RthTokenizer(None, 4096)
tk.train_tokenizer_from_folder("./petit_nicolas/", "pn_4k.json")