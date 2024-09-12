import RthGPT

tk = RthGPT.RthTokenizer("./wiki16k.json")
ecded = tk.tokenizer.encode("""Je m'apelle Rohethan, et je suis un fier membre de Péguland.""")
print("""Je m'apelle Rohethan, et je suis un fier membre de Péguland.""")
print(ecded.tokens)
print(ecded.ids)
print(len(ecded.tokens))