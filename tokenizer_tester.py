import RthGPT

txt = """T'as vu ? C'est dingue, tu crois pas ?
"""

tk = RthGPT.RthTokenizer("./pn_4k.json")
ecded = tk.tokenizer.encode(txt)
print(
    txt
)
print(ecded.tokens)
print(ecded.ids)
print(len(ecded.tokens))