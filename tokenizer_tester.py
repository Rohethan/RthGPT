import RthGPT
tk = RthGPT.RthTokenizer("./pn_4k.json")
txt = """T'as vu ? C'est dingue, tu crois pas ?
"""


ecded = tk.tokenizer.encode(txt)
print(
    txt
)
print(ecded.tokens)
print(ecded.ids)
print(len(ecded.tokens))