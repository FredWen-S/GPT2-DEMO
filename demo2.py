import tiktoken

enc=tiktoken.get_encoding("gpt2")

ids=enc.encode("Hello,world")

print(ids)
print(enc.decode(ids))
print('词表大小',enc.n_vocab)