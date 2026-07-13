import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embedding = embed(["I am going for a walk"])  # (1, 512)

print(embedding.shape)
print(embedding)

