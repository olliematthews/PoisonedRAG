from openai import OpenAI, AsyncOpenAI

client = OpenAI()
aclient = OpenAI()


def get_embeddings(texts, model="text-embedding-3-small"):
    texts = [t.replace("\n", " ") for t in texts]
    return [
        d.embedding for d in client.embeddings.create(input=texts, model=model).data
    ]


async def get_embedding_async(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return await aclient.embeddings.create(input=[text], model=model).data[0].embedding
