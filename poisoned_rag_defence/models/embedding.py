from openai import OpenAI

client = OpenAI()


def get_openai_embeddings(texts: list[str], model: str = "text-embedding-3-small"):
    """Get the openai embeddings for a list of texts"""
    texts = [t.replace("\n", " ") for t in texts]
    return [
        d.embedding for d in client.embeddings.create(input=texts, model=model).data
    ]
