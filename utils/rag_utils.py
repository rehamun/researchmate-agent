import numpy as np
from .llm_utils import get_client, DEFAULT_EMBEDDING_MODEL


def _normalize(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def embed_texts(texts, model=None, batch_size=32):
    client = get_client()
    model = model or DEFAULT_EMBEDDING_MODEL

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def build_chunk_index(chunks, embedding_model=None):
    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts, model=embedding_model)

    indexed_chunks = []
    for chunk, emb in zip(chunks, embeddings):
        record = dict(chunk)
        record["embedding"] = _normalize(emb).tolist()
        indexed_chunks.append(record)

    return indexed_chunks


def retrieve_top_chunks(query, indexed_chunks, top_k=5, embedding_model=None):
    if not indexed_chunks:
        return []

    query_embedding = embed_texts([query], model=embedding_model)[0]
    query_embedding = _normalize(query_embedding)

    scored = []
    for chunk in indexed_chunks:
        chunk_embedding = np.array(chunk["embedding"], dtype=np.float32)
        score = float(np.dot(query_embedding, chunk_embedding))
        item = dict(chunk)
        item["score"] = score
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
