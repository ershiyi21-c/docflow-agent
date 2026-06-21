from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"

model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("文本不能为空。")

    vector = model.encode(
        cleaned_text,
        normalize_embeddings=True,
    )

    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    cleaned_texts = [text.strip() for text in texts if text.strip()]

    if not cleaned_texts:
        raise ValueError("文本列表不能为空。")

    vectors = model.encode(
        cleaned_texts,
        normalize_embeddings=True,
    )

    return vectors.tolist()