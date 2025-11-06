from llama_index.embeddings.openai import OpenAIEmbedding
from config.env import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, EMBED_DIM

embedding_model = OpenAIEmbedding(
    api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL, dimensions=EMBED_DIM
)
