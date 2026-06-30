from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str
    tavily_api_key: str
    # Point at Project 1's existing ChromaDB — no need to re-upload documents
    chroma_persist_dir: str = "../../01-rag-evaluation-system/backend/chroma_db"
    top_k_results: int = 4

    class Config:
        env_file = ".env"


settings = Settings()
