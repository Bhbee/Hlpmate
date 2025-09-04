import os
import shutil
import yaml
import openai
from dotenv import load_dotenv
from pathlib import Path
from pyprojroot import here
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()


class LoadConfig:
    def __init__(self) -> None:
        # Load YAML configuration file
        config_path = here("configs/app_config.yml")
        with open(config_path, "r") as cfg_file:
            app_config = yaml.safe_load(cfg_file)

        # === LLM Config ===
        self.llm_engine = app_config["llm_config"].get("engine", "gpt-3.5-turbo")
        self.llm_system_role = app_config["llm_config"].get("llm_system_role", "You are a helpful assistant.")
        self.temperature = app_config["llm_config"].get("temperature", 0.7)

        # === Directories ===
        self.persist_directory = here(app_config["directories"].get("persist_directory", "vector_db")).resolve()
        self.custom_persist_directory = here(app_config["directories"].get("custom_persist_directory", "custom_db")).resolve()
        self.data_directory = here(app_config["directories"].get("data_directory", "data")).resolve()

        # === Embeddings ===
        self.embedding_model_engine = app_config["embedding_model_config"].get("engine", "text-embedding-ada-002")
        self.embedding_model = OpenAIEmbeddings()  # Automatically uses env key

        # === RAG & Chunking ===
        self.k = app_config["retrieval_config"].get("k", 5)
        self.chunk_size = app_config["splitter_config"].get("chunk_size", 1000)
        self.chunk_overlap = app_config["splitter_config"].get("chunk_overlap", 200)

        # === Summarizer Settings ===
        self.max_final_token = app_config["summarizer_config"].get("max_final_token", 3000)
        self.token_threshold = app_config["summarizer_config"].get("token_threshold", 2000)
        self.summarizer_llm_system_role = app_config["summarizer_config"].get(
            "summarizer_llm_system_role", "Summarize academic documents."
        )
        self.final_summarizer_llm_system_role = app_config["summarizer_config"].get(
            "final_summarizer_llm_system_role", "Provide a final summary."
        )
        self.character_overlap = app_config["summarizer_config"].get("character_overlap", 50)

        # === Memory ===
        self.number_of_q_a_pairs = app_config["memory"].get("number_of_q_a_pairs", 5)

        # === Load OpenAI Credentials ===
        self.load_openai_cfg()

        # === Ensure required directories exist ===
        self.create_directory(self.persist_directory)
        self.create_directory(self.custom_persist_directory)
        self.create_directory(self.data_directory)

    def load_openai_cfg(self):
        """Load OpenAI-related environment variables."""
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_type = os.getenv("OPENAI_API_TYPE", "openai")
        openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        openai.api_version = os.getenv("OPENAI_API_VERSION", "v1")
        self.openai_api_key = openai.api_key

    def create_directory(self, path: Path):
        """Create directory if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)

    def remove_directory(self, path: Path):
        """Remove a directory safely."""
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"Removed: {path}")
            except Exception as e:
                print(f"Failed to remove {path}: {e}")

    def to_dict(self):
        """Optional: return config as dictionary for debugging/logging."""
        return {
            "llm_engine": self.llm_engine,
            "temperature": self.temperature,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model_engine": self.embedding_model_engine,
            "openai_api_key": "****" if self.openai_api_key else None,
        }
