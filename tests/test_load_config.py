import os
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from utils.load_config import LoadConfig


# === Test loading valid config from YAML ===
@patch("utils.load_config.here")
@patch("utils.load_config.OpenAIEmbeddings")
def test_load_config_success(mock_embeddings, mock_here, tmp_path):
    # Setup fake YAML content
    fake_config = {
        "llm_config": {
            "engine": "gpt-4",
            "temperature": 0.5,
            "llm_system_role": "You are smart."
        },
        "directories": {
            "persist_directory": "vector_db",
            "custom_persist_directory": "custom_db",
            "data_directory": "data"
        },
        "embedding_model_config": {
            "engine": "text-embedding-ada-002"
        },
        "retrieval_config": {
            "k": 3
        },
        "splitter_config": {
            "chunk_size": 500,
            "chunk_overlap": 100
        },
        "summarizer_config": {
            "max_final_token": 2000,
            "token_threshold": 1500
        },
        "memory": {
            "number_of_q_a_pairs": 3
        }
    }

    # Write the config to a temporary YAML file
    cfg_file = tmp_path / "app_config.yml"
    with open(cfg_file, "w") as f:
        yaml.dump(fake_config, f)

    # Mock the `here()` function to return our temp path
    mock_here.side_effect = lambda x=None: cfg_file if x == "configs/app_config.yml" else tmp_path / x

    cfg = LoadConfig()

    assert cfg.llm_engine == "gpt-4"
    assert cfg.temperature == 0.5
    assert cfg.k == 3
    assert cfg.chunk_size == 500
    assert cfg.chunk_overlap == 100
    assert cfg.max_final_token == 2000
    assert cfg.persist_directory.exists()
    assert cfg.custom_persist_directory.exists()
    assert cfg.data_directory.exists()


# === Test OpenAI credentials loaded from env ===
@patch.dict(os.environ, {
    "OPENAI_API_KEY": "sk-test-123",
    "OPENAI_API_TYPE": "openai",
    "OPENAI_API_BASE": "https://test.api",
    "OPENAI_API_VERSION": "v1"
})
@patch("utils.load_config.here")
@patch("utils.load_config.OpenAIEmbeddings")
def test_openai_config_loaded(mock_embeddings, mock_here, tmp_path):
    cfg_file = tmp_path / "app_config.yml"
    mock_here.side_effect = lambda x=None: cfg_file if x == "configs/app_config.yml" else tmp_path / x

    with open(cfg_file, "w") as f:
        yaml.dump({
            "llm_config": {},
            "directories": {},
            "embedding_model_config": {},
            "retrieval_config": {},
            "splitter_config": {},
            "summarizer_config": {},
            "memory": {}
        }, f)

    cfg = LoadConfig()
    assert cfg.openai_api_key == "sk-test-123"


# === Test remove_directory ===
def test_remove_directory(tmp_path):
    cfg = LoadConfig.__new__(LoadConfig)  # avoid init
    test_dir = tmp_path / "test_remove"
    test_dir.mkdir()

    assert test_dir.exists()
    cfg.remove_directory(test_dir)
    assert not test_dir.exists()


# === Test to_dict returns key configs ===
@patch("utils.load_config.here")
@patch("utils.load_config.OpenAIEmbeddings")
def test_to_dict_output(mock_embeddings, mock_here, tmp_path):
    cfg_file = tmp_path / "app_config.yml"
    mock_here.side_effect = lambda x=None: cfg_file if x == "configs/app_config.yml" else tmp_path / x

    with open(cfg_file, "w") as f:
        yaml.dump({
            "llm_config": {"engine": "gpt-4"},
            "directories": {},
            "embedding_model_config": {},
            "retrieval_config": {},
            "splitter_config": {},
            "summarizer_config": {},
            "memory": {}
        }, f)

    cfg = LoadConfig()
    result = cfg.to_dict()
    assert isinstance(result, dict)
    assert "llm_engine" in result
    assert result["llm_engine"] == "gpt-4"
