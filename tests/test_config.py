"""Test config module attributes and env-var overrides."""


def test_max_file_size_bytes_exists():
    """config.MAX_FILE_SIZE_BYTES should be an integer."""
    import config
    assert hasattr(config, "MAX_FILE_SIZE_BYTES")
    assert isinstance(config.MAX_FILE_SIZE_BYTES, int)
    assert config.MAX_FILE_SIZE_BYTES > 0


def test_max_file_size_bytes_default():
    """Default MAX_FILE_SIZE_BYTES should be 50 MB."""
    import config
    assert config.MAX_FILE_SIZE_BYTES == 50 * 1024 * 1024


def test_embedding_model_exists():
    """config.EMBEDDING_MODEL should be a non-empty string."""
    import config
    assert hasattr(config, "EMBEDDING_MODEL")
    assert isinstance(config.EMBEDDING_MODEL, str)
    assert len(config.EMBEDDING_MODEL) > 0


def test_vector_db_dir_exists():
    """VECTOR_DB_DIR should be created at import time."""
    import config
    assert config.VECTOR_DB_DIR.exists()
