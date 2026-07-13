def test_package_importable() -> None:
    """Verify the ingestion package is installed and importable."""
    import ingestion

    assert ingestion.__name__ == "ingestion"
