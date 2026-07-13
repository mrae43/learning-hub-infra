def test_package_importable() -> None:
    """Verify the api package is installed and importable."""
    import api

    assert api.__name__ == "api"
