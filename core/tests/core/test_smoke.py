def test_package_importable() -> None:
    """Verify the core package is installed and importable."""
    import core

    assert core.__name__ == "core"
