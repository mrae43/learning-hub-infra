def test_package_importable() -> None:
    """Verify the retrieval_qa package is installed and importable."""
    import retrieval_qa

    assert retrieval_qa.__name__ == "retrieval_qa"
