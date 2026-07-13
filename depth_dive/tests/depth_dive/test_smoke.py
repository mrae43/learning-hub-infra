def test_package_importable() -> None:
    """Verify the depth_dive package is installed and importable."""
    import depth_dive

    assert depth_dive.__name__ == "depth_dive"
