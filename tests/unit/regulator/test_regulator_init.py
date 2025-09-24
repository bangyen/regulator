"""
Tests for the regulator package __init__.py file.

This module tests the regulator package initialization and metadata.
"""

import re


class TestRegulatorPackage:
    """Test the regulator package initialization."""

    def test_import_regulator_package(self):
        """Test that regulator package can be imported."""
        import src.regulator

        assert src.regulator is not None

    def test_version_attribute(self):
        """Test that package has version attribute."""
        from src.regulator import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"

    def test_version_format(self):
        """Test that version follows semantic versioning format."""
        from src.regulator import __version__

        # Should match semantic versioning pattern: major.minor.patch
        version_pattern = r"^\d+\.\d+\.\d+$"
        assert re.match(version_pattern, __version__)

    def test_author_attribute(self):
        """Test that package has author attribute."""
        from src.regulator import __author__

        assert __author__ is not None
        assert isinstance(__author__, str)
        assert __author__ == "Bangyen Pham"

    def test_email_attribute(self):
        """Test that package has email attribute."""
        from src.regulator import __email__

        assert __email__ is not None
        assert isinstance(__email__, str)
        assert __email__ == "bangyenp@gmail.com"

    def test_email_format(self):
        """Test that email has valid format."""
        from src.regulator import __email__

        # Basic email format validation
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        assert re.match(email_pattern, __email__)

    def test_package_docstring(self):
        """Test that package has appropriate docstring."""
        import src.regulator

        assert src.regulator.__doc__ is not None
        assert (
            "Regulator: Market Competition & Collusion Detection"
            in src.regulator.__doc__
        )
        assert "machine learning" in src.regulator.__doc__
        assert "Python package" in src.regulator.__doc__

    def test_package_metadata_consistency(self):
        """Test that package metadata is consistent."""
        from src.regulator import __version__, __author__, __email__

        # All metadata should be strings
        assert isinstance(__version__, str)
        assert isinstance(__author__, str)
        assert isinstance(__email__, str)

        # All should be non-empty
        assert len(__version__) > 0
        assert len(__author__) > 0
        assert len(__email__) > 0

    def test_no_unexpected_exports(self):
        """Test that package doesn't export unexpected items."""
        import src.regulator

        # Get all public attributes (not starting with _)
        public_attrs = [
            attr
            for attr in dir(src.regulator)
            if not attr.startswith("_")
            or attr in ["__version__", "__author__", "__email__"]
        ]

        expected_attrs = ["__version__", "__author__", "__email__"]

        # Should only have the expected metadata attributes
        for attr in public_attrs:
            assert attr in expected_attrs, f"Unexpected attribute: {attr}"

    def test_import_all_metadata_together(self):
        """Test importing all metadata attributes together."""
        from src.regulator import __version__, __author__, __email__

        # Should all be accessible
        metadata = {"version": __version__, "author": __author__, "email": __email__}

        assert all(value is not None for value in metadata.values())
        assert all(isinstance(value, str) for value in metadata.values())

    def test_package_description_accuracy(self):
        """Test that package description accurately describes functionality."""
        import src.regulator

        docstring = src.regulator.__doc__.lower()

        # Should mention key concepts
        assert "market" in docstring
        assert "competition" in docstring
        assert "collusion" in docstring or "collusive" in docstring
        assert "detection" in docstring or "detecting" in docstring
        assert "machine learning" in docstring

    def test_version_incrementability(self):
        """Test that version number is in a format that can be incremented."""
        from src.regulator import __version__

        # Split version into parts
        parts = __version__.split(".")
        assert len(parts) == 3, "Version should have 3 parts (major.minor.patch)"

        # Each part should be a valid integer
        for part in parts:
            assert part.isdigit(), f"Version part '{part}' should be numeric"
            assert int(part) >= 0, f"Version part '{part}' should be non-negative"

    def test_contact_information_validity(self):
        """Test that contact information appears valid."""
        from src.regulator import __author__, __email__

        # Author should contain reasonable name format
        assert len(__author__.split()) >= 2, "Author should contain first and last name"

        # Email should match author context
        email_local_part = __email__.split("@")[0]

        # Basic sanity check that email relates to author name
        assert len(email_local_part) > 0
        assert "@" in __email__
        assert "." in __email__.split("@")[1]  # Domain should have TLD
