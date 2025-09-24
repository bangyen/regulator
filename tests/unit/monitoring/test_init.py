"""
Tests for the monitoring package __init__.py file.

This module tests the monitoring package initialization and exports.
"""


class TestMonitoringPackage:
    """Test the monitoring package initialization."""

    def test_import_monitoring_package(self):
        """Test that monitoring package can be imported."""
        import src.monitoring

        assert src.monitoring is not None

    def test_enhanced_dashboard_import(self):
        """Test that EnhancedMonitoringDashboard can be imported from package."""
        from src.monitoring import EnhancedMonitoringDashboard

        assert EnhancedMonitoringDashboard is not None

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from src.monitoring import __all__ as monitoring_all

        expected_exports = ["EnhancedMonitoringDashboard"]
        assert monitoring_all == expected_exports

    def test_enhanced_dashboard_class_available(self):
        """Test that the imported class is actually the correct class."""
        from src.monitoring import EnhancedMonitoringDashboard
        from src.monitoring.enhanced_dashboard import (
            EnhancedMonitoringDashboard as DirectImport,
        )

        assert EnhancedMonitoringDashboard is DirectImport

    def test_package_docstring(self):
        """Test that package has appropriate docstring."""
        import src.monitoring

        assert src.monitoring.__doc__ is not None
        assert "Enhanced Monitoring Package" in src.monitoring.__doc__
        assert "monitoring capabilities" in src.monitoring.__doc__

    def test_no_unexpected_exports(self):
        """Test that package doesn't export unexpected items."""
        import src.monitoring

        # Get all public attributes (not starting with _)
        public_attrs = [
            attr for attr in dir(src.monitoring) if not attr.startswith("_")
        ]

        # Should only have the expected exports plus any standard module attributes
        expected_attrs = ["EnhancedMonitoringDashboard"]

        for attr in public_attrs:
            # Allow standard module attributes or expected exports
            assert attr in expected_attrs or attr in [
                "enhanced_dashboard"
            ]  # Module names are also accessible

    def test_import_from_enhanced_dashboard_module(self):
        """Test direct import from the enhanced_dashboard module."""
        from src.monitoring.enhanced_dashboard import EnhancedMonitoringDashboard

        # Should be able to create an instance
        dashboard = EnhancedMonitoringDashboard()
        assert dashboard is not None
        assert hasattr(dashboard, "log_dir")
        assert hasattr(dashboard, "colors")

    def test_package_structure_consistency(self):
        """Test that package structure is consistent."""
        # Import both ways
        from src.monitoring import EnhancedMonitoringDashboard as PackageImport
        from src.monitoring.enhanced_dashboard import (
            EnhancedMonitoringDashboard as ModuleImport,
        )

        # Should be the same class
        assert PackageImport.__name__ == ModuleImport.__name__
        assert PackageImport.__module__ == ModuleImport.__module__
