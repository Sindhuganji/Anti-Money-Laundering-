"""
Unit tests for Phase 6: Streamlit dashboard switching and multi-dataset integration.
Uses streamlit.testing.v1.AppTest to simulate interactive user sessions.
"""
import unittest
import os
from streamlit.testing.v1 import AppTest

class TestEllipticDashboardIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "streamlit_app.py"))

    def test_01_dashboard_loads_without_exceptions(self):
        """Test default load (IBM Banking mode)."""
        at = AppTest.from_file(self.app_path, default_timeout=30)
        at.run()
        self.assertFalse(at.exception)

    def test_02_switch_to_cryptocurrency_mode(self):
        """Test switching Active Surveillance Domain to Cryptocurrency (Elliptic Bitcoin)."""
        at = AppTest.from_file(self.app_path, default_timeout=30)
        at.run()
        self.assertFalse(at.exception)
        
        # Switch domain to Cryptocurrency
        domain_sb = at.sidebar.selectbox[0]
        domain_sb.select("Cryptocurrency (Elliptic Bitcoin)").run()
        self.assertFalse(at.exception)
        
        # Verify page options now include Blockchain Graph Analytics
        nav_radio = at.sidebar.radio[0]
        self.assertIn("🌐 Blockchain Graph Analytics", nav_radio.options)

    def test_03_crypto_graph_analytics_page(self):
        """Test navigating to Blockchain Graph Analytics page."""
        at = AppTest.from_file(self.app_path, default_timeout=30)
        at.run()
        at.sidebar.selectbox[0].select("Cryptocurrency (Elliptic Bitcoin)").run()
        at.sidebar.radio[0].set_value("🌐 Blockchain Graph Analytics").run()
        self.assertFalse(at.exception)

    def test_04_crypto_model_performance_page(self):
        """Test navigating to Model Performance in Cryptocurrency mode."""
        at = AppTest.from_file(self.app_path, default_timeout=30)
        at.run()
        at.sidebar.selectbox[0].select("Cryptocurrency (Elliptic Bitcoin)").run()
        at.sidebar.radio[0].set_value("📈 Model Performance").run()
        self.assertFalse(at.exception)

    def test_05_crypto_explainable_ai_page(self):
        """Test navigating to Explainable AI in Cryptocurrency mode."""
        at = AppTest.from_file(self.app_path, default_timeout=30)
        at.run()
        at.sidebar.selectbox[0].select("Cryptocurrency (Elliptic Bitcoin)").run()
        at.sidebar.radio[0].set_value("🧠 Explainable AI (SHAP)").run()
        self.assertFalse(at.exception)

if __name__ == "__main__":
    unittest.main()
