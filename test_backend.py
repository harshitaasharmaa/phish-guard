"""PhishGuard - Backend Test Suite.

Tests the Flask backend endpoints using Flask's test client:
1. GET /health - Service liveness and model initialization.
2. POST /predict - Validation failures on empty, non-JSON, or malformed inputs.
3. POST /predict - Safe refusal behavior when given a raw URL with incomplete feature coverage.
4. POST /predict - Successful prediction when a complete 30-feature vector is provided.
5. Error handlers - 404 for nonexistent endpoints.
"""

import unittest
import pandas as pd
from app import app, expected_features


class TestPhishGuardBackend(unittest.TestCase):

    def setUp(self):
        """Set up Flask test client."""
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_index_route(self):
        """Test GET / renders the frontend HTML dashboard."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"PhishGuard", response.data)
        self.assertIn(b"URL Heuristic Scanner", response.data)
        self.assertIn(b"Check URL", response.data)

    def test_health_endpoint(self):
        """Test GET /health returns 200 OK with model status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["features_required_count"], 30)
        self.assertIn("RandomForestClassifier", data["model_type"])

    def test_predict_non_json_request(self):
        """Test POST /predict rejects non-JSON payloads with 400."""
        response = self.client.post("/predict", data="not json", content_type="text/plain")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("Invalid Content-Type", data["error"])

    def test_predict_missing_url_field(self):
        """Test POST /predict rejects JSON without 'url' or 'features' with 400."""
        response = self.client.post("/predict", json={"website": "https://example.com"})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("Missing required field", data["error"])

    def test_predict_empty_url(self):
        """Test POST /predict rejects empty or whitespace-only URL with 400."""
        response = self.client.post("/predict", json={"url": "   "})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("Invalid URL", data["error"])

    def test_predict_non_string_url(self):
        """Test POST /predict rejects non-string URL with 400."""
        response = self.client.post("/predict", json={"url": 12345})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("Invalid URL", data["error"])

    def test_predict_url_safe_incomplete_refusal(self):
        """Test POST /predict safely refuses raw URL when features cannot be fully computed."""
        test_url = "https://www.paypal-update.com/login@verify"
        response = self.client.post("/predict", json={"url": test_url})
        self.assertEqual(response.status_code, 422)

        data = response.get_json()
        self.assertEqual(data["status"], "incomplete_feature_extraction")
        self.assertFalse(data["can_predict"])
        self.assertIn("Incomplete feature extraction", data["error"])

        # Check coverage breakdown
        summary = data["coverage_summary"]
        self.assertEqual(summary["total_required"], 30)
        self.assertEqual(summary["total_available"], 8)
        self.assertEqual(summary["total_missing"], 22)

        # Check extracted lexical features
        extracted = data["extracted_lexical_features"]
        self.assertIn("having_ip_address", extracted)
        self.assertIn("prefix_suffix", extracted)
        self.assertIn("having_at_symbol", extracted)
        self.assertEqual(extracted["having_at_symbol"], -1)  # '@' was present
        self.assertEqual(extracted["prefix_suffix"], -1)     # '-' was in domain

        # Check missing categories are reported
        missing_cats = data["missing_features_by_category"]
        self.assertIn("webpage_content_dom", missing_cats)
        self.assertIn("network_and_dns", missing_cats)
        self.assertIn("external_third_party_apis", missing_cats)

    def test_predict_complete_30_features_success(self):
        """Test POST /predict succeeds when a complete 30-feature vector is provided."""
        # Load one real sample from dataset
        df = pd.read_csv("data/phishing_websites.csv")
        sample_row = df.iloc[0].drop(labels=["result"]).to_dict()

        response = self.client.post("/predict", json={"features": sample_row})
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn(data["prediction"], [-1, 1])
        self.assertIn(data["label"], ["Phishing", "Legitimate"])
        self.assertGreaterEqual(data["confidence"], 50.0)
        self.assertIn("probabilities", data)
        self.assertEqual(data["features_used_count"], 30)

    def test_predict_partial_features_vector_rejected(self):
        """Test POST /predict rejects a features dictionary with missing keys."""
        partial = {"having_ip_address": 1, "url_length": 1}
        response = self.client.post("/predict", json={"features": partial})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("Missing required features", data["error"])
        self.assertEqual(data["missing_count"], 28)

    def test_404_not_found(self):
        """Test non-existent endpoints return 404 JSON."""
        response = self.client.get("/unknown_endpoint")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertEqual(data["error"], "Not Found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
