import os

PORT = os.environ.get("PORT", 8005)
VALIDATOR_SERVICE_PORT = os.environ.get("VALIDATOR_SERVICE_PORT", 8006)
EXPECTED_ACCESS_KEY = os.environ.get("EXPECTED_ACCESS_KEY", "test")
