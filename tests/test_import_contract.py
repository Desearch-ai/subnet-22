import importlib
import sys


def test_desearch_utils_import_does_not_require_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("APIFY_API_KEY", "test-apify-token")
    sys.modules.pop("desearch.utils", None)

    module = importlib.import_module("desearch.utils")

    assert hasattr(module, "call_openai")


def test_twitter_scraper_actor_import_does_not_require_apify_api_key(monkeypatch):
    monkeypatch.delenv("APIFY_API_KEY", raising=False)
    sys.modules.pop("neurons.validators.apify.twitter_scraper_actor", None)

    module = importlib.import_module("neurons.validators.apify.twitter_scraper_actor")
    actor = module.TwitterScraperActor()

    assert actor.client is None


def test_setup_package_discovery_includes_neurons_subpackages():
    from setuptools import find_packages

    packages = set(find_packages())

    assert "neurons.miners" in packages
    assert "neurons.validators" in packages
    assert "neurons.validators.apify" in packages
