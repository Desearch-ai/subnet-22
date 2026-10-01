def test_setup_package_discovery_includes_the_miner_validator_and_shared_code():
    from setuptools import find_packages

    packages = set(find_packages())

    assert {
        "desearch",
        "desearch.extraction",
        "neurons.miners",
        "neurons.validators",
    } <= packages
