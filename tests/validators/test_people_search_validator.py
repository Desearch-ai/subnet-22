import unittest
from unittest.mock import patch
from neurons.validators.people_search_validator import PeopleSearchValidator
from tests_data.mocks.neuron import mock_neuron
from datura.protocol import PeopleSearchSynapse


class TestPeopleSearchValidator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.neuron = mock_neuron()
        self.validator = PeopleSearchValidator(self.neuron)

    async def test_generate_criteria(self):
        synapse = PeopleSearchSynapse(
            query="AI startup founders in London with a PhD in machine learning"
        )
        await self.validator.generate_criteria(synapse)
        print(synapse.criteria)


if __name__ == "__main__":
    unittest.main()
