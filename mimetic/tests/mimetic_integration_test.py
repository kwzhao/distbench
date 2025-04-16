import unittest
from mimetic import dataset, utils
from mimetic.model import event, decision, builder, system

class MimeticIntegrationTest(unittest.TestCase):
    def test_imports(self):
        """Test that all modules can be imported properly."""
        # Verify we can use the imports
        self.assertIsNotNone(dataset)
        self.assertIsNotNone(utils)
        self.assertIsNotNone(event)
        self.assertIsNotNone(decision)
        self.assertIsNotNone(builder)
        self.assertIsNotNone(system)

if __name__ == '__main__':
    unittest.main()