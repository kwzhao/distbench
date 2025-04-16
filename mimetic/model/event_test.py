import unittest

import event


class EventTest(unittest.TestCase):
    def tests_dependency_on_start_equality_ignore_delta(self):
        a = event.DependencyOnStart(delta_ns=0)
        b = event.DependencyOnStart(delta_ns=1)
        self.assertEqual(a, b)

    if __name__ == "__main__":
        unittest.main()
