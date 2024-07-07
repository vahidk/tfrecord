import unittest
import sys

if __name__ == '__main__':
    loader = unittest.TestLoader()
    tests = loader.discover('tests')
    testRunner = unittest.TextTestRunner()
    result = testRunner.run(tests)
    # Exit with a non-zero status code if tests failed
    if not result.wasSuccessful():
        sys.exit(1)
