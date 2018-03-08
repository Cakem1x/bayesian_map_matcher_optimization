from unittest import TestCase

from bayropt import SampleDatabase

class TestDatabase(TestCase):
    def test_creation(self):
        db = SampleDatabase("sample_db.pkl", ".", None)
        self.assertTrue(isinstance(db, SampleDatabase))
