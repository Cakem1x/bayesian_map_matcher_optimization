from unittest import TestCase
import os
import shutil

from bayropt import SampleDatabase

class TestDatabase(TestCase):
    def setUp(self):
        # Get the dir in which this file resides
        self.test_path = os.path.dirname(os.path.realpath(__file__))
        self.assertTrue(os.path.isdir(self.test_path)) # check that this dir exists
        # add a working dir for our tests to that path
        self.test_path = os.path.join(self.test_path, "sample_sources_workdir")
        self.assertFalse(os.path.exists(self.test_path)) # Make sure this directory doesn't exist yet
        os.mkdir(self.test_path)
        os.mkdir(os.path.join(self.test_path, "samples"))
        # finally, create sample db object to test with
        self.sample_db = SampleDatabase(os.path.join(self.test_path, "sample_db.pkl"), os.path.join(self.test_path, "samples"), None)

    def tearDown(self):
        shutil.rmtree(self.test_path) # delete the workdir again

    def test_creation(self):
        self.assertTrue(isinstance(self.sample_db, SampleDatabase))
        self.assertTrue(os.path.isdir(os.path.join(self.test_path, "samples")))
        self.assertTrue(os.path.isfile(os.path.join(self.test_path, "sample_db.pkl")))
