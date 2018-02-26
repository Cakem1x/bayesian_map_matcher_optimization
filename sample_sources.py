#!/usr/bin/env python3

##########################################################################
# Copyright (c) 2017 German Aerospace Center (DLR). All rights reserved. #
# SPDX-License-Identifier: BSD-2-Clause                                  #
##########################################################################

import os
import pickle

import samples
# import the DLR specific code to generae data with the DLR map matcher
import dlr_map_matcher_interface_tools
MAP_MATCHER_INTERFACE_MODULE = dlr_map_matcher_interface_tools


"""
Contains classes that serve as sample sources and are able to generate samples.
Samples are used to define an objective function (see objective_function.py) via discrete observations of that function.

Sample sources are expected to implement the __getitem__ method over which they'll return a sample.
This modules expects parameters that define a sample as a python dictionary.

For quick tests, you can use MapMatcherSampleSourceTest to get some fake MapMatcherSample objects without long evaluation durations.
When using a real sample source, consider using within the SampleDatabase class.
The SampleDatabase can be used as a intermediate layer between objective function and your actual sample source.
It will save all generated sample in a pickled dictionary.
This way, samples won't be generated twice, for example if you rerun your experiment.
Additionally, SampleDatabase supplies methods to iterate over all generated samples, which can useful for visualization purposes.
"""


class SampleDatabase(object):
    """
    The SampleDatabase class can be used as an intermediate module between objective function and an actual sample source.
    This class will only work while the PYTHONHASHSEED remains the same. To enforce this, set this env variable to a fixed value before starting your script.
    Otherwise, the dict_hash method returns different results for the same parameter dictionary in different sessions.

    When a sample is requested via __getitem__, the database will immediately return that sample if it was previously generated.
    If it doesn't exist in the database, the request will be conveyed to the actual sample source, which will generate the sample.
    Before returning from the __getitem__ call, the database will register the newly generated sample in its database.
    Samples aren't kept in memory, but are stored on disk as pickle files.

    The "database" is just a python dictionary. It is indexed via hash of the parameters with which a sample is generated (see dict_hash function).
    Each item contains the following data:
        * pickle_name: The name or identifier of the pickled sample object. Used to find the sample's pickled representation in the sample_dir.
        * params_dict: The complete rosparams dict used to generate this Sample. Its hash should be equal to the item's key.
    """

    def __init__(self, database_path, sample_dir_path, sample_source):
        """
        Initializes the SampleDatabase object.

        :param database_path: Path to the database file (the pickled database dict).
        :param sample_dir_path: Path to the directory where samples created by this SampleDatabase should be stored.
        :param sample_source: Sample source object, which generates new samples via its __getitem__(params_dict) method.
        """
        # Error checking
        if os.path.isdir(database_path):
            raise ValueError("Given database path is a directory!", database_path)
        if not os.path.isdir(sample_dir_path):
            raise ValueError("Given sample_dir_path path is not a directory!", sample_dir_path)

        self._database_path = database_path
        self.sample_dir_path = sample_dir_path
        self.sample_source = sample_source

        if os.path.exists(self._database_path): # If file exists...
            print("\tFound existing datapase pickle, loading from:", self._database_path, end=" ")
            # ...initialize the database from the file handle (hopefully points to a pickled dict...)
            with open(self._database_path, 'rb') as db_handle:
                self._db_dict = pickle.load(db_handle)
            print("- Loaded", len(self._db_dict), "samples.")
        else:
            print("\tDidn't find existing database pickle, initializing new database at", self._database_path, end=".\n")
            self._db_dict = {} # ..otherwise initialize as an empty dict and save it
            self._save()

    def _save(self):
        """
        Pickles the current state of the database dict.
        """
        with open(self._database_path, 'wb') as db_handle:
            pickle.dump(self._db_dict, db_handle)

    def exists(self, params_dict):
        """
        Returns whether a sample with the given rosparams already exists in the database.
        """
        return SampleDatabase.dict_hash(params_dict) in self._db_dict.keys()

    def __len__(self):
        """
        Returns the total number of samples stored in this database.
        """
        return len(self._db_dict)

    def __iter__(self):
        """
        Iterator for getting all sample objects contained in the database.
        """
        for sample in self._db_dict.values():
            yield self._unpickle_sample(sample['pickle_name'])

    def __getitem__(self, params_dict):
        """
        Main interface method of this class.
        Returns the sample corresponding to the given params_dict.
        Either from the dictionary, if a sample was already generated with those parameters.
        Otherwise, if the requested sample doesn't exist in the db, it'll get generated via the sample_source member.
        This causes the function call to block until it's done. (possibly for hours)

        :param params_dict: The parameters dictionary that defines the requested sample.
        """

        params_hashed = SampleDatabase.dict_hash(params_dict)
        if not self.exists(params_dict): # Check whether the sample needs to get generated
            # Generate a new sample and store it in the database
            print("\tNo sample with hash ", params_hashed, " in database, forwarding request to my sample_source.")
            generated_sample = self.sample_source[params_dict]
            print("\tSample generation finished, adding it to database.")
            self.add_sample(generated_sample)
        # Get the sample's db entry
        db_entry = self._db_dict[params_hashed]
        # load the Sample from disk
        print("\tRetrieving sample ", db_entry['pickle_name'], "(hash: ", params_hashed, ") from db.", sep="'")
        extracted_sample = self._unpickle_sample(pickle_name)
        # Do a sanity check of the parameters, just in case of a hash collision
        if not params_dict == db_entry['params_dict']:
            raise LookupError("Got a sample with hash " + params_hashed + ", but its parameters didn't match the requested parameters. (Hash function collision?)", params_dict, db_entry['params_dict'])
        return extracted_sample

    def _pickle_sample(self, sample, pickle_name, override_existing=False):
        """
        Helper method for saving samples to disk.

        :param sample: The Sample which should be pickled.
        :param pickle_name: The name of the Sample's pickled representation.
        """
        pickle_path = os.path.abspath(os.path.join(self.sample_dir_path, pickle_name, ".pkl"))
        # Safety check, don't just overwrite other pickles!
        if not override_existing and os.path.exists(pickle_path):
            raise ValueError("A pickle file already exists at the calculated location:", pickle_path)
        print("\tPickling Sample object for later usage to:", pickle_path)
        with open(pickle_path, 'wb') as sample_pickle_handle:
            pickle.dump(sample, sample_pickle_handle)

    def _unpickle_sample(self, pickle_name):
        """
        Helper method for loading samples from disk

        :param pickle_name: Path to the Sample's pickled representation.
        """
        pickle_path = os.path.abspath(os.path.join(self.sample_dir_path, pickle_name, ".pkl"))
        with open(pickle_path, 'rb') as sample_pickle_handle:
            sample = pickle.load(sample_pickle_handle)
            if not isinstance(sample, Sample):
                raise TypeError("The object unpickled from", pickle_path, "isn't an Sample instance!")
            return sample

    def add_sample(self, sample, params_dict, override_existing=False):
        """
        Adds a new Sample to the database and saves its pickled representation to disk.

        :param sample: The Sample object itself.
        :param params_dict: The dictionary of parameters that were used to generate the sample.
        :param override_existing: Whether an exception should be thrown if a sample with that name or hash already exists.
        """
        if sample.name is None:
            sample.name = str(SampleDatabase.dict_hash(sample))
            print("\tWarning:", "sample's name is None. Setting it to the hash of its parameters:", sample.name)
        self._pickle_sample(sample, sample.name, override_existing)
        params_hashed = SampleDatabase.dict_hash(params_dict)
        # Safety check, don't just overwrite a db entry
        if not override_existing and params_hashed in self._db_dict.keys():
            raise LookupError("Newly created sample's hash already exists in the database! Hash:", str(params_hashed),\
                              "Existing sample's pickle name is:", self._db_dict[params_hashed]['pickle_name'])
        # Add new Sample to db and save the db
        print("\tRegistering sample to database at hash(params):", params_hashed)
        self._db_dict[params_hashed] = {'pickle_name': sample.name, 'params_dict': params_dict}
        self._save()

    def remove_sample(self, params_hashed):
        """
        Removes a Sample's entry from the database and its pickled representation from disk.

        :param params_hashed: The sample's parameter's hash.
        """

        if not params_hashed in self._db_dict:
            raise LookupError("Couldn't find a sample with hash", params_hashed)
        # Get sample from pickled representation
        pickle_name = self._db_dict[params_hashed]['pickle_name']
        pickle_path = os.path.join(self.sample_dir_path, pickle_name, ".pkl")
        if not os.path.isfile(pickle_path):
            print("\tWarning: Couldn't find Sample's pickled representation at '" +\
                  pickle_path + "'.")
        else:
            print("\tRemoving Sample's pickle '" + pickle_path + "' from disk.")
            os.remove(pickle_path)
        print("\tRemoving Sample's db entry '" + str(sample_hash) + "'.")
        del self._db_dict[sample_hash]
            
        # Only save the db at the end, after we know everything worked
        self._save()

    @classmethod
    def dict_hash(cls, params_dict):
        """
        Calculates and returns a hash from the given params_dict.
        """
        # Create a copy of the params dict and convert all lists to tuples. 
        # This is done, because lists aren't hashable.
        params_dict = params_dict.copy()
        for key, value in params_dict.items():
            if isinstance(value, list):
                params_dict[key] = tuple(value)
        return hash(frozenset(params_dict.items()))


class MapMatcherSampleSource(object):
    """
    The MapMatcherSampleSource generates MapMatcherSamples by using an external map matcher pipeline.
    This class calls the external map matcher pipeline by using two methods defined in MAP_MATCHER_INTERFACE_MODULE:
        * create_objective_function_sample
        * generate_sample
    """
    def __init__(self, config):
        """
        Initializes the MapMatcherSampleSource.
        :param config: Config for the sample generation in the MAP_MATCHER_INTERFACE_MODULE.
        """
        self._config = config

    def __getitem__(self, params_dict):
        """
        Generates a new MapMatcherSample with the given parameters and returns it.

        :param params_dict: A dictionary of parameters of the requested sample.
        """
        # This call will lock until the map matcher evaluation is finished
        results_path = MAP_MATCHER_INTERFACE_MODULE.generate_sample(params_dict, self._config)
        generated_sample_params_dict, generated_sample = self.create_sample_from_map_matcher_results(results_path)
        # Check if the parameters were conveyed correctly
        if not generated_sample_params_dict == params_dict:
            raise RuntimeError("Sample requested with parameters", params_dict, "ended up being generated with parameters", generated_sample_params_dict, "!")
        return generated_sample

    def create_sample_from_map_matcher_results(self, results_path, override_existing=False):
        """
        Creates a new Sample object from a finished map matcher run and adds it to the database.

        :param results_path: The path to the directory which contains the map matcher's results.
        """

        results_path = os.path.abspath(results_path)
        print("\tCreating new Sample from map matcher result at", results_path, end=".\n")
        sample = MapMatcherSample()
        # This function actually fills the sample with data.
        # Its implementation depends on which map matching pipeline is optimized.
        params_dict = INTERFACE_MODULE.create_objective_function_sample(results_path, sample, self._config)
        # Calculate a name for the sample
        sample.name = os.path.basename(results_path)
        if sample.name == "results": # In some cases the results are placed in a dir called 'results'
            # If that's the case, we'll use the name of the directory above, since 'results' is a bad name & probably not unique
            sample.name = os.path.basename(os.path.dirname(results_path))
        return params_dict, sample

    # TODO: Sanity check after sample generation: params from file equal to params from request?
    # TODO: get sample from finished eval (with params from the results dir), for --add functionality
