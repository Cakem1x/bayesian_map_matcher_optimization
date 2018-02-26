##########################################################################
# Copyright (c) 2017 German Aerospace Center (DLR). All rights reserved. #
# SPDX-License-Identifier: BSD-2-Clause                                  #
##########################################################################

"""
Contains the code for modelling the ObjectiveFunction via discrete samples.

A Sample is defined by its full set of rosparams ("complete_rosparams"), since they're necessary to run
the map matcher.
A Sample contains resulting data from the map matcher evaluation with its complete_rosparams.

The Samples are managed by the SampleDatabase, which uses a hashing to quickly find the Sample for a given complete_rosparams set.
If it doesn't exist, a new Sample will be generated using the INTERFACE_MODULE. (this is where you can define some method for generating Sample data for your specific map matcher implementation)

The ObjectiveFunction differentiates "complete_rosparams" and "optimized_rosparams".
It acts as the interface for the optimizer, which only knows about the "optimized_rosparams" (subset of "complete_rosparams").
"""

import os
import pickle

from performance_measures import PerformanceMeasure

# import the DLR specific code to gather data
import dlr_map_matcher_interface_tools
INTERFACE_MODULE = dlr_map_matcher_interface_tools

class ObjectiveFunction(object):
    """
    Models the evaluation function that is optimized by the gaussian process.

    In our use-case, feature-based map matcher evaluation, the evaluation function can't be defined analytically.
    Instead, this class models it as a set of Sample points for which the map matcher pipeline was evaluated.

    This class uses the SampleDatabase class to manage the Samples over which it is defined.
    This should reduce the time subsequent experiments will require, after a bunch of samples have already been generated.
    """

    def __init__(self, sample_db, default_rosparams, optimization_bounds, performance_measure, rounding_decimal_places=0):
        """
        Creates an ObjectiveFunction object.
        
        :param sample_db: A sample database object.
        :param default_rosparams: A dict that contains all rosparams required for running the map matcher
                                  with the default values.
        :param optimization_bounds: A dict that contains the rosparam name of each rosparams to be optimized.
                                    The dict's values are (min, max)-tuples for the optimization bounds of the respective rosparam.
        :param performance_measure: An PerformanceMeasure object
                            Some metrics may terminate the experiment, if your Sample instances don't contain the necessary data.
        :param rounding_decimal_places: The number of decimal places to which parameters of type float should be rounded to.
                                        If zero, no rounding will take place.
        """

        # error checking
        if not isinstance(performance_measure, PerformanceMeasure):
            raise ValueError("Object given wasn't a PerformanceMeasure")

        self.performance_measure = performance_measure
        self._sample_db = sample_db
        self.default_rosparams = default_rosparams
        self.optimization_bounds = optimization_bounds
        self._rounding_decimal_places = rounding_decimal_places

    def evaluate(self, **optimized_rosparams):
        """
        This method supplies the interface for the Optimizer.
        Calculates and returns the current metric ("y") at the given parameters ("X").

        Will convert the optimized_rosparams (from the Optimizer-world) to the complete_rosparams which define 
        a Sample in the map matching pipeline.
        This may also involve casting types of optimized_rosparams to the type the parameter has in the default_rosparams.

        This function may return quickly, if the sample already exists in the sample database.
        Otherwise, the call will block until the map matching pipeline has finished generating the requested sample.

        :param optimized_rosparams: A keyworded argument list.
        """

        # Preprocess the parameters
        self.preprocess_optimized_params(optimized_rosparams)
        print("\033[1;34mSampling evaluation function at:", end="")
        for name, value in optimized_rosparams.items():
            print() # newline
            print("\t", name, " = ", value, sep="", end="")
        print("\033[0m")
        # Create the full set of parameters by updating the default parameters with the optimized parameters.
        complete_rosparams = self.default_rosparams.copy()
        complete_rosparams.update(optimized_rosparams)
        # Get the sample from the db (this call blocks until the sample is generated, if it isn't in the db)
        sample = self._sample_db[complete_rosparams]
        # Calculate and return the metric
        value = self.performance_measure(sample)
        print("\033[1;34m\tSample's performance measure:\033[1;37m", value, "\033[0m")
        return value

    def normalize_parameters(self, optimized_params):
        """
        Takes a dict of optimized parameters as given by the Map Matcher ecosystem and normalizes the values
        to fit the Optimizer ecosystem.

        This method uses the optimization_bounds member.
        The process is inverse to the denormalization method and should be used on all parameters that 
        come from the Map Matcher part of the system and are to be passed into the Optimizer.

        :param optimized_params: The params dict, from the Map Matcher ecosystem
        :returns: The normalized params dict where parameter values are in [0,1].
        """
        for rosparam_name, bounds in self.optimization_bounds.items():
            old_val = optimized_params[rosparam_name] # TODO remove after test
            param_range = bounds[1] - bounds[0]
            optimized_params[rosparam_name] = (optimized_params[rosparam_name] - bounds[0]) / param_range
            print("Normalizing", rosparam_name, "from", old_val, "to", optimized_params[rosparam_name]) # TODO remove after test

        return optimized_params

    def denormalize_parameters(self, optimized_params):
        """
        Takes a dict of optimized parameters as given by the Optimizer module and denormalizes the values
        to fit the Map Matcher ecosystem.

        The process is inverse to the normalization method and should be used on all parameters that 
        come from the Optimizer module to be passed into the Map Matcher part of the system.
        Also, the denormalized values are probably more informative for user output as well.

        :param optimized_params: The params dict, from the Optimizer ecosystem (should be preprocessed before passed in here, see preprocess_optimized_params).
        :returns: The denormalized params dict with the parameter values that actually should be used for map matcher evaluation.
        """
        for rosparam_name, bounds in self.optimization_bounds.items():
            old_val = optimized_params[rosparam_name] # TODO remove after test
            param_range = bounds[1] - bounds[0]
            optimized_params[rosparam_name] = (optimized_params[rosparam_name] * param_range) + bounds[0]
            print("Denormalizing", rosparam_name, "from", old_val, "to", optimized_params[rosparam_name]) # TODO remove after test

        return optimized_params

    def preprocess_optimized_params(self, optimized_params):
        """
        Takes a dict of optimized parameters as given by the optimizer and preprocesses it to fit the ros ecosystem.
        
        This includes a safety check to make sure the parameters requested by the optimizer don't violate the optimization bounds.
        Additionally, all parameter types in the given dict will get casted to their respective type in the initial rosparams.
        Otherwise, dumping them to a yaml file would create binarized numpy.float64 (for example) values, which can't be parsed by rosparam.
        Also, values will get rounded according to the rounding_decimal_places parameter.

        :param optimized_params: The params dict, as requested by the optimizer.
        :returns: The preprocessed params dict, as needed by the ros ecosystem.
        """
        # Iterate over the optimization bounds and check if the current request doesn't violate them
        for rosparam_name, bounds in self.optimization_bounds.items():
            if not rosparam_name in optimized_params:
                raise ValueError(rosparam_name + " should get optimized, but wasn't in given dict of optimized parameters.", optimized_params)
            p_value = optimized_params[rosparam_name]
            if p_value > bounds[1]: # max bound
                raise ValueError(rosparam_name + " value (" + str(p_value) + ") is over max bound (" +\
                                 str(bounds[1]) + ").")
            if p_value < bounds[0]: # min bound
                raise ValueError(rosparam_name + " value (" + str(p_value) + ") is under min bound (" +\
                                 str(bounds[0]) + ").")
        # Iterate over the current request...
        for p_name, p_value in optimized_params.items():
            # ...check if there are parameters in there, that shouldn't be optimized.
            if not p_name in self.optimization_bounds.keys():
                raise ValueError(str(p_name) + " shouldn't get optimized, but was in given dict of optimized parameters.")
            # ...also CAST their type to the type in the default rosparams dict. Otherwise, we may serialize
            # some high-precision numpy float class instead of having a built-in float value on the yaml, that
            # rosparams can actually read. Sadl, we'll lose some precision through that.
            if type(self.default_rosparams[p_name]) != type(p_value):
                #print("\tWarning, casting parameter type", type(p_value), "of", p_name, "to", type(self.default_rosparams[p_name]))
                optimized_params[p_name] = type(self.default_rosparams[p_name])(p_value)
            # ...also ROUND float values according to rounding_decimal_places parameter
            if self._rounding_decimal_places and isinstance(p_value, float):
                rounded_p_value = round(optimized_params[p_name], self._rounding_decimal_places)
                #print("\tWarning, rounding float value", p_name, ":", p_value, "->", rounded_p_value)
                optimized_params[p_name] = rounded_p_value

        return optimized_params

    def samples_filtered(self, fixed_params, enforce_bounds=False):
        """
        Iterator that yields only samples that satisfy all fixed_params definitions.

        :param fixed_params: A dict that maps a subset of optimized rosparams to a desired value.
        Returns samples in the same format as __iter__.
        :param enforce_bounds: If set to True, only samples are returned which also satisfy the current optimization bounds.
        """
        for x, y, s in self:
            # For each sample, check if it's usable:
            usable = True
            for p_name, p_value in x.items():
                if p_name in fixed_params: # For all fixed_params, check if the value is correct
                    if not self.default_rosparams[p_name] == p_value:
                        usable = False
                        break
                else: # For all non-fixed-params: Check if the value is in the optimization bounds
                    if p_value < self.optimization_bounds[p_name][0] or p_value > self.optimization_bounds[p_name][1]:
                        usable = False
                        break
            if usable:
                yield x, y, s

    def __iter__(self):
        """
        Iterator for getting information from all available samples from the database, which define this ObjectiveFunction.

        This means only information from those Samples are yielded, which have parameter values matching the ones in default_rosparams.
        Only parameter values of currently optimized parameters are allowed to differ.
        The filtering is done in the _defined_by function.
        
        Yields tuples (x, y, s), with
            x: A dict of the complete rosparams that were used to create that sample.
            y: The sample's value as given by the current performance measure.
            s: The sample itself, in case other information needs to be extracted.
        """
        for sample in self._sample_db:
            if self._defined_by(sample.parameters):
                yield sample.parameters, self.performance_measure(sample), sample

    def _defined_by(self, complete_rosparams):
        """
        Returns whether the given complete_rosparams is valid for defining this ObjectiveFunction.
        That's the case if all non-optimized parameters of complete_rosparams are equal to this ObjectiveFunction's default_rosparams.
        """
        nr_of_optimized_params_found = 0
        for param, value in complete_rosparams.items():
            # Check if the current param is optimized
            if param in self.optimization_bounds.keys():
                nr_of_optimized_params_found += 1
                continue # Move on to the next parameter
            else: # If it's not optimized
                # check whether it has the right value (equal to the one set in default_rosparams)
                if not value == self.default_rosparams[param]:
                    return False # return False immediately

        if not nr_of_optimized_params_found == len(self.optimization_bounds):
            raise LookupError("There are parameters in the optimization_bounds, which aren't in the sample's complete rosparams.", str(self._sample_db[complete_rosparams]))

        return True

class Sample(object):
    """
    Represents one sample of the evaluation function.
    On the map matcher side, this is one run of the map matcher on the whole dataset.
    It contains all available data, though only parts of it may be used by the ObjectiveFunction that 
    is actually optimized.
    The data fields are as follows:
        * translation_errors: A list of translation errors meters per match
        * rotation_errors: A list of rotation errors in degree per match
        *---> Translation error n and rotation error n are both expected to be the result of match n.
        * parameters: A dict of all rosparams used for the map matcher run
        * time: A datetime.timedelta object, expressing the time-cost of this Sample.
        * origin: A string which describes where that sample was generated. Probably should be a path to the map matcher results directory. (for debugging)
    """

    def __init__(self):
        """
        Creates a function sample object with empty data contents.
        Fill with available data by accessing its propteries.
        All data fields are initialized with None here, so the code will crash if you use
        metrics on samples which don't have the data they need. (instead of generating wrong results)
        """
        self.translation_errors = None
        self.rotation_errors = None
        self.parameters = None
        self.time = None
        self.origin = None

    @property
    def nr_matches(self):
        """
        The number of matches the map matcher made with this Sample's parameters.
        """
        return len(self.translation_errors)

    def __str__(self):
        """
        Small, readable representation of this Sample
        """
        return "Sample generated from " + self.origin + " (database-hash: " + str(SampleDatabase.rosparam_hash(self.parameters)) + ")"

class SampleDatabase(object):
    """
    This object manages Samples in a dictionary (the "database").
    It knows nothing of the currently optimized parameters or their bounds, this functionality is covered by the ObjectiveFunction.
    The sample object themselves aren't kept in memory, but are stored on disk as pickle files.
    They'll get loaded into when requested via __getitem__ like this: sample_db_obj[rosparams_of_requested_sample].

    The database-dict is indexed by the hashed parameters (hashed using the rosparam_hash function) and contains:
        * pickle_name: The name of the pickled Sample object.
        * params: The complete rosparams dict used to generate this Sample. Its hash should be equal to the item's key.
    """

    def __init__(self, database_path, sample_dir_path, sample_generator_config):
        """
        Initializes a SampleDatabase object.

        :param database_path: Path to the database file; The pickled database dict.
        :param sample_dir_path: Path to the directory where samples created by this SampleDatabase should be stored.
        :param sample_generator_config: Config for the sample generation in the INTERFACE_MODULE.
        """
        # Error checking
        if os.path.isdir(database_path):
            raise ValueError("Given database path is a directory!", database_path)

        self._database_path = database_path
        self._sample_dir_path = sample_dir_path
        self.sample_generator_config = sample_generator_config
        if os.path.exists(self._database_path): # If file exists...
            print("\tFound existing datapase pickle, loading from:", self._database_path, end=" ")
            # ...initialize the database from the file handle (hopefully points to a pickled dict...)
            with open(database_path, 'rb') as db_handle:
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

    def create_sample_from_map_matcher_results(self, results_path, override_existing=False):
        """
        Creates a new Sample object from a finished map matcher run and adds it to the database.

        :param results_path: The path to the directory which contains the map matcher's results.
        """

        results_path = os.path.abspath(results_path)
        print("\tCreating new Sample from map matcher result at", results_path, end=".\n")
        sample = Sample()
        # This function actually fills the sample with data.
        # Its implementation depends on which map matching pipeline is optimized.
        INTERFACE_MODULE.create_objective_function_sample(results_path, sample, self.sample_generator_config)
        # Calculate the path where the new sample's pickle file should be placed
        pickle_basename = os.path.basename(results_path)
        if pickle_basename == "results": # In some cases the results are placed in a dir called 'results'
            # If that's the case, we'll use the name of the directory above, since 'results' is a bad name & probably not unique
            pickle_basename = os.path.basename(os.path.dirname(results_path))
        pickle_name = pickle_basename + ".pkl"
        # save the generated sample. (both in the db and as pkl to disk)
        self.add_sample(sample, pickle_name, override_existing)

    def add_sample(self, sample, pickle_name, override_existing=False):
        """
        Adds a new Sample to the database and saves its pickled representation to disk.

        :param sample: The Sample object itself.
        :param pickle_name: The name under which the sample should be pickled.
        :param override_existing: Whether an exception should be thrown if a sample with that name or hash already exists.
        """
        self._pickle_sample(sample, pickle_name, override_existing)
        complete_rosparams = sample.parameters
        params_hashed = SampleDatabase.rosparam_hash(complete_rosparams)
        # Safety check, don't just overwrite a db entry
        if not override_existing and params_hashed in self._db_dict.keys():
            raise LookupError("Newly created sample's hash already exists in the database! Hash:", params_hashed,\
                              "Existing sample's pickle path is:", self._db_dict[params_hashed]['pickle_name'])
        # Add new Sample to db and save the db
        print("\tRegistering sample to database at hash(params):", params_hashed)
        self._db_dict[params_hashed] = {'pickle_name': pickle_name, 'params': complete_rosparams}
        self._save()

    def remove_sample(self, sample_identifier):
        """
        Removes a Sample's entry from the database and its pickled representation from disk.

        :param sample_identifier: A string which identifies the sample.
                                  Can either be a Sample's hash or
                                  the path where its pickled representation is stored on disk.
        Returns the pickle_name, which may be useful e.g. when updating a sample.
        """

        if os.path.splitext(str(sample_identifier))[1] == '.pkl': # sample_identifier points to the pickle file
            pickle_name = sample_identifier
            # Get the sample's representation
            sample = self._unpickle_sample(pickle_name)
            # Try to find and remove the Sample's entry in the database
            if not self.exists(sample.parameters):
                print("\tWarning: Couldn't find db-entry for pickled Sample '" + pickle_name + "'.")
            else:
                print("\tRemoving Sample's db entry '" + str(self.rosparam_hash(sample.parameters)) + "'.")
                del self._db_dict[SampleDatabase.rosparam_hash(sample.parameters)]
            # Remove it from disk
            print("\tRemoving Sample's pickle '" + pickle_name + "' from disk.")
            os.remove(os.path.join(self._sample_dir_path, pickle_name))
        else: # sample_identifier should be a hash in the db
            sample_hash = int(sample_identifier) # throws ValueError, if the file doesn't exist...
            if not sample_hash in self._db_dict:
                raise LookupError("Couldn't find a sample with hash", sample_hash)
            # Get sample from pickled representation
            pickle_name = self._db_dict[sample_hash]['pickle_name']
            pickle_path = os.path.join(self._sample_dir_path, pickle_name)
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
        return pickle_name

    def update_sample(self, old_sample_identifier, updated_sample):
        """
        Updates a Sample that already exists in the database.
        This also updates its pickled representation.

        :param old_sample_identifier: A string which identifies the sample.
                                      Can either be a Sample's hash or
                                      the path where its pickled representation is stored on disk.
        :param updated_sample: The updated Sample object.
        """
        # Remove the sample in its unpatched state
        pickle_name = self.remove_sample(old_sample_identifier)
        # add the updated sample
        self.add_sample(updated_sample, pickle_name)

    def exists(self, complete_rosparams):
        """
        Returns whether a sample with the given rosparams already exists in the database.
        """

        return SampleDatabase.rosparam_hash(complete_rosparams) in self._db_dict.keys()

    def __len__(self):
        """
        Returns the total number of samples stored in this database.
        """
        return len(self._db_dict)

    def __iter__(self):
        """
        Iterator for getting all sample objects contained in the database.
        """
        for sample_hash, sample in self._db_dict.items():
            yield self._unpickle_sample(sample['pickle_name'])

    def __getitem__(self, complete_rosparams):
        """
        Returns a Sample object from the database.
        If the requested sample doesn't exist, it'll get generated via the INTERFACE_MODULE.generate_sample() function.
        This causes the function call to block until it's done. (possibly for hours)

        :param complete_rosparams: The complete rosparams dictionary (not just the ones getting optimized).
        """

        params_hashed = SampleDatabase.rosparam_hash(complete_rosparams)
        # Check if we need to generate the sample first
        if not params_hashed in self._db_dict.keys():
            # Generate a new sample and store it in the database
            print("\tNo sample with hash ", params_hashed, " in database, generating new sample:", sep="'")
            data_path = INTERFACE_MODULE.generate_sample(complete_rosparams, self.sample_generator_config)
            print("\tSample generation finished, adding it to database.")
            self.create_sample_from_map_matcher_results(data_path)
        # Get the sample's db entry
        pickle_name = self._db_dict[params_hashed]['pickle_name']
        # load the Sample from disk
        print("\tRetrieved sample ", params_hashed, " from db, loading its representation from ", pickle_name, ".", sep="'")
        extracted_sample = self._unpickle_sample(pickle_name)
        # Do a sanity check of the parameters, just in case of a hash collision
        if not complete_rosparams == extracted_sample.parameters:
            raise LookupError("Got a sample with hash " + str(params_hashed) + ", but its parameters didn't match the requested parameters. (Hash function collision?)", complete_rosparams, extracted_sample.parameters)
        return extracted_sample

    def _pickle_sample(self, sample, pickle_name, override_existing=False):
        """
        Helper method for saving samples to disk.

        :param sample: The Sample which should be pickled.
        :param pickle_name: The name of the Sample's pickled representation.
        """
        pickle_path = os.path.abspath(os.path.join(self._sample_dir_path, pickle_name))
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
        pickle_path = os.path.abspath(os.path.join(self._sample_dir_path, pickle_name))
        with open(pickle_path, 'rb') as sample_pickle_handle:
            sample = pickle.load(sample_pickle_handle)
            if not isinstance(sample, Sample):
                raise TypeError("The object unpickled from", pickle_path, "isn't an Sample instance!")
            return sample

    @classmethod
    def rosparam_hash(cls, params_dict):
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
