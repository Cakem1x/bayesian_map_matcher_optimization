#!/usr/bin/env python3

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

from performance_measures import PerformanceMeasure

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
