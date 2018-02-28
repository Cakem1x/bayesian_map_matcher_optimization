#!/usr/bin/env python3

##########################################################################
# Copyright (c) 2017 German Aerospace Center (DLR). All rights reserved. #
# SPDX-License-Identifier: BSD-2-Clause                                  #
##########################################################################

"""
Contains the code for modelling an objective function via discrete samples.
The objective function is the interface between the Bayesian optimization modules and the evaluation modules.

The evaluation modules supply the discrete samples that define the objective function (see sample_sources.py).
To supply a good interface to the Bayesian optimization modules, the objective function uses a performance measure that
maps samples on [0,1] (see performance_measures.py).
Additionally, the objective function is used to check and enforce that the optimization bounds are satisfied.

Since the evaluation process may require more parameters than currently optimized, the evaluation function differentiates between complete_params and optimized_params.
complete_params contains all parameters necessary for the evaluation process.
optimized_params contains only those parameters that are currently being optimized by the Bayesian optimization modules.
To get from a specific set of optimized_params to the complete_params, default_params are used to determine the values of all non-optimized parameters.
"""

from .performance_measures import PerformanceMeasure

class ObjectiveFunction(object):
    """
    Basic ObjectiveFunction implementation, see module documentation for more details.
    Its main interface method is evaluate(optimized_params), which returns the objective function's value (in [0,1])
    at the point defined by optimized_params.
    To do that, it translates the optimized_params into complete_params by using the default_params.
    The complete_params are used to get a sample from the sample_source that contains evaluation results.
    Finally, the sample is rated with the performance_measure, to get a value in [0,1].

    To supply a smooth interface between the Bayesian optimization modules and the evaluation modules,
    the objective function is able to round float values to a specified precision. This allows re-using
    previously generated samples if SampleDatabase is used as sample_source.
    Also, the function will typecast each optimized_params to the type defined in default_params.
    This is useful in case the Bayesian optimization modules use different types (e.g. numpy float instead of standard float).
    However, this will only work as long as the types are somewhat compatible.
    """

    def __init__(self, sample_source, performance_measure, default_params, design_space, rounding_decimal_places=0, normalization=True):
        """
        Creates an ObjectiveFunction object.
        
        :param sample_source: A sample source object to get discrete samples (via __getitem__) that make up this objective function.
        :param performance_measure: A performance measure object to map a sample to [0,1] (via __call__).
        :param default_params: A dict that contains the complete set of parameters required for running the evaluation process of the sample_source.
                               The values of this parameter set will be used for all non-optimized parameters.
                               Additionally, this parameter set can be used as a reference in some plots.
        :param design_space: A dict that defines what parameters are to be optimized within which bounds
                             Its keys are used to determine the set of optimized_params.
                             The dict's values are (min, max)-tuples to define the value bounds of the respective parameter.
        :param rounding_decimal_places: The number of decimal places to which parameters of type float should be rounded to.
                                        If zero, no rounding will take place.
                                        This is useful in case a SampleDatabase is used as sample_source:
                                        Without rounding, every sample would have slightly different (float-)values.
        :param normalization: Set to False if the optimization modules aren't working on normalized values.
                              If True, the objective function will denormalize requests it gets from the optimization modules before passing
                              them to the evaluation modules.
        """

        # error checking
        print("Initializing the objective function:")
        if not isinstance(performance_measure, PerformanceMeasure):
            raise ValueError("Given performance_measure is not an instance of PerformanceMeasure.")

        self.performance_measure = performance_measure
        self.sample_source = sample_source
        self.default_params  = default_params 
        self.design_space = design_space
        self._rounding_decimal_places = rounding_decimal_places
        self._normalization = normalization
        if not self._rounding_decimal_places == 0:
            print("\tWill round floating parameters to", self._rounding_decimal_places, "decimal places."\
                  " (e.g. 0.12918318241288 to", round(0.12918318241288, self._rounding_decimal_places), ")")
        if self._normalization:
            print("\tWill normalize parameter values within the bounds given by the design space.")
        else:
            print("\tWill not normalize parameters, this may degenerate optimization performance.")

    def evaluate(self, **optimized_params):
        """
        This method supplies the interface for the Bayesian optimization modules.
        It calculates and returns the objective function's value ("y") at the point given by optimized_params ("X").

        Will convert the requested optimized_params (from the Optimizer world) to complete_params, which defines
        a Sample in the evaluation modules.
        This may also involve casting types of optimized_params to the type the parameter has in the default_params
        (see preprocess_optimized_params).

        If a new sample needs to be generated, this call will block until the evaluation modules have finished generating the requested sample.

        :param optimized_params: A keyworded argument list (used as a dictionary with parameter names as keys).
        """

        # Preprocess the parameters
        self.preprocess_optimized_params(optimized_params)
        print("\033[1;34mSampling objective function at:")
        if self._normalization:
            # denormalize parameters
            normalized_parameters = optimized_params.copy()
            print("\n\t >> Denormalizing parameters! << ", end="")
            self.denormalize_parameters(optimized_params)
        for name, value in optimized_params.items():
            print() # newline
            print("\t", name, " = ", value, sep="", end="")
            if self._normalization:
                print(" (norm. val = ", normalized_parameters[name], ")", sep="", end="")
        print("\033[0m")
        # Create the full set of parameters by updating the default parameters with the optimized parameters.
        complete_params = self.default_params.copy()
        complete_params.update(optimized_params)
        # Get the sample from the sample source
        sample = self.sample_source[complete_params]
        # Calculate and return the metric
        value = self.performance_measure(sample)
        print("\033[1;34m\tSample's performance measure:\033[1;37m", value, "\033[0m")
        return value

    def normalize_parameters(self, optimized_params):
        """
        Takes a dict of optimized parameters as given by the evaluation modules and normalizes the values.

        This method uses the design_space member.
        The process is inverse to the denormalization method and should be used on all parameters that 
        come from the evaluation part of the system and are to be passed into the Bayesian optimization.

        :param optimized_params: The params dict, from the evaluation modules, where parameter values are in [min_bound, max_bound].
        :returns: The normalized params dict for the optimization modules, where parameter values are in [0,1].
        """
        normalized_params = optimized_params.copy()
        for rosparam_name, bounds in self.design_space.items():
            old_val = normalized_params[rosparam_name] # TODO remove after test
            param_range = bounds[1] - bounds[0]
            normalized_params[rosparam_name] = (normalized_params[rosparam_name] - bounds[0]) / param_range
            print("Normalizing", rosparam_name, "from", old_val, "to", normalized_params[rosparam_name]) # TODO remove after test

        return normalized_params

    def denormalize_parameters(self, optimized_params):
        """
        Takes a dict of optimized parameters as given by the optimization modules and denormalizes the values.

        The process is inverse to the normalization method and should be used on all parameters that 
        come from optimization modules to be passed into evaluation modules.
        Also, the denormalized values are probably more informative for user output as well.

        :param optimized_params: The normalized params dict from the optimization modules, where parameter values are in [0,1].
        :returns: The params dict, from the evaluation modules, where parameter values are in [min_bound, max_bound].
        """
        for rosparam_name, bounds in self.design_space.items():
            old_val = optimized_params[rosparam_name] # TODO remove after test
            param_range = bounds[1] - bounds[0]
            optimized_params[rosparam_name] = (optimized_params[rosparam_name] * param_range) + bounds[0]
            print("Denormalizing", rosparam_name, "from", old_val, "to", optimized_params[rosparam_name]) # TODO remove after test

        return optimized_params

    def preprocess_optimized_params(self, optimized_params):
        """
        Takes a dict of optimized parameters as given by the optimizer modules and preprocesses it to fit the evaluation modules.
        
        This includes a safety check to make sure the parameters requested by the optimizer don't violate the optimization bounds.
        Additionally, all parameter types in the given dict will get casted to their respective type in default_params.
        Otherwise, dumping them to a yaml file would (for example) create binarized numpy.float64 values, which possibly can't be parsed by the evaluation modules.
        Also, values will get rounded according to the _rounding_decimal_places member.

        :param optimized_params: The params dict, as requested by the optimizer.
        :returns: The preprocessed params dict, as needed by the ros ecosystem.
        """
        # Iterate over the optimization bounds and check if the current request doesn't violate them
        for rosparam_name, bounds in self.design_space.items():
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
            if not p_name in self.design_space.keys():
                raise ValueError(str(p_name) + " shouldn't get optimized, but was in given dict of optimized parameters.")
            # ...also CAST their type to the type in the default rosparams dict. Otherwise, we may serialize
            # some high-precision numpy float class instead of having a built-in float value on the yaml, that
            # rosparams can actually read. Sadl, we'll lose some precision through that.
            if type(self.default_params [p_name]) != type(p_value):
                #print("\tWarning, casting parameter type", type(p_value), "of", p_name, "to", type(self.default_params [p_name]))
                optimized_params[p_name] = type(self.default_params [p_name])(p_value)
            # ...also ROUND float values according to rounding_decimal_places parameter
            if self._rounding_decimal_places and isinstance(p_value, float):
                rounded_p_value = round(optimized_params[p_name], self._rounding_decimal_places)
                #print("\tWarning, rounding float value", p_name, ":", p_value, "->", rounded_p_value)
                optimized_params[p_name] = rounded_p_value

        return optimized_params

    def samples_filtered(self, fixed_params, enforce_bounds=False):
        """
        Iterator that yields only samples that satisfy all fixed_params definitions.
        This method requires a sample_source like the SampleDatabase, which implements the __iter__ method.

        :param fixed_params: A dict that maps a subset of optimized_params to a desired value.
        Returns samples in the same format as __iter__.
        :param enforce_bounds: If set to True, only samples are returned which also lie in the current design_space.
        """
        for x, y, s in self:
            # For each sample, check if it's usable:
            usable = True
            for p_name, p_value in x.items():
                if p_name in fixed_params: # For all fixed_params, check if the value is correct
                    if not self.default_params [p_name] == p_value:
                        usable = False
                        break
                else: # For all non-fixed-params: Check if the value is in the optimization bounds
                    if p_value < self.design_space[p_name][0] or p_value > self.design_space[p_name][1]:
                        usable = False
                        break
            if usable:
                yield x, y, s

    def __iter__(self):
        """
        Iterator for getting information from all available samples from the database, which define this ObjectiveFunction.
        This method requires a sample_source like the SampleDatabase, which implements the __iter__ method.

        This means only information from those Samples are yielded, which have parameter values matching the ones in default_params .
        Only parameter values of currently optimized parameters are allowed to differ.
        The filtering is done in the _defined_by function.
        
        Yields tuples (x, y, s), with
            x: A dict of the complete rosparams that were used to create that sample.
            y: The sample's value as given by the current performance measure.
            s: The sample itself, in case other information needs to be extracted.
        """
        for sample in self.sample_source:
            if self._defined_by(sample.parameters):
                yield sample.parameters, self.performance_measure(sample), sample

    def _defined_by(self, complete_params):
        """
        Returns whether the given complete_params is valid for defining this ObjectiveFunction.
        That's the case if all non-optimized parameters of complete_params are equal to this ObjectiveFunction's default_params.
        """
        nr_of_optimized_params_found = 0
        for param, value in complete_params.items():
            # Check if the current param is optimized
            if param in self.design_space.keys():
                nr_of_optimized_params_found += 1
                continue # Move on to the next parameter
            else: # If it's not optimized
                # check whether it has the right value (equal to the one set in default_params )
                if not value == self.default_params [param]:
                    return False # return False immediately

        if not nr_of_optimized_params_found == len(self.design_space):
            raise LookupError("There are parameters in design_space, which aren't in the sample's complete_params.", str(self.sample_source[complete_params]))

        return True
