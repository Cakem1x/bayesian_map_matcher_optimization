#!/usr/bin/env python3

##########################################################################
# Copyright (c) 2017 German Aerospace Center (DLR). All rights reserved. #
# SPDX-License-Identifier: BSD-2-Clause                                  #
##########################################################################

"""
Classes for different ways to measure the map matcher's performance, based on information in a sample.
Only classes that end with Measure (not *Function classes) take a Sample as __call__ parameter and are meant for immediate usage with objective_function.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib

def translation_to_rotation_error(translation_error, submap_size):
    """
    Turns a translation error into a rotation error by finding the smallest rotation that could've caused the given translation error.
    For this, the submap size is used, which should be the distance to the submap's point farthest from the origin.
    Returns the estimated rotation error in degrees.
    :params translation_error: The translation error or list of translation errors.
    :params submap_size: The submap's size in the same unit as your translation errors.
    """
    rotation_error_rad = 2*np.arcsin(translation_error/(2*submap_size))
    return np.rad2deg(rotation_error_rad)

def rotation_to_translation_error(rotation_error, submap_size):
    """
    Turns a rotation error into a translation error by looking at the upper bound of translation errors that
    are caused by that rotation error.
    For this, the submap size is used, which should be the distance to the submap's point farthest from the origin.
    Returns a list of translation errors if a list of rotation errors were given.
    Otherwise, a single value will be returned. (behaves just like other numpy functions)
    :params rotation_error: The rotation error or list of rotation errors in degrees.
    :params submap_size: The submap's size in the same unit as your translation errors.
    """
    rotation_error_rad = np.deg2rad(rotation_error)
    return 2*submap_size * np.sin(rotation_error_rad/2)

class PerformanceMeasure(object):
    """
    Contains some common methods used in all PerformanceMeasures.
    Also contains a utility class function to create a PerformanceMeasure subclass from a dict.

    Other than that, it mainly exists for showing what is necessary to implement when adding a new PerformanceMeasure subclass.
    And to allow isinstance tests with the PerformanceMeasure type.
    """

    AVAILABLE_TYPES = ['LogisticTranslationErrorMeasure', 'LogisticMaximumErrorMeasure', 'MixerMeasure', 'ZeroMeanMixerMeasure', 'NrMatchesMeasure']

    def __init__(self):
        """
        Placeholder constructor, will maybe do sth in the future...
        """
        pass

    def __str__(self):
        """
        Placeholder str representation, meant for LaTeX mathmode parser in matplotlib.
        """
        return u"Performance Measure $p$"

    def __call__(self, sample):
        """
        Placeholder call method, so one could directly call an PerformanceMeasure object with a given sample.
        """
        raise RuntimeError("Shouldn't call (or instantiate...) the PerformanceMeasure superclass")
        return sample.something * 1337 # Do some magic with the sample's data

    def _prepare_plot(self, x_min, x_max, resolution=1000):
        x_space = np.linspace(x_min, x_max, resolution)
        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel(str(self))
        ax.plot(x_space, self(x_space))
        ax.set_xlim(x_min, x_max)
        return fig, ax

    def plot(self, path, x_min, x_max):
        """
        Plots the Performance with continuous input values from x_min to x_max.
        """
        fig, ax = self._prepare_plot(x_min, x_max)
        fig.savefig(path)
        fig.clf()

    @property
    def value_range(self):
        """
        The range in which this measure's values can be.
        Returns a (min_bound, max_bound) tuple.
        """
        return (0, 1)

    @classmethod
    def from_dict(cls, measure_dict):
        if measure_dict['type'] == cls.AVAILABLE_TYPES[0]: # LogisticTranslationErrorMeasure
            return LogisticTranslationErrorMeasure(max_relevant_error=measure_dict['max_relevant_error'])
        elif measure_dict['type'] == cls.AVAILABLE_TYPES[1]: # LogisticMaximumErrorMeasure
            return LogisticMaximumErrorMeasure(submap_size=measure_dict['submap_size'], max_relevant_error=measure_dict['max_relevant_error'])
        elif measure_dict['type'] == cls.AVAILABLE_TYPES[2]: # MixerMeasure
            error_measure = PerformanceMeasure.from_dict(measure_dict['error_measure'])
            matches_measure = PerformanceMeasure.from_dict(measure_dict['matches_measure'])
            return MixerMeasure(error_measure, matches_measure, measure_dict['matches_weight'])
        elif measure_dict['type'] == cls.AVAILABLE_TYPES[3]: # ZeroMeanMixerMeasure
            error_measure = PerformanceMeasure.from_dict(measure_dict['error_measure'])
            matches_measure = PerformanceMeasure.from_dict(measure_dict['matches_measure'])
            return ZeroMeanMixerMeasure(error_measure, matches_measure, measure_dict['matches_weight'])
        elif measure_dict['type'] == cls.AVAILABLE_TYPES[4]: # NrMatchesMeasure
            return NrMatchesMeasure(measure_dict['expected_nr_matches'])
        else:
            raise ValueError("Type not available", cls.AVAILABLE_TYPES)

class LogisticFunction(PerformanceMeasure):
    """
    PerformanceMeasure that uses the logistic function in its raw form:
    f(x) = 1 - l / (1 + e^(-k(x-x0)))
    with x being float.
    """
    def __init__(self, l=1, x0=0, k=1):
        """
        Initialize with setting parameters
        :param l: Maximum the function approaches for x towards infinity.
        :param x0: The point where the curve passes f(x_0)=1-(l/2), and where it has a saddle point.
        :param k: Scales curve width.
        """
        super().__init__()
        self.l = float(l)
        self.x0 = float(x0)
        self.k = float(k)

    def __str__(self):
        return u"$\\epsilon^t_{k, x_0}(e^t_i) = 1 - \\frac{" + str(int(self.l)) + u"}{1 + e^{-k (x - x_0)}}$"

    def __call__(self, x):
        return 1 - self.l / (1 + np.exp(-self.k * (x - self.x0)))

class LogisticTranslationErrorMeasure(LogisticFunction):
    """
    PerformanceMeasure that uses the logistic function to map possible translation errors between 0 and 1.
    When considering single translation errors, ones close to max_relevant_error will be mapped to 0, bigger ones to negative numbers approaching -1.
    The measure will return the sum of all of logistic_func(err_i) values, but will make sure the lowest value it returns is 0.
    By allowing single, big outlier errors to be mapped to values smaller than 0, they can be optimized against more easily.
    """
    def __init__(self, max_relevant_error, min_relevant_error=0.05):
        """
        Initialize with setting parameters
        :param max_relevant_error: The maximum translation error that should still be distinguishable from higher errors. (i.e. mapped not too close to 0)
        :param min_relevant_error: Same for the minimum, defaults to 0.05.
        """
        self.x_min_y_value = 0.95 # determines how close to 1 min_relevant_error gets mapped (max_relevant error always gets mapped to 0)
        self.min_relevant_error = float(min_relevant_error)
        self.max_relevant_error = float(max_relevant_error)
        l = 2 # set l to 2, so the function goes from 1 to -1 with the strongest slope at f(max_relevant_error) = 0.
        x0 = self.max_relevant_error
        # Find k by using point (min_relevant_error, x_min_y_value)
        # Solved logistic function for k:
        k = - (np.log(-(l / (self.x_min_y_value - 1)) - 1)) / (self.min_relevant_error - self.max_relevant_error)
        super().__init__(l, x0, k)

    def __call__(self, sample):
        if isinstance(sample, np.ndarray) or isinstance(sample, float): # Special case for plotting the function
            return super(LogisticTranslationErrorMeasure, self).__call__(sample)

        # Put each translation error through the Logistic function (super().__call__)
        match_errors = [super(LogisticTranslationErrorMeasure, self).__call__(err_t) for err_t in sample.translation_errors]
        # Guard against crash if no matches were made; Return 0 in that case
        if not sample.nr_matches == 0:
             # Sum them up and normalize with the number of matches
             # max() guard against negative measure value in case all errors were too big
            return max(0, sum(match_errors, 0) / sample.nr_matches)
        else:
            return 0

    def plot(self, path, x_min, x_max):
        fig, ax = self._prepare_plot(x_min, x_max)
        ax.set_xlabel(u"$e^t_i$")
        ax.scatter([self.min_relevant_error, self.max_relevant_error],
                   [self(self.min_relevant_error), self(self.max_relevant_error)],
                   c='r', label=u"${(" + str(round(self.min_relevant_error, 2)) + u"," + str(round(self(self.min_relevant_error), 2)) + "),(" + str(round(self.max_relevant_error, 2)) + u"," + str(round(self(self.max_relevant_error), 2)) + u")}$")
        ax.axvline(linestyle='dashed', color='gray', x=self.min_relevant_error)
        ax.axhline(linestyle='dashed', color='gray', y=self(self.min_relevant_error))
        ax.axvline(linestyle='dashed', color='gray', x=self.max_relevant_error)
        ax.axhline(linestyle='dashed', color='gray', y=self(self.max_relevant_error))
        ax.set_ylim(-1,1)
        plt.tight_layout()
        fig.savefig(path)
        fig.clf()

class MixerMeasure(PerformanceMeasure):
    """
    Mixes two measures, one for the nr of matches and one for the errors.
    """

    def __init__(self, error_measure, matches_measure, matches_weight):
        """
        Initialize with setting parameters
        :param error_measure: Performance measure to measure the error(s) of a sample.
        :param matches_measure: Performance measure to measure the number of matches of a sample.
        :param matches_weight: The weight of matches_measure in [0,1]. The error_measure's weight will be 1-matches_weight.
        """
        super().__init__()
        self.error_measure = error_measure
        self.matches_measure = matches_measure
        self.matches_weight = matches_weight

    @property
    def error_weight(self):
        """
        The weight of the error measure in [0,1].
        """
        return 1 - self.matches_weight

    def __str__(self):
        #return u"$p(\\mathbf{e^t}, \\mathbf{e^r}, m) = " + str(self.matches_weight) + u" \\cdot \\upsilon(m) + " + str(self.error_weight) + u" \\cdot \\epsilon(\\mathbf{e^t}, \\mathbf{e^r})$"
        return u"$p(\\mathbf{e^t}, \\mathbf{e^r}, m)$"

    def __call__(self, sample):
        return self.error_weight * self.error_measure(sample) + self.matches_weight * self.matches_measure(sample)

class ZeroMeanMixerMeasure(MixerMeasure):
    """
    Same as class MixerMeasure, but doesn't output a value from 0 to 1.
    Instead, its value-range is translated so it is between -0.5 and 0.5.
    May yield better results if the objective function's surrogate model assumes zero mean for unseen areas.
    """

    def __init__(self, error_measure, matches_measure, matches_weight):
        """
        Initialize with setting parameters
        :param error_measure: Performance measure to measure the error(s) of a sample.
        :param matches_measure: Performance measure to measure the number of matches of a sample.
        :param matches_weight: The weight of matches_measure in [0,1]. The error_measure's weight will be 1-matches_weight.
        """
        super().__init__(error_measure, matches_measure, matches_weight)

    def __call__(self, sample):
        return super().__call__(sample) - 0.5

    @property
    def value_range(self):
        """
        The range in which this measure's values can be.
        Returns a (min_bound, max_bound) tuple.
        """
        return (-0.5, 0.5)

class LogisticMaximumErrorMeasure(LogisticTranslationErrorMeasure):
    """
    Based on class LogisticTranslationErrorMeasure.
    This PerformanceMeasure considers both translation and rotation errors.
    Rotation errors will be turned into translation errors:
        By considering the maximum translation error that could've been caused by the rotation
        around the submap's origin with a given distance to the point farthest away from the origin. (submap size)
    Then, the maximum of the translation error and the "turned-to-translation"-rotation error will be put into the LogisticTranslationErrorMeasure.
    """
    def __init__(self, submap_size, max_relevant_error):
        """
        Initialize with setting parameters
        :param submap_size: The size of each submap, used for turning rotation errors into translation errors.
        :param max_relevant_error: The translation error that will be mapped to 0, higher ones will get negative values, lower ones positive values.
        """
        self.submap_size = submap_size
        super().__init__(max_relevant_error)

    def __call__(self, sample):
        if isinstance(sample, np.ndarray) or isinstance(sample, float): # Special case for plotting the function
            return super(LogisticMaximumErrorMeasure, self).__call__(sample)

        # Iterate over rotation and translation error lists simultaneously and chose the biggest errors
        considered_errors = [max(err_t, err_r) for err_t, err_r in
                             zip(sample.translation_errors,
                                 rotation_to_translation_error(sample.rotation_errors, self.submap_size))]
        match_errors = [super(LogisticMaximumErrorMeasure, self).__call__(err) for err in considered_errors]
        # Guard against crash if no matches were made; Return 0 in that case
        if not sample.nr_matches == 0:
            normalized_error_sum = sum(match_errors, 0) / sample.nr_matches
            return max(0, normalized_error_sum) # guard against returning negative measure if all errors were really big
        else:
            return 0

class NrMatchesMeasure(PerformanceMeasure):
    """
    Uses the function x / (a + x) to map the number of matches to [0,1).
    More matches means this measure gets closer to 1.
    """
    def __init__(self, expected_nr_matches):
        """
        :param expected_nr_matches: A number of matches which is already considered quite good for the dataset
                                    Will be used to calculate a, so that the function passes point (expected_nr_matches, 0.98)
        """
        self.expected_nr_matches = int(expected_nr_matches)
        # Can be used to tune how much space the measure has left for matches above the expected_nr_matches threshold
        self.expected_nr_matches_y_value = 0.9
        self.a = self.expected_nr_matches * (1 - self.expected_nr_matches_y_value) / self.expected_nr_matches_y_value

    def __str__(self):
        return u"$\\upsilon_a(m) = \\frac{m}{a + m}$"

    def __call__(self, sample):
        if isinstance(sample, np.ndarray) or isinstance(sample, int): # Special case for plotting the function
            return sample / (self.a + sample)

        return sample.nr_matches / (self.a + sample.nr_matches)

    def plot(self, path, x_min, x_max):
        fig, ax = self._prepare_plot(x_min, x_max)
        ax.set_xlabel(u"$m$")
        ax.scatter([self.expected_nr_matches],
                   [self(self.expected_nr_matches)],
                   c='r', label=u"${(" + str(round(self.expected_nr_matches, 2)) + u"," + str(round(self(self.expected_nr_matches), 2)) + ")}$")
        ax.axvline(linestyle='dashed', color='gray', x=self.expected_nr_matches)
        ax.axhline(linestyle='dashed', color='gray', y=self(self.expected_nr_matches))
        ax.set_ylim(0,1)
        plt.tight_layout()
        fig.savefig(path)
        fig.clf()

class SinusTestFunction(PerformanceMeasure):
    """
    PerformanceMeasure that can be used for testing without any samples.
    It simply models sin(x)=a*x*sin(b*x), with x being a float.
    """
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def __str__(self):
        return u"$" + str(self.a) + u"x * \sin(" + str(self.b) + u"x)$"

    def __call__(self, x):
        return self.a * x * np.sin(self.b * x)

if __name__ == '__main__': # don't execute when module is imported
    import argparse # for the cmd-line interface
    matplotlib.rcParams['axes.labelsize'] = 25
    matplotlib.rcParams['xtick.labelsize'] = 16
    matplotlib.rcParams['ytick.labelsize'] = 16
    matplotlib.rcParams['legend.fontsize'] = 18

    def error_multi_plot(path, max_relevant_errors, x_min, x_max, resolution=1000):
        x_space = np.linspace(x_min, x_max, resolution)
        fig, ax = plt.subplots()
        ax.set_xlabel(u"$e^t_i$")
        for max_relevant_error in max_relevant_errors:
            measure = LogisticTranslationErrorMeasure(max_relevant_error)
            ax.plot(x_space, measure(x_space), label=u"$k=" + str(round(measure.k, 2)) + u",$ $x_0=" + str(round(measure.x0,2)) + u"$")
        ax.axhline(0, color='black') # line at y=0
        ax.axvline(0, color='black') # line at x=0
        ax.set_ylabel(str(measure))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1,1)
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(path)
        fig.clf()

    def nr_matches_multi_plot(path, expected_nr_matches, x_min, x_max, resolution=1000):
        measures = [NrMatchesMeasure(e_m) for e_m in expected_nr_matches]
        x_space = np.linspace(x_min, x_max, resolution)
        fig, ax = plt.subplots()
        ax.set_xlabel(u"$m$")
        ax.set_ylabel(str(measures[0]))
        for measure in measures:
            ax.plot(x_space, measure(x_space), label=u"$a=" + str(round(measure.a, 2)) + u"$")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0,1)
        ax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(path)
        fig.clf()

    def command_line_interface():
        parser = argparse.ArgumentParser(description="tests performance measures")
        parser.add_argument('path', help="Path for plot")
        parser.add_argument('type', help="Type of plot to create")
        parser.add_argument('--min', help="x_min for plot range", required=True, type=float)
        parser.add_argument('--max', help="x_max for plot range", required=True, type=float)
        parser.add_argument('params', help="list of params, depends on chosen type", nargs='*')
        args = parser.parse_args()
        
        if args.type == 'logistic_raw':
            print("Logistic measure with x0 =", args.params[0], "and k =", args.params[1])
            l = LogisticFunction(x0 = args.params[0], k=args.params[1])
            l.plot(args.path, args.min, args.max)
        elif args.type == 'logistic_tra':
            print("Logistic translation measure with max_relevant_error =", args.params[0], "min_relevant_error =", args.params[1])
            l = LogisticTranslationErrorMeasure(max_relevant_error = args.params[0], min_relevant_error = args.params[1])
            l.plot(args.path, args.min, args.max)
        elif args.type == 'nr_matches':
            print("Measure for nr matches with expected nr matches =", args.params[0])
            l = NrMatchesMeasure(args.params[0])
            l.plot(args.path, args.min, args.max)
        elif args.type == 'multiplot_matches':
            print("Multiplot for nr matches with expected_nr_matches", args.params)
            nr_matches_multi_plot(args.path, args.params, args.min, args.max)
        elif args.type == 'multiplot_errors':
            print("Multiplot for errors with max_relevant_errors", args.params)
            error_multi_plot(args.path, args.params, args.min, args.max)
        else:
            print("unknown type")

    command_line_interface()
