#! /usr/bin/env python3
##########################################################################
# Copyright (c) 2017 German Aerospace Center (DLR). All rights reserved. #
# SPDX-License-Identifier: BSD-2-Clause                                  #
##########################################################################

# Local imports
import objective_function
import performance_measures

# Foreign packages
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import axes3d # required for 3d plots
import numpy as np
import os
import rosparam
import yaml
import sys
import shutil # for removing full filetrees
import itertools
from sklearn.gaussian_process.kernels import Matern
from bayes_opt import BayesianOptimization

colors = {'orange': '#FDB462',
          'yellow': '#FFFFB3',
          'red': '#FB8072',
          'green': '#8DD3C7',
          'blue': '#80B1D3',
          'violet': '#BEBADA'}

class ExperimentCoordinator(object):
    """
    Can be thought of as the 'toplevel' class for this repo.
    Will bring together the different parts of the system and manage the information flow between them.
    
    Much code concerns reading / fixing paths, putting together the right initialization params for the used modules (mostly in __init__).
    Another big part is code for plotting results.

    For the plotting and all user-interface-related manners, the parameters will be described by their "display_name", i.e. their key in the experiment's yaml file.
    The code in the objective_function module doesn't know about the display_name and only uses the rosparam_name.
    """

    def __init__(self, params_dict, relpath_root):
        """
        :param params_dict: A dictionary which contains all parameters needed to define an experiment.
        :param relpath_root: Basepath for all (other) relative paths in the params_dict.
        """
        self.iteration = 0 # Holds the current iteration's number
        # Set before iterating to use the fine_tune kappa value (should be lower, to focus on exploitation)
        self.fine_tune = False
        # Stores all discovered "best samples" during the experiment as tuples of (iteration, sample)
        self.best_samples = list()
        self.max_performance_measure = 0 # Holds the currently best known performance measure score
        self._params = params_dict
        self._relpath_root = relpath_root

        if "rng_seed" in self._params.keys():
            # Set seed of numpy random number generator
            print("Setting RNG seed to", self._params["rng_seed"])
            np.random.seed(self._params["rng_seed"])

        # Resolve relative paths
        self._params['plots_directory'] = self._resolve_relative_path(self._params['plots_directory'])
        database_path = self._resolve_relative_path(self._params['database_path'])
        sample_directory_path = self._resolve_relative_path(self._params['sample_directory'])
        rosparams_path = self._resolve_relative_path(self._params['default_rosparams_yaml_path'])
        sample_generator_config = self._params['sample_generator_config']
        sample_generator_config['evaluator_executable'] = self._resolve_relative_path(sample_generator_config['evaluator_executable'])
        sample_generator_config['environment'] = self._resolve_relative_path(sample_generator_config['environment'])
        sample_generator_config['dataset'] = self._resolve_relative_path(sample_generator_config['dataset'])

        ###########
        # Setup the SampleDatabase object
        ###########
        print("Setting up Sample database...")
        # Create the sample db
        self.sample_db = objective_function.SampleDatabase(database_path, sample_directory_path, sample_generator_config)

        ###########
        # Setup the evaluation function object
        ###########
        print("Setting up ObjectiveFunction...")
        self.optimization_defs = self._params['optimization_definitions']
        default_rosparams = rosparam.load_file(rosparams_path)[0][0]
        self.performance_measure = performance_measures.PerformanceMeasure.from_dict(self._params['performance_measure'])
        self.eval_function = objective_function.ObjectiveFunction(self.sample_db, default_rosparams,
                                                                    self.opt_bounds,
                                                                    self.performance_measure,
                                                                    self._params['rounding_decimal_places'])
        ###########
        # Create an BayesianOptimization object, that contains the GPR logic.
        # Will supply us with new param-samples and will try to model the map matcher metric function.
        ###########
        print("Setting up Optimizer...")
        # Create the optimizer object
        self.optimizer = BayesianOptimization(self.eval_function.evaluate, self.opt_bounds, verbose=0)
        # Create a kwargs member for passing to the maximize method (see iterate())
        # Those parameters will be passed to the GPR member of the optimizer
        gpr_params = {'alpha': 1e-10, 'matern_nu': 2.5} # set default parameters
        if 'gpr_params' in self._params:
            gpr_params.update(self._params['gpr_params']) # update all fields to the values from the config file (fields undefined in the config will remain at the default value set above)
        # Build gpr_kwargs dict for further usage
        self.gpr_kwargs = {'alpha': gpr_params['alpha'], 'kernel': Matern(nu=float(gpr_params['matern_nu']))}

    def initialize_optimizer(self, use_previous_observations=False, only_nonzero_observations=False):
        """
        Method for initializing the optimizer with a set of observations.
        The method will automatically initialization observations from the parameters defined via the experiment.yaml file.
            --> via 'optimizer_initialization', which defines a list of optimized_params dicts.
        Initialization samples will be generated if they don't exist.

        :param use_previous_observations: If this parameter is set to True, the optimizer will be initialized with all samples in the sample database that fit the current optimizer definitions.
        """
        # Get the initialization samples from the ObjectiveFunction
        print("\tInitializing optimizer at", self._params['optimizer_initialization'])
        # init_dict will store the initialization data in the format the optimizer likes:
        # A list for each parameter with their values plus a 'target' list for the respective result value
        init_dict = {p_name: [] for p_name in self._params['optimizer_initialization'][0]}
        # Fill init_dict with values from the optimizer initialization:
        for optimized_rosparams in self._params['optimizer_initialization']:
            for p_name, p_value in optimized_rosparams.items():
                init_dict[p_name].append(p_value)
        # If desired, previously generated observations will be used as initialization as well.
        if use_previous_observations:
            print("\tUsing previous observations to initialize the optimizer.")
            for sample_complete_rosparams, y, s in self.eval_function: # gets the params dict of each sample
                if y > 0 or not only_nonzero_observations:
                    # only get values of optimized params; requires at least one initialization value defined in the experiment.yaml, but that should be the case anyway.
                    for optimized_rosparam in self._params['optimizer_initialization'][0].keys():
                        init_dict[optimized_rosparam].append(sample_complete_rosparams[optimized_rosparam])
        # Add the initilizations via the BayesianOptimization framework's explore method
        self.optimizer.explore(init_dict)

    def _setup_metric_axis(self, axis, param_display_name):
        """
        Helper method which sets up common properties of an axis object.

        :param axis: The axis object that needs to be set up
        :param param_display_name: The display name of the parameter, as given in the experiment's yaml file.
        """
        axis.set_xlim(self.eval_function.optimization_bounds[self._to_rosparam(param_display_name)])
        axis.set_ylim(self.performance_measure.value_range)
        axis.set_xlabel(param_display_name)
        axis.set_ylabel(str(self.performance_measure))

    def output_sampled_params_table(self):
        """
        Creates a markdown file sampled_params.md which contains a markdown-table of the sampled parameters.
        Resulting table is supposed to look something like this:
        | Param 1 | Param 2 | Param 3 |
        | -------:| -------:| -------:|
        |  0.2912 |   1.231 |   23.12 |
        |  0.1422 |   1.914 |   2.182 |
        |  0.7811 |   2.111 |   12.29 |
        Formatting may break in the current implementation, if parameter values are bigger (i.e. not rounded)
        then the longest display name size.
        """
        # get length of longest display name
        max_length = max([len(display_name) for display_name in self.optimization_defs.keys()])
        left_sep = "| "
        right_sep = " |"
        center_sep = " | "
        with open("sampled_params.md", 'w') as table_file:
            # Write table headers
            table_file.write(left_sep)
            for i, display_name in enumerate(self.optimization_defs.keys()):
                table_file.write(display_name.rjust(max_length, ' ')) # rjust fills string with spaces
                # write center or right separator, depending on whether we're at the last element
                table_file.write((center_sep if not i == len(self.optimization_defs.keys()) - 1 else right_sep))
            # Write table header separator
            table_file.write('\n' + left_sep)
            for i in range(len(self.optimization_defs)):
                # the colon position defines alignment of column text, in this case to the right
                table_file.write('-' * max_length + (":| " if not i == len(self.optimization_defs.keys()) - 1 else ":|"))
            # For each sample, create a row
            for x in self.optimizer.X:
                # Write sample's row
                table_file.write('\n' + left_sep)
                for i, display_name in enumerate(self.optimization_defs.keys()):
                    param_value = round(x[self._to_optimizer_id(display_name)], self._params['rounding_decimal_places'])
                    table_file.write(str(param_value).rjust(max_length, ' '))
                    # write center or right separator, depending on whether we're at the last element
                    table_file.write((center_sep if not i == len(self.optimization_defs.keys()) - 1 else right_sep))

    def plot_error_distribution(self, plot_name, sample, max_rotation_error, max_translation_error, iteration=None):
        """
        Saves a plot with the error distribution of the given sample as a violin plot.
        """
        if iteration is None:
            iteration = self.iteration
        fig, axes = plt.subplots(ncols=2)
        # Create the violin plots on the axes
        if sample.nr_matches > 1:
            v_axes = (axes[0].violinplot(sample.rotation_errors, points=100, widths=0.7, bw_method=0.5,
                                         showmeans=True, showextrema=True, showmedians=False),
                      axes[1].violinplot(sample.translation_errors, points=100, widths=0.7, bw_method=0.5,
                                         showmeans=True, showextrema=True, showmedians=False))
            for i, axis in enumerate(v_axes):
                plot_body = axis["bodies"][0]
                plot_body.set_facecolor('blue' if i == 0 else 'yellow')
                plot_body.set_edgecolor('black')
        else: # Special case for when only one match happened
            axes[0].scatter([1], sample.rotation_errors)
            axes[1].scatter([1], sample.translation_errors)
        # Set titles
        fig.suptitle("Iteration " + str(iteration) + ", " + str(sample.nr_matches) + " matches.")
        axes[0].set_title("Rotation Errors")
        axes[1].set_title("Translation Errors")
        # Set x-axis limits
        axes[0].set_yticks(np.linspace(0, max_rotation_error, 10))
        axes[1].set_yticks(np.linspace(0, max_translation_error, 10))
        # Save and close
        path = os.path.join(self._params['plots_directory'], plot_name)
        fig.savefig(path)
        print("\tSaved error distribution plot to", path)
        plt.close()

    def query_points_plot(self):
        """
        Creates a Parallel Coordinate Plot (PCP) to visualize the locations in parameter-space
        at which the optimizer chose to place its queries.
        Each parallel axis represents one paramater and each line represents one sample (or query point).

        Method only supports plots with three or more optimized parameters.

        Based on code from
        https://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
        """
        # create a temporary throwaway contourplot, so creating a colorbar further down the line is possible
        hack = plt.contourf([[0,0],[0,0]], np.linspace(0, 1, 11), cmap='jet')
        colormap = plt.get_cmap('jet')
        plt.clf() # throw away the contourf

        x = range(len(self.optimizer.X[0]))
        fig, axes = plt.subplots(1, len(x)-1, sharey=False)
        
        # Holds tuples (min, max, range) for each parameter
        paramspace_bounds = dict()
        for param in self.optimization_defs.keys():
            # Get the definitions of the current param
            defs_dict = self.optimization_defs[param]
            # Create tuple (min, max, range) for normalizing data
            mn = float(defs_dict['min_bound'])
            mx = float(defs_dict['max_bound'])
            paramspace_bounds[param] = (mn, mx, mx-mn)

        # Acquire and normalize data, ordered exactly like the keys of paramspace_bounds
        normalized_data = list()
        normalized_data_performance = list()
        for sample, performance in zip(self.optimizer.X, self.optimizer.Y):
            normalized_sample_data = list()
            for param, bounds in paramspace_bounds.items():
                val = sample[self._to_optimizer_id(param)]
                normalized_sample_data.append((val - bounds[0]) / bounds[2])
            normalized_data.append(normalized_sample_data)
            normalized_data_performance.append(performance)

        # Plot the data on the subplots
        for ax_id, ax in enumerate(axes):
            for data, performance in zip(normalized_data, normalized_data_performance):
                ax.plot(x, data, color=colormap(performance))
            ax.set_xlim([x[ax_id], x[ax_id+1]])

        # Set the x axis ticks
        for (paramspace_bound, ax,xx) in zip(paramspace_bounds.items(), axes, x[:-1]):
            ax.xaxis.set_major_locator(ticker.FixedLocator([xx]))
            ticks = len(ax.get_yticklabels())
            labels = list()
            step = paramspace_bound[1][2] / (ticks - 1)
            mn   = paramspace_bound[1][0]
            for i in range(ticks):
                v = mn + i*step
                labels.append('%4.2f' % v)
            ax.set_yticklabels(labels)
            # set labels
            ax.set_xlabel(paramspace_bound[0])
            y_coord = xx%2 # Alternate between top and bottom
            OFFSET_FROM_FIG = 0.05
            y_coord += (OFFSET_FROM_FIG + 0.025) if xx%2 == 1 else -OFFSET_FROM_FIG # offset away from the rest of the figure
            # Alternate extra offset so labels don't overlap
            OFFSET_FROM_LABELS = 0.025
            y_coord += OFFSET_FROM_LABELS if (xx%4) > 1 else -OFFSET_FROM_LABELS
            ax.xaxis.set_label_coords(0, y_coord) # x_coord is 0, since those coordinates are relative to the ax
            ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # turn off the normal xticks

        # add a color bar 
        #fig.colorbar(hack, ax=axes.ravel().tolist(), label=str(self.performance_measure)) # currently the positioning is broken

        # Move the final axis' ticks to the right-hand side
        ax = plt.twinx(axes[-1])
        ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
        ticks = len(ax.get_yticklabels())
        step = list(paramspace_bounds.values())[-1][2] / (ticks - 1)
        mn   = list(paramspace_bounds.values())[-1][0]
        labels = ['%4.2f' % (mn + i*step) for i in range(ticks)]
        ax.set_yticklabels(labels)
        # set label
        ax.set_xlabel(list(paramspace_bounds.keys())[-1])
        y_coord = x[-1]%2 # Alternate between top and bottom
        y_coord += (OFFSET_FROM_FIG + 0.025) if x[-1]%2 == 1 else -OFFSET_FROM_FIG # offset away from the rest of the figure
        y_coord += OFFSET_FROM_LABELS if (x[-1]%4) > 1 else -OFFSET_FROM_LABELS
        ax.xaxis.set_label_coords(0, y_coord) # x_coord is 0, since those coordinates are relative to the ax

        # Stack the subplots 
        plt.subplots_adjust(wspace=0)

        # Save and close
        path = os.path.join(self._params['plots_directory'], "query_points_pcp_iteration_" + self.iteration_string() + ".svg")
        fig.savefig(path)
        print("\tSaved PCP of sample locations to", path)
        plt.close()

    def best_samples_plot(self):
        """
        Creates a plot to visualize how the best known parameters evolved and how good they were.
        Convenience wrapper around _samples_plot, see its doc for more details.
        """
        # make a list of iterations, to be used as x-axis tick labels
        iterations = [best_sample_tuple[0] for best_sample_tuple in self.best_samples]
        # add another element for the inital params set
        iterations.append("baseline")
        # make a list of corresponding values from 0 to n-1
        x_axis = range(len(iterations))
        # get the best samples
        best_samples = [best_sample_tuple[1] for best_sample_tuple in self.best_samples]
        # add the baseline sample
        best_samples.append(self.baseline_sample)
        # create the plot and store fix,axes for further fine tuning and saving
        fig, axes = self._samples_plot(x_axis_ticks=iterations, samples=best_samples, x_axis_pos=x_axis, bar_width=0.3)
        #fig.suptitle("Best Sample per Iteration", fontsize=16, fontweight='bold')
        axes[3].set_xlabel("iteration nr.")

        # Save and close
        path = os.path.join(self._params['plots_directory'], "best_samples_boxplot.svg")
        fig.savefig(path)
        print("\tSaved boxplots of best samples to", path)
        plt.close(fig)

    def _samples_plot(self, x_axis_ticks, samples, x_axis_pos=None, show_pm_values=True, bar_width=1, xticklabels_spacing=1):
        """
        Creates a plot to visualize the data contained in a set of samples
        and how their performance measure value came together.
        Contains subplots for:
            the performance measure: Stacked bar plot for the two mixed components of the measure.
            the number of matches: Simple scatter and line plot.
            the translation errors: Boxplots to visualize the error distributions
            the rotation errors: Boxplots to visualize the error distributions
        Use axes[3].set_xlabel("label") on the returned axes, to set a label for the x axis.
        """
        if x_axis_pos is None:
            x_axis_pos = x_axis_ticks

        fig, axes = plt.subplots(4, sharex=True, figsize=(10,12))
        # Setup the axis for the performance measure
        axes[0].set_ylabel(str(self.performance_measure))
        axes[0].tick_params(axis='y', colors='black')
        # Get the weighted error performance measure value for all samples
        error_measure_data = [self.performance_measure.error_measure(s) for s in samples]
        weighted_error_measure_data = [self.performance_measure.error_weight * e for e in error_measure_data]
        axes[0].bar([x-bar_width/2 for x in x_axis_pos], weighted_error_measure_data, color=colors['violet'], width=bar_width,
                    label=u"$\\epsilon$") # plot bars for the error measure
        if show_pm_values:
            # Add text for the value of the error_measure
            for x, bar_top, val in zip(x_axis_pos, weighted_error_measure_data, error_measure_data):
                axes[0].text(x+bar_width/2+0.02, bar_top/2-0.035, str(round(val,2)), color=colors['violet'])
        # Get the weighted matches performance measure value for all samples
        matches_measure_data = [self.performance_measure.matches_measure(s) for s in samples]
        weighted_matches_measure_data = [self.performance_measure.matches_weight * m for m in matches_measure_data]
        axes[0].bar([x-bar_width/2 for x in x_axis_pos], weighted_matches_measure_data, color=colors['red'], width=bar_width, bottom=weighted_error_measure_data,
                    label=u"$\\upsilon$") # plot bars for the matches measure on top of the error measure
        # Add text for the value of the matches_measure
        if show_pm_values:
            for x, bar_top_err, bar_top_ma, val in zip(x_axis_pos, weighted_error_measure_data, weighted_matches_measure_data, matches_measure_data):
                axes[0].text(x+bar_width/2+0.02, bar_top_err + (bar_top_ma/2)-0.035, str(round(val,2)), color=colors['red'])
        # Get the complete performance measure value for all samples
        complete_measure_data = [self.performance_measure(s) for s in samples]
        if show_pm_values:
            # Add text for the value of the complete measure (the weighted sum)
            for x, bar_top, val in zip(x_axis_pos, [sum(x) for x in zip(weighted_error_measure_data, weighted_matches_measure_data)], complete_measure_data):
                axes[0].text(x-bar_width*1.6-0.02, bar_top-0.03, str(round(val,2)), color='black')
        # use the handles and labels fields to reverse the order of the elements in the legend
        handles, labels = axes[0].get_legend_handles_labels()
        legend = axes[0].legend(handles[::-1], labels[::-1], loc='upper right', handletextpad=-1)
        for handle in legend.legendHandles:
            handle.set_width(7.0)
            handle.set_height(7.0)
        axes[0].set_ylim(0,1)
        # Setup the axis for the number of matches
        axes[1].set_ylabel(u'$m$')
        axes[1].tick_params(axis='y', colors='black')
        axes[1].ticklabel_format(useOffset=False) # Forbid offsetting y-axis values
        axes[1].scatter(x_axis_pos, # plot points for nr of matches
                        [s.nr_matches for s in samples],
                        color=colors['red'])
        axes[1].set_ylim(bottom=0)
        # Setup the axis for the translation error boxplots (in magenta)
        axes[2].set_ylabel(u"$\\mathbf{e^t}$ [m]")
        axes[2].tick_params(axis='y', colors='black')
        axes[2].boxplot([s.translation_errors for s in samples],
                        positions=x_axis_pos)
        axes[2].set_xticks(x_axis_pos, x_axis_ticks)
        # Setup the axis for the rotation error boxplots (in magenta)
        axes[3].set_ylabel(u"$\\mathbf{e^r}$ [deg]")
        axes[3].tick_params(axis='y', colors='black')
        axes[3].boxplot([s.rotation_errors for s in samples],
                        positions=x_axis_pos)
        axes[3].set_xticks(x_axis_pos, x_axis_ticks)
        xticklabels = []
        for i, label in enumerate(x_axis_ticks):
            if i % xticklabels_spacing == 0:
                xticklabels.append(label)
            else:
                xticklabels.append("")
        axes[3].set_xticklabels(xticklabels, fontsize=16)

        # Add grid lines to all plots
        for ax in axes:
            ax.grid(axis="y", linestyle="dotted")
        
        return fig, axes

    def paramspace_line_visualization_plot(self, plot_name, param_name, value_range_list = []):
        """
        Saves a plot to visualize the parameter space along a line.
        The line varies along the param_name dimension of the parameter space.
        All other parameters are set to the value from the default_rosparams dict.
        The plot shows the performance measure's behaviour and the samples' data.
        Convenience wrapper around _samples_plot, see its doc for further details.
        """
        # Get sampels, using the eval function
        x_axis = []
        samples = []
        fixed_params = self.eval_function.default_rosparams.copy()
        value_range_list = [round(float(v), 2) for v in value_range_list]
        del(fixed_params[self._to_rosparam(param_name)])
        for params_dict, metric_value, sample in self.eval_function.samples_filtered(fixed_params):
            if value_range_list == [] or round(float(params_dict[self._to_rosparam(param_name)]), 2) in value_range_list:
                x_axis.append(params_dict[self._to_rosparam(param_name)])
                samples.append(sample)
        # Sort both lists by x_axis value before plotting
        temp_sorted_lists = sorted(zip(*[x_axis, samples]))
        x_axis, samples = list(zip(*temp_sorted_lists))
        fig, axes = self._samples_plot(x_axis, samples, np.arange(len(x_axis)), show_pm_values=False, xticklabels_spacing=2)
        #fig.suptitle(param_name, fontsize=16, fontweight='bold')
        axes[3].set_xlabel(param_name)
        axes[3].tick_params(labelsize=12) # Reduce the fontsize of the xticklabels, otherwise they overlap

        # Save and close
        path = os.path.join(self._params['plots_directory'], plot_name)
        fig.savefig(path)
        print("\tSaved metric plot to", path)
        plt.close(fig)

    def plot_gpr_two_param_contour(self, plot_name, param_names):
        """
        Saves a 4 contour plot of the GPR to disk:
        One subplot for GPR mean, target function, GPR variance and the acquisiton function.
        The two axis will be the varied parameters, the color codes the respective values listed above.
        :param plot_name: Filename of the plot. (plot will be saved into the plots_directory of the experiment yaml)
                          If None, the plot will not be saved but instead be opened in interactive mode.
        :param param_names: List of names of the parameters that are shown in this plot.
        """
        fig, axes = plt.subplots(2, 2)
        RESOLUTION = 50 # determines the plots' RESOLUTION, i.e. how many values per dimension get sampled the GP
        OFFSET = 0.01 # offsets the x-, y-axes, so markers at the edge are visible
        # Prepare x and y data (the same for all plots)
        x_bounds = self.eval_function.optimization_bounds[self._to_rosparam(param_names[0])]
        x = np.linspace(x_bounds[0] - OFFSET, x_bounds[1] + OFFSET, RESOLUTION)
        y_bounds = self.eval_function.optimization_bounds[self._to_rosparam(param_names[1])]
        y = np.linspace(y_bounds[0] - OFFSET, y_bounds[1] + OFFSET, RESOLUTION)
        value_range = self.performance_measure.value_range
        # Set labels for all axes
        for ax in [sub_axes for sublist in axes for sub_axes in sublist]:
            ax.set_xlabel(param_names[0], fontsize=9)
            ax.set_ylabel(param_names[1], fontsize=9)
        # Set shared arguments for the contourplot method
        contour_kwargs = {'cmap': 'hot', 'extend': 'both'}
        ############
        # Prepare known samples plot
        samples_x, samples_y, samples_z = self._get_filtered_samples(param_names)
        if len(samples_x) > 0: # guard against crashes if no known samples exist in the plane we're currently looking at
            plot = axes[0][0].scatter(samples_x, samples_y, c=samples_z, cmap='hot', edgecolor='', vmin=value_range[0], vmax=value_range[1])
            # set x,y axis bounds, only needed for the scatter plot
            axes[0][0].set_xlim(x_bounds[0] - OFFSET, x_bounds[1] + OFFSET)
            axes[0][0].set_ylim(y_bounds[0] - OFFSET, y_bounds[1] + OFFSET)
            axes[0][0].set_title("All Known Samples")
            fig.colorbar(plot, ax=axes[0][0], ticks=np.linspace(value_range[0], value_range[1], 11), label=str(self.performance_measure))
        ############
        # Prepare mean plot
        # Get observations known to the GPR
        filtered_X, filtered_Y = self._get_filtered_observations(param_names)
        # Get the GPR's estimate data
        predictionspace = self._get_prediction_space(param_names, RESOLUTION)
        mean, sigma = self.optimizer.gp.predict(predictionspace, return_std=True)
        z_mean = np.reshape(mean, (RESOLUTION, RESOLUTION))
        plot = axes[0][1].contourf(x, y, z_mean, levels=np.linspace(value_range[0], value_range[1], RESOLUTION), **contour_kwargs)
        axes[0][1].set_title("Estimated Mean")
        fig.colorbar(plot, ax=axes[0][1], ticks=np.linspace(value_range[0], value_range[1], 11), label=str(self.performance_measure))
        # plot all observations the GPR has
        axes[0][1].scatter(filtered_X.T[self._to_optimizer_id(param_names[0])],
                           filtered_X.T[self._to_optimizer_id(param_names[1])], marker='+', edgecolor='white')
        ############
        # Prepare variance plot
        z_var = np.reshape(sigma * sigma, (RESOLUTION, RESOLUTION))
        plot = axes[1][0].contourf(x, y, z_var, levels=np.linspace(0, 0.15, RESOLUTION), **contour_kwargs)
        axes[1][0].set_title("Estimation Variance")
        fig.colorbar(plot, ax=axes[1][0], ticks=np.linspace(0, 0.5, 11), label=u"$\sigma^2$")
        # plot all observations the GPR has
        axes[1][0].scatter(filtered_X.T[self._to_optimizer_id(param_names[0])],
                           filtered_X.T[self._to_optimizer_id(param_names[1])], marker='+', edgecolor='white')
        ############
        # Prepare acquisiton function plot
        acq = self.optimizer.util.utility(predictionspace, gp=self.optimizer.gp,
                                          y_max=self.optimizer.res['max']['max_val']) 
        z_acq = np.reshape(acq, (RESOLUTION, RESOLUTION))
        plot = axes[1][1].contourf(x, y, z_acq, levels=np.linspace(0, 1, RESOLUTION), **contour_kwargs)
        axes[1][1].set_title("Acquisition Function")
        kappa = 2 if not 'optimizer_params' in self._params.keys() else self._params['optimizer_params']['kappa']
        acq_label = u"$\mu + " + str(kappa) + u"\sigma$" # TODO: Change if using different acquisiton functions
        fig.colorbar(plot, ax=axes[1][1], ticks=np.linspace(0, 1, 11), label=acq_label)
        # plot all observations the GPR has
        axes[1][1].scatter(filtered_X.T[self._to_optimizer_id(param_names[0])],
                           filtered_X.T[self._to_optimizer_id(param_names[1])], marker='+', edgecolor='white')
        # Call mpl's magic layouting method (elimates overlap etc.)
        plt.tight_layout()
        # save and close
        path = os.path.join(self._params['plots_directory'], plot_name)
        fig.savefig(path) # Save an image of the 3d plot (which, of course, only shows one specific projection)
        print("\tSaved contour plot to", path)
        plt.close()

    def plot_gpr_two_param_3d(self, plot_name, param_names):
        """
        Saves a 3D plot of the GPR estimate to disk.
        The X,Y axes will be parameter values, the Z axis the performance measure.
        :param plot_name: Filename of the plot. (plot will be saved into the plots_directory of the experiment yaml)
                          If None, the plot will not be saved but instead be opened in interactive mode.
        :param param_names: List of names of the parameters that are shown in this plot.
        """
        fig = plt.figure()
        ax_3d = fig.add_subplot(111, projection='3d')

        # Get data from the GPR
        RESOLUTION = 50
        predictionspace = self._get_prediction_space(param_names, RESOLUTION)
        mean, sigma = self.optimizer.gp.predict(predictionspace, return_std=True)
        filtered_X, filtered_Y = self._get_filtered_observations(param_names)
        ax_3d.scatter(xs = filtered_X.T[self._to_optimizer_id(param_names[0])],
                      ys = filtered_X.T[self._to_optimizer_id(param_names[1])],
                      zs = filtered_Y, c='blue', label=u'Observations', s=50)
        ax_3d.plot_wireframe(predictionspace[:,self._to_optimizer_id(param_names[0])].reshape((RESOLUTION, RESOLUTION)),
                             predictionspace[:,self._to_optimizer_id(param_names[1])].reshape((RESOLUTION, RESOLUTION)),
                             mean.reshape((RESOLUTION, RESOLUTION)), label=u'Prediction')

        # Plot all evaluation function samples we have
        samples_x, samples_y, samples_z = self._get_filtered_samples(param_names)
        ax_3d.scatter(samples_x, samples_y, samples_z, c='red', label=u"All known samples", s=5)
        # Add legend
        ax_3d.legend(loc='upper right')
        # set limits
        ax_3d.set_xlim(self.eval_function.optimization_bounds[self._to_rosparam(param_names[0])])
        ax_3d.set_ylim(self.eval_function.optimization_bounds[self._to_rosparam(param_names[1])])
        ax_3d.set_zlim=(self.performance_measure.value_range)
        # set labels
        ax_3d.set_xlabel(param_names[0])
        ax_3d.set_ylabel(param_names[1])
        ax_3d.set_zlabel(str(self.performance_measure))

        # Save, show, clean up fig
        if plot_name is None:
            plt.show()
        else:
            path = os.path.join(self._params['plots_directory'], plot_name)
            fig.savefig(path) # Save an image of the 3d plot (which, of course, only shows one specific projection)
            print("\tSaved 3d plot to", path)
        plt.close()

    def plot_gpr_single_param(self, plot_name, param_name):
        """
        Saves a plot of the GPR estimate to disk.
        Visualizes the estimated function mean and 95% confidence area from the GPR.
        Additionally shows the actual values for *all* available samples of the current ObjectiveFunction.
        """

        # Setup the figure and its axes
        fig, metric_axis = plt.subplots()
        self._setup_metric_axis(metric_axis, param_name)

        predictionspace = self._get_prediction_space([param_name])
        # Get mean and sigma from the gaussian process
        y_pred, sigma = self.optimizer.gp.predict(predictionspace, return_std=True)
        # Plot the observations available to the gaussian process
        filtered_X, filtered_Y = self._get_filtered_observations((param_name))
        metric_axis.plot(filtered_X.T[self._to_optimizer_id(param_name)], filtered_Y, '.', color=colors['green'], markersize=10, label=u'Given Observations $D_n$')
        # Plot the gp's prediction mean
        plotspace = predictionspace[:,self._to_optimizer_id(param_name)]
        metric_axis.plot(plotspace, y_pred, '-', color="black", label=u'Prediction')
        # Plot the gp's prediction 'sigma-tube'
        metric_axis.fill(np.concatenate([plotspace, plotspace[::-1]]),
                         np.concatenate([y_pred - 1.9600 * sigma,
                                        (y_pred + 1.9600 * sigma)[::-1]]),
                         alpha=.5, fc=colors['blue'], ec='None', label='95% confidence interval')

        # Plot all evaluation function samples we have
        samples_x = []
        samples_y = []
        # Gather available x, y pairs in the above two variables
        fixed_params = self.eval_function.default_rosparams.copy()
        del(fixed_params[self._to_rosparam(param_name)])
        for params_dict, metric_value, sample in self.eval_function.samples_filtered(fixed_params):
            samples_x.append(params_dict[self._to_rosparam(param_name)])
            samples_y.append(metric_value)
        metric_axis.scatter(samples_x, samples_y, c=colors['red'], label=u"All Observations $D_*$")

        metric_axis.legend(loc='lower right')

        # Save and close
        path = os.path.join(self._params['plots_directory'], plot_name)
        fig.savefig(path)
        print("\tSaved plot to", path)
        plt.close()

    def _get_prediction_space(self, free_params, RESOLUTION=1000):
        """
        Returns a predictionspace for the gaussian process.
        :param free_params: Parameters which are supposed to vary in the prediction space.
                            For all other dimensions, the respective parameter will only have the value from the default rosparams.
        :param RESOLUTION: The RESOLUTION for the prediction (higher means better quality)
        """
        predictionspace = np.zeros((len(self.optimizer.keys), np.power(RESOLUTION, len(free_params))))
        for key in [fixed_param for fixed_param in self.optimization_defs.keys() if fixed_param not in free_params]: # Fill all fixed params with the default value
            predictionspace[self._to_optimizer_id(key)] = np.full((1,np.power(RESOLUTION, len(free_params))), self.eval_function.default_rosparams[self._to_rosparam(key)])
        # Handle free params
        free_param_bounds = [self.eval_function.optimization_bounds[self._to_rosparam(key)] for key in free_params]
        free_param_spaces = [np.linspace(bounds[0], bounds[1], RESOLUTION) for bounds in free_param_bounds]
        free_param_coordinates = np.meshgrid(*free_param_spaces)
        for i, key in enumerate(free_params):
            predictionspace[self._to_optimizer_id(key)] = free_param_coordinates[i].flatten()
        predictionspace = predictionspace.T
        return predictionspace

    def _get_filtered_samples( self, free_params_display_names):
        """
        Returns samples from the sample database, filtered by their fixed params values.
        All parameters in the default rosparam dict are fixed, except those in the free_params_display_names list.
        NOTE: Currently only really works for len(free_params_display_names) == 2...

        :param free_params_display_names: List of parameter display names which are not fixed.
        """
        # Those members will hold the data
        samples_x = []
        samples_y = []
        samples_z = []
        # Gather available x, y pairs in the above two variables
        fixed_params = self.eval_function.default_rosparams.copy()
        # delete the free parameters from it
        del(fixed_params[self._to_rosparam(free_params_display_names[0])])
        del(fixed_params[self._to_rosparam(free_params_display_names[1])])
        for params_dict, metric_value, sample in self.eval_function.samples_filtered(fixed_params):
            samples_x.append(params_dict[self._to_rosparam(free_params_display_names[0])])
            samples_y.append(params_dict[self._to_rosparam(free_params_display_names[1])])
            samples_z.append(metric_value)
        return samples_x, samples_y, samples_z

    def _get_filtered_observations(self, free_params_display_names):
        fixed_params_display_names = [p for p in self.optimization_defs.keys() if not p in free_params_display_names]
        usables = []
        for i, x in enumerate(self.optimizer.X):
            # For each sample, check if it's usable:
            usable = True
            for fixed_param in fixed_params_display_names:
                if not self.eval_function.default_rosparams[self._to_rosparam(fixed_param)] == x[self._to_optimizer_id(fixed_param)]:
                    usable = False
                    break
            if usable:
                usables.append(i)
        # Filters self.optimizer.X with indices in usables along axis 0
        return np.take(self.optimizer.X, usables, 0), np.take(self.optimizer.Y, usables)

    def _to_optimizer_id(self, display_name):
        """
        Takes a "display_name" (i.e. the name for a parameter as defined in the experiment's yaml file)
        and returns its id in the optimizer's X-array.
        """
        rosparam_name = self._to_rosparam(display_name)
        for i, key in enumerate(self.optimizer.keys):
            if key == rosparam_name:
                return i
        raise LookupError("Rosparam", rosparam_name, "is not in the optimizer's key list. Maybe it isn't one of the optimized parameters?")

    def _to_rosparam(self, display_name):
        """
        Takes a "display_name" (i.e. the name for a parameter as defined in the experiment's yaml file)
        and returns its respective rosparam_name
        """
        return self.optimization_defs[display_name]['rosparam_name']

    def _resolve_relative_path(self, path):
        """
        Helper function for resolving paths relative to the _relpath_root-member.
        """
        if not os.path.isabs(path):
            return os.path.join(self._relpath_root, path)
        else:
            return path

    def grid_search(self):
        """
        Only usable when optimizing a single parameter.
        Performs a grid search in the parameter space, ignoring the acquisition function and performance measures.
        The grid search starts at the min_bound, according to the optimization_defs.
        A sample is drawn for each i*step_size, until that value gets bigger than the 
        max_bound from the optimization_defs.

        This method is not (yet?) intended to use in conjunction with the normal iterate-method.
        """
        # error handling
        if not len(self.optimization_defs) == 1:
            raise RuntimeError("grid search method only allowed when optimizing a single parameter")
        if not 'grid_search_step_size' in self._params['optimizer_initialization'].keys():
            raise RuntimeError("Missing required parameter in experiment_yaml: 'grid_search_step_size'")
        
        display_name, p_dict = list(self.optimization_defs.items())[0]
        rosparam_name = p_dict['rosparam_name']
        min_bound = p_dict['min_bound']
        max_bound = p_dict['max_bound']
        print("Performing grid search on parameter", display_name, "(aka", rosparam_name, ").")
        for step_size in self._params['optimizer_initialization']['grid_search_step_size']:
            print("Setting step size to", step_size)
            paramspace_grid = {rosparam_name: np.arange(min_bound, max_bound, step_size)}
            print("Querying at", len(paramspace_grid[rosparam_name]), "locations:")
            print(paramspace_grid)
            self.optimizer.explore(paramspace_grid)
            # just fit the gp hyperparams to the data given via explore, don't choose new samples
            self.optimizer.maximize(init_points=0, n_iter=0, kappa=0, **self.gpr_kwargs)
            self.plot_gpr_single_param(display_name.replace(" ", "_") + "_step_size_" + str(step_size) + ".svg", display_name)
            # this is kind of hacky, but will induce a string to identify which of the resulting plots and param files corresponds to which step_size
            self.iteration = step_size
            # another hacky piece of code to make the Bayesian Optimization output a maximum without having to do at least one iteration
            self.optimizer.res['max'] = {'max_val': self.optimizer.Y.max(),
                               'max_params': dict(zip(self.optimizer.keys,
                                                      self.optimizer.X[self.optimizer.Y.argmax()]))
                               }
            
            self.handle_new_best_parameters()
            # reset optimizer
            self.optimizer = BayesianOptimization(self.eval_function.evaluate, self.opt_bounds, verbose=0)

    def iterate(self):
        """
        Runs one iteration of the system
        """
        if 'optimizer_params' in self._params.keys():
            opt_params_dict = self._params['optimizer_params']
            init_points = opt_params_dict.get('pre_iteration_random_points', 0)
            n_iter = opt_params_dict.get('samples_per_iteration', 1)
            kappa = opt_params_dict.get('kappa', 2)
            kappa_fine_tuning = opt_params_dict.get('kappa_fine_tuning', 1)
        else:
            init_points = 0
            n_iter = 1
            kappa = 2
            kappa_fine_tuning = 1

        print("\033[1;4;35m", self.iteration_string(), ":\033[0m", sep="")
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter, kappa=kappa if not self.fine_tune else kappa_fine_tuning, **self.gpr_kwargs)
        # Dump the gpr's state for later use (e.g. interactive plots)
        pickle.dump(self, open(os.path.join(self._params['plots_directory'], "experiment_state.pkl"), 'wb'))
        # plot this iteration's gpr state in 2d for all optimized parameters
        self.plot_all_single_param()
        # plot this iteration's gpr state in 3d for the first two parameters (only if there are more than one parameter)
        display_names = list(self._params['optimization_definitions'].keys())
        if len(display_names) > 1 and len(display_names) < 4: # don't plot all pairs of parameters when there are more then 4, it'll take too much time.
            self.plot_all_two_params()
        # Check if we found a new best parameter set
        self.handle_new_best_parameters()
        self.output_sampled_params_table() # output a markdown table with all sampled params
        if len(display_names) > 2:
            self.query_points_plot() # output a pcp with lines for each sampled param
        # increase iteration counter
        self.iteration += 1

    def handle_new_best_parameters(self):
        """
        Call regularily to check whether a new best set of parameters has beend found (e.g. each iteration).
        If there's a new best parameter set, the method will add it to the best_samples list
        and (re)create relevant plots and param text files.
        """
        if self.max_performance_measure < self.optimizer.res['max']['max_val']:
            print("\t\033[1;35mNew maximum found, outputting params and plots!\033[0m")
            self.max_performance_measure = self.optimizer.res['max']['max_val']
            # Dump the best parameter set currently known by the optimizer
            yaml.dump(self.max_rosparams, open(os.path.join(self._params['plots_directory'], "best_rosparams_" + self.iteration_string() + ".yaml"), 'w'))
            # store the best known sample in the best_samples dict, for boxplots
            self.best_samples.append((self.iteration, self.max_sample))
            self.plot_all_new_best_params()

    def plot_all_new_best_params(self, plot_all_violin=False):
        """
        Plots all kinds of plots for visualizing how the best parameters evolved.

        :param plot_all_violin: Whether to plot all violin plots.
                                If false, only the violin plot of self.max_sample is created except when a new max error is found.
        """
        # Find max rotation and translation errors from all best samples
        max_rotation_error = max(max(s[1].rotation_errors) for s in self.best_samples)
        max_translation_error = max(max(s[1].translation_errors) for s in self.best_samples)
        # If the new best sample induced a new max rotation/translation error, recreate all previous violin plots.
        # Otherwise, the y-axis will jump around when comparing the violin plots, which makes them pretty useless.
        if max_rotation_error == max(self.max_sample.rotation_errors) or max_translation_error == max(self.max_sample.translation_errors):
            print("\tNew max error, replotting all violin plots.")
            plot_all_violin = True

        if not plot_all_violin:
            # violin plot of the new best sample
            plot_path = os.path.join(self._params['plots_directory'], "violin_plot_" + self.iteration_string() + ".svg")
            self.plot_error_distribution(plot_path, self.max_sample, max_rotation_error, max_translation_error)
        else:
            # plot violin plots for all best samples
            for iteration, sample in self.best_samples:
                plot_path = os.path.join(self._params['plots_directory'], "violin_plot_" + self.iteration_string(iteration) + ".svg")
                self.plot_error_distribution(plot_path, sample, max_rotation_error, max_translation_error, iteration)
            # also (re-)plot the default params distribution
            plot_path = os.path.join(self._params['plots_directory'], "violin_plot_baseline.svg")
            self.plot_error_distribution(plot_path, self.baseline_sample, max_rotation_error, max_translation_error, "baseline")

        # create a new version of the boxplot which includes the new sample
        self.best_samples_plot()

    def plot_all_two_params(self):
        """
        Plots all kinds of plots that visualize two parameters at once.
        The method does this for all pairs of optimized parameters.
        """
        display_names = list(self._params['optimization_definitions'].keys())
        for display_name_pairs in itertools.combinations(display_names, 2):
            two_param_plot_name_prefix = display_name_pairs[0].replace(" ", "_") + "_" + display_name_pairs[1].replace(" ", "_") + "_" + self.iteration_string() + ".svg"
            self.plot_gpr_two_param_3d("3d_" + two_param_plot_name_prefix, display_name_pairs)
            self.plot_gpr_two_param_contour("contour_" + two_param_plot_name_prefix, display_name_pairs)

    def plot_all_single_param(self):
        """
        Plots all kinds of plots that visualize a single parameter.
        The method does this for all optimized parameters.
        """
        display_names = list(self._params['optimization_definitions'].keys())
        for display_name in display_names:
            self.plot_gpr_single_param(display_name.replace(" ", "_") + "_" + self.iteration_string() + ".svg", display_name)

    def iteration_string(self, iteration=None):
        """
        Creates a string to identify an iteration, mainly used for naming plots.
        Returns a string of form "DDDDD_iteration", with DDDDD being the iteration's number, with prefixed zeros, if necessary.
        Or "DDDDD_iteration_finetuned", if finetuning is active.

        :param iteration: The iteration number as an integer, or None. If None, the current iteration (self.iteration) is used.
        """
        if iteration is None:
            iteration = self.iteration
        fine_tune_suffix = ""
        if self.fine_tune:
            fine_tune_suffix = "_finetuned"
        return str(iteration).zfill(5) + "_iteration" + fine_tune_suffix

    @property
    def opt_bounds(self):
        """
        Returns a dict of optimization bounds.
        It's indexed via rosparam_name and contains a tuple (min, max) bounds.
        """
        opt_bounds = dict()
        for p_name, p_defs in self.optimization_defs.items():
            opt_bounds[p_defs['rosparam_name']] = (p_defs['min_bound'], p_defs['max_bound'])
        return opt_bounds

    @property
    def baseline_sample(self):
        """
        Returns the sample of the default parameterset.
        """
        return self.sample_db[self.eval_function.default_rosparams]

    @property
    def max_sample(self):
        """
        Returns the sample of the best known parameterset.
        """
        return self.sample_db[self.max_rosparams]

    @property
    def max_rosparams(self):
        """
        Returns the complete rosparams dict with the best known parameters set.
        Of course, only optimized parameters will potentially be different from the inital param set.
        """
        # Get the default parameter values, including those which didn't get optimized
        best_rosparams = self.eval_function.default_rosparams.copy()
        # Get the best known optimized parameters as a dict from the optimizer
        best_optimized_params = self.optimizer.res['max']['max_params'].copy()
        # Fix parameter types and round its values
        self.eval_function.preprocess_optimized_params(best_optimized_params)
        # Update the default params dict with optimized params dict
        best_rosparams.update(best_optimized_params)
        return best_rosparams


if __name__ == '__main__': # don't execute when module is imported
    import argparse # for the cmd-line interface

    def command_line_interface():
        """
        Commandline interface for using this module
        """
        description_string = ("Command line utility for running Bayesian Optimization experiments on"
                              " parameter sets of a feature-based map matching pipeline."
                              " The program's default mode is the Experiment Mode."
                              " It only requires passing a yaml file as experiment_yaml parameter."
                              " See the other arguments' descriptions for different modes."
                             )
        rm_arg_help = ("Enters Remove Samples mode:"
                       " Manually remove ObjectiveFunction Samples from the database and from disk."
                       " (only the pickled Sample, not the map matcher results from which it was generated)"
                       " Samples can be identified via their hash or the pickle path."
                       " Exits after all supplied samples have been removed."
                      )
        add_arg_help = ("Enters Add Samples mode:"
                        " Manually add ObjectiveFunction Samples to the database."
                        " Simply supply the paths to the directories from which INTERFACE_MODULE can create it."
                        " In this mode, existing Samples with the same pickle location on disk,"
                        " and existing entries in the database (same rosparams-hash) will be overwritten."
                        " Exits after adding all supplied samples."
                       )

        parser = argparse.ArgumentParser(description=description_string)
        parser.add_argument('experiment_yaml',
                            help="Path to the yaml file which defines all parameters of one experiment run.")
        parser.add_argument('--fine-tune', '-ft',
                            dest='fine_tune', action='store_true',
                            help="Lets the optimizer use kappa_fine_tuning to rather improve upon the current known maximum.")
        parser.add_argument('--use-previous-observations', '-oa',
                            dest='use_previous_observations', action='store_true',
                            help="Will initialize the optimizer with ALL samples in the database that fit the current optimization definitions")
        parser.add_argument('--use-previous-nonzero-observations', '-onz',
                            dest='use_previous_nonzero_observations', action='store_true',
                            help="Will initialize the optimizer will samples in the database that fit the current optimization definitions and have a nonzero performance measure.")
        parser.add_argument('--list-all-samples', '-la',
                            dest='list_all_samples', action='store_true',
                            help="Lists all samples available in the database and exits.")
        parser.add_argument('--list-samples', '-ls',
                            dest='list_samples', action='store_true',
                            help="Lists those samples in the database, which are relevant for this experiment and exits.")
        parser.add_argument('--remove-samples', '-rm',
                            dest='remove_samples', nargs='+', help=rm_arg_help)
        parser.add_argument('--clean-map-matcher-env', '-cmme',
                            dest='clean_mme', action='store_true',
                            help="Removes all map matcher environment directories, which don't have a sample in the sample database associated with it." +\
                                 "I.e. remove all directories of map matcher runs that didn't finish and, because of that, weren't added to the database.")
        parser.add_argument('--add-samples', '-a',
                            dest='add_samples', nargs='+', help=add_arg_help)
        parser.add_argument('--plot-paramspace-line', '-p',
                            dest='plot_paramspace_line', nargs='+',
                            help="Plots a 1D visualization of the metric's behaviour when changing the given parameter." +\
                                 " (parameter name has to fit the optimization definitions in the yaml file)")
        parser.add_argument('--value-range', '-vr',
                            dest='value_range', nargs=3, type=float,
                            help="Only used with --plot-paramspace-line. Adds restrictions on the parameter values of samples that are plottet." +\
                                 " Expects three values min, max, step_size to create a list of allowed parameter values." +\
                                 " E.g. -vr 0 5 0.1 would only allow parameter values in [0, 0.1, 0.2, ..., 5.0]." +\
                                 " Currently only supports step sizes with two decimal places...")
        parser.add_argument('--3d-plot', '-3d',
                            dest='plot3d', nargs='+',
                            help="Loads the supplied experiment state pickle file(s) and opens interactive figures from their states. " +\
                                  "This may require setting your backend to something that supports interactive mode. " +\
                                  "(see ~/.config/matplotlib/matplotlibrc for example)")
        parser.add_argument('--plot-single', '-p1',
                            dest='plot_single', nargs='+',
                            help="Loads the supplied experiment state pickle file(s) and plots their contents in 'single' plots. " +\
                                 "Have a look at ExperimentCoordinator's 'plot_all_single_param' method, for further information.")
        parser.add_argument('--plot-two', '-p2',
                            dest='plot_two', nargs='+',
                            help="Loads the supplied experiment state pickle file(s) and plots their contents in 'single' plots. " +\
                                 "Have a look at ExperimentCoordinator's 'plot_all_two_params' method, for further information.")
        parser.add_argument('--plot-max', '-pmax',
                            dest='plot_max', nargs='+',
                            help="Loads the supplied experiment state pickle file(s) and plots their contents in 'single' plots. " +\
                                 "Have a look at ExperimentCoordinator's 'plot_all_new_best_params' method, for further information.")
        parser.add_argument('--plot-queries', '-pq', dest='plot_queries',
                            help="Loads the supplied experiment state pickle file and plots a PCP visualizing query point locations in param-space.")
        parser.add_argument('--add-new-param',
                            dest='new_param', nargs='+',
                            help="Utility command that can be used when a new parameter has been added to the map matcher. " +\
                                 "Expects two arguments: First, a single string, corresponding to the rosparam name of the new parameter. " +\
                                 "Second, that parameter's default value, which should induce the same map matcher behaviour as before the parameter was introduced. " +\
                                 "This method will iterate over all samples in this experiment's sample database and add the new parameter with the given value to each sample. " +\
                                 "The new parameter will only be added if it wasn't already present in the sample's parameter dict.")
        parser.add_argument('--resume', '-r',
                            help="Expects a path to an old, pickled experiment state. Will try to resume that experiment." +\
                                 "Take care: Changes to the code and to the parameters won't take effect when restarting an old experiment.")
        args = parser.parse_args()

        if args.plot3d:
            print("--> Interactive 3D plot mode <--")
            for path in args.plot3d:
                experiment_coordinator = pickle.load(open(path, 'rb'))
                experiment_coordinator.plot_gpr_two_param_3d(None, list(experiment_coordinator._params['optimization_definitions'].keys()))
            sys.exit()
        if args.plot_single:
            print("--> Plot mode (single) <--")
            for path in args.plot_single:
                experiment_coordinator = pickle.load(open(path, 'rb'))
                experiment_coordinator.plot_all_single_param()
            sys.exit()
        if args.plot_two:
            print("--> Plot mode (two) <--")
            for path in args.plot_two:
                experiment_coordinator = pickle.load(open(path, 'rb'))
                experiment_coordinator.plot_all_two_params()
            sys.exit()
        if args.plot_max:
            print("--> Plot mode (max) <--")
            for path in args.plot_max:
                experiment_coordinator = pickle.load(open(path, 'rb'))
                experiment_coordinator.plot_all_new_best_params(plot_all_violin=True)
            sys.exit()
        if args.plot_queries:
            print("--> Mode: Plot Queries <--")
            experiment_coordinator = pickle.load(open(args.plot_queries, 'rb'))
            experiment_coordinator.query_points_plot()
            sys.exit()
        if args.resume:
            print("--> Resuming old experiment <--\n"+\
                  "May cause unexpected behaviour: Code changes won't magically appear in the pickled experiment state and parameter changes won't take effect when resuming an old experiment.")
            experiment_coordinator = pickle.load(open(args.resume, 'rb'))
            experiment_coordinator.fine_tune = args.fine_tune # set the fine_tune flag if the user started this script with --fine-tune
            while True:
                experiment_coordinator.iterate()
            sys.exit()

        # Load the parameters from the yaml into a dict
        experiment_parameters_dict = yaml.safe_load(open(args.experiment_yaml))
        relpath_root = os.path.abspath(os.path.dirname(args.experiment_yaml))
        # Open the file handle to the database dict
        experiment_coordinator = ExperimentCoordinator(experiment_parameters_dict, relpath_root)

        # Check cmdline arguments for special modes, default mode (Experiment mode) is below
        if args.list_all_samples:
            print("--> Mode: List All Samples <--")
            for sample in experiment_coordinator.sample_db:
                print(sample)
            print("Total number of samples", len(experiment_coordinator.sample_db._db_dict))
            sys.exit()
        if args.list_samples:
            print("--> Mode: List Samples <--")
            count = 0
            for x, y, s in experiment_coordinator.eval_function:
                count += 1
                print(s)
                print("\tOptimized Parameters:")
                for display_name in experiment_coordinator.optimization_defs.keys():
                    print("\t\t", display_name, "=", x[experiment_coordinator._to_rosparam(display_name)])
                print("\tMetric-value:", y)
                if isinstance(experiment_coordinator.performance_measure, performance_measures.MixerMeasure):
                    print("\t\terror measure", type(experiment_coordinator.performance_measure.error_measure), "=",
                          experiment_coordinator.performance_measure.error_measure(s),
                          "(weight", 1 - experiment_coordinator.performance_measure.matches_weight, ")")
                    print("\t\tnr. matches measure", type(experiment_coordinator.performance_measure.matches_measure), "=",
                          experiment_coordinator.performance_measure.matches_measure(s),
                          "(weight", experiment_coordinator.performance_measure.matches_weight, ")")
            print("Number of usable samples:", count)
            sys.exit()
        if args.clean_mme:
            print("--> Mode: Clean Up Map Matcher Environment <--")
            # List of all sample origins in our database
            sample_origins = [os.path.basename(os.path.dirname(sample.origin)) for sample in experiment_coordinator.sample_db]
            mme_path = experiment_coordinator.sample_db.sample_generator_config['environment']
            # List of all map matcher ens
            map_matcher_envs = [os.path.abspath(os.path.join(mme_path, path)) for path in os.listdir(mme_path)]
            print("Number of map matcher envs:", len(map_matcher_envs), "; Number of sample origins:", len(sample_origins))
            print("Generating list of map matcher envs which don't have a sample associated with it...")
            to_delete_list = [mme_path for mme_path in map_matcher_envs if not os.path.basename(mme_path) in sample_origins]
            for path in to_delete_list:
                print(path)
            print("Delete those", len(to_delete_list), "map matcher envs? (y/n)")
            if input() in ['y', 'Y', 'yes', 'Yes']:
                count = 0
                for path in to_delete_list:
                    count += 1
                    print("deleted", count, "of", len(to_delete_list), "directories.", end='\r')
                    shutil.rmtree(path)
                print("All unused envs deleted.")
            else:
                print("aborted.")
            sys.exit()
        if args.remove_samples:
            print("--> Mode: Remove Samples <--")
            for sample_id in args.remove_samples:
                experiment_coordinator.sample_db.remove_sample(sample_id)
            sys.exit()
        if args.add_samples:
            print("--> Mode: Add Samples <--")
            for sample_path in args.add_samples:
                experiment_coordinator.sample_db.create_sample_from_map_matcher_results(sample_path, override_existing=True)
            sys.exit()
        if args.plot_paramspace_line:
            print("--> Mode: Plot Paramspace Projected on Line <--")
            param_name = ' '.join(args.plot_paramspace_line)
            print("\tFor parameter '", param_name, "'", sep="")
            value_range_list = []
            if args.value_range:
                value_range_list = np.arange(args.value_range[0], args.value_range[1] + args.value_range[2], args.value_range[2])
            experiment_coordinator.paramspace_line_visualization_plot("line_projection_" + param_name.replace(" ", "_") + ".svg", param_name, value_range_list)
            sys.exit()
        if args.new_param:
            print("--> Mode: New Parameter Patching <--")
            print("Patching new parameter", args.new_param[0], "with default value", args.new_param[1])
            if not len(args.new_param) == 2:
                raise ValueError("new_param's length isn't 2. --add-new-param requires exactly two arguments (see -h for more information).")
            # Create a list that contains all samples which don't contain the new parameter (new_param[0])
            patch_samples_list = []
            # and a member to save a single sample which already contains the new parameter, used for typecasting
            other_sample = None
            # fill the two variables above
            for sample in experiment_coordinator.sample_db:
                if not args.new_param[0] in sample.parameters.keys():
                    patch_samples_list.append(sample)
                elif other_sample is None:
                    other_sample = sample
            # check if there was a sample in the db to extract the type information
            if other_sample is None:
                raise RuntimeError("No sample in the db already has the new parameter. Can't go on with patching, since I don't know which type it should have. (fixme, maybe?)")
            new_param_type = type(other_sample.parameters[args.new_param[0]])
            print("Total number of samples that need to be patched:", len(patch_samples_list))
            print(len(experiment_coordinator.sample_db) - len(patch_samples_list), "already seem to have that parameter.")
            print("New parameter's type is", new_param_type)
            print("Execute the patching routine? (y/n)")
            if input() in ['y', 'Y', 'yes', 'Yes']:
                for sample in patch_samples_list:
                    # Get the hash to find the unpatched sample in the db
                    sample_hash = objective_function.SampleDatabase.rosparam_hash(sample.parameters)
                    # Patch the sample's parameters dict, casting the default value to the correct type
                    sample.parameters[args.new_param[0]] = new_param_type(args.new_param[1])
                    # Update the sample in the db (and its .pkl on disk)
                    experiment_coordinator.sample_db.update_sample(sample_hash, sample)
            else:
                print("aborted.")
            sys.exit()

        if 'optimizer_initialization' in experiment_coordinator._params:
            if isinstance(experiment_coordinator._params['optimizer_initialization'], list):
                print("--> Mode: Standard Experiment <--")
                experiment_coordinator.initialize_optimizer(use_previous_observations=(args.use_previous_observations or args.use_previous_nonzero_observations),
                                                            only_nonzero_observations=args.use_previous_nonzero_observations)
            elif "grid_search_step_size" in experiment_coordinator._params['optimizer_initialization']:
                print("--> Mode: Grid Search Experiment <--")
                experiment_coordinator.grid_search()
                sys.exit()
        elif "comparison_plot" in experiment_coordinator._params:
            print("--> Mode: Comparison Plot <--")
            print("\tComparing the following samples:")
            # make a list of param_names, to be used as x-axis tick labels and get the parameter sets of the respective samples
            param_names = []
            param_file_paths = []
            for pair in experiment_coordinator._params["comparison_plot"]:
                param_names.append(pair[0])
                param_file_paths.append(pair[1])
            for name, path in zip(param_names, param_file_paths):
                print("\t\t", name, ": ", path, sep="")
            # make a list of corresponding values from 0 to n-1, to fix the positions on the x-axis
            x_axis = range(len(param_names))
            best_samples = [experiment_coordinator.sample_db[rosparam.load_file(path)[0][0]] for path in param_file_paths]
            # create the plot and store fix,axes for further fine tuning and saving
            fig, axes = experiment_coordinator._samples_plot(x_axis_ticks=param_names, samples=best_samples, x_axis_pos=x_axis, bar_width=0.3)
            #fig.suptitle("Parameter Comparison", fontsize=16, fontweight='bold')
            axes[3].set_xlabel("Parameter Sets")

            # Save and close
            path = os.path.join(experiment_coordinator._params['plots_directory'], "comparison_plot.svg")
            fig.savefig(path)
            print("\tSaved comparison plot to", path)
            plt.close(fig)
            sys.exit()
        else:
            print("Unknown mode, the optimizer intialization should be a list of dicts for the standard experiment. Exiting.")
            sys.exit()

        while True:
            experiment_coordinator.iterate()

    # Execute cmdline interface
    command_line_interface()
