# should:
# save newest model
# log various statistics + timestamps
# log parameters
# copy code
# have a mechanism to compare / aggregate runs

from tabulate import tabulate
import time
import os
import pickle
import matplotlib.pyplot as plt
import glob
import numpy as np
from datetime import datetime
from scipy.ndimage.filters import gaussian_filter1d

_time_fmt_str = '%Y-%m-%d_%H:%M:%S.%f'
_time_str_len = len(datetime.utcnow().strftime(_time_fmt_str))

class Grapher:
    # From http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
    colorblind_friendly = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    def __init__(self, names = [], parent_dir = './log'):
        self.parent_dir = parent_dir
        self.names = [os.path.join(parent_dir, name) for name in names]

    def _get_matches(self, name):
        match = os.path.join(self.parent_dir, name + '-*')
        return [s for s in glob.glob(match) if len(os.path.basename(s)) == 1 + len(name) + _time_str_len]

    def add_all(self, name):
        self.names.extend(self._get_matches(name))

    def add_last(self, name, k = 1):
        self.names.extend(sorted(self._get_matches(name))[-k:])

    def _get_data(self, y_name, x_name, smooth_sigma):
        all_xs, all_ys = [], []
        for name in self.names:
            with open(os.path.join(name, 'log.pickle'), 'rb') as pickle_f:
                table = pickle.load(pickle_f)
                xs, ys = [], []
                for step in table:
                    if x_name in step and y_name in step:
                        xs.append(step[x_name])
                        ys.append(step[y_name])
                if smooth_sigma > 0.0:
                    ys = gaussian_filter1d(ys, sigma = smooth_sigma)
                all_xs.append(xs)
                all_ys.append(ys)
        return all_xs, all_ys

    def plot(self, y_name, x_name = '_n', match_name_colors = True,
                smooth_sigma = 0.0, update_t = 1.0, **plotargs):
        palette = {}
        palette_i = 0
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        lines = []

        for name, xs, ys in zip(self.names, *self._get_data(y_name, x_name, smooth_sigma)):
            if match_name_colors:
                basename = os.path.basename(name)[:-(1 + _time_str_len)]
                if basename not in palette:
                    palette[basename] = palette_i
                    palette_i += 1
                    if palette_i >= len(self.colorblind_friendly):
                        self.colorblind_friendly.append(np.random.rand(3,))
                    line, = ax.plot(xs, ys, label = basename, color = self.colorblind_friendly[palette[basename]],
                                    **plotargs)
                else:
                    line, = ax.plot(xs, ys, color = self.colorblind_friendly[palette[basename]], **plotargs)
            else:
                line, = ax.plot(xs, ys, label = os.path.basename(name), **plotargs)
            lines.append(line)
        ax.legend()
        
        last_update = time.time()
        # Bad practice but there's no "wait for close" function
        while plt.fignum_exists(fig.number):
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1)
            if time.time() - last_update > update_t:
                last_update = time.time()
                for line, xs, ys in zip(lines, *self._get_data(y_name, x_name, smooth_sigma)):
                    line.set_xdata(xs)
                    line.set_ydata(ys)


class Log:
    def __init__(self, run_name, parent_dir = './log', autosave_freq = 30,
                 continue_last_checkpoint = False):
        """
        Will save every autosave_freq seconds, unless autosave_freq == 0.
        """
        # TODO: if continue_last_checkpoint is set to be true, continue + load last
        # saved variables. Remove info logged after last checkpoint.
        self.start_time = time.time()
        self.step_n = 0
        self.table = []
        self.extra_info = []
        self.globals = {}
        self.autosave_freq = autosave_freq
        self.last_saved = self.start_time

        self.name = run_name
        self.name_full = run_name + '-' + datetime.fromtimestamp(self.start_time).strftime(_time_fmt_str)
        self.parent_dir = parent_dir
        self.created_directory = False

    def add_globals(self, info):
        self.globals = {**self.globals, **info}

    def add_info(self, info):
        """
        This will include extra information in the upcoming step that will
        not be printed out. 
        """
        self.extra_info.append(info)

    def step(self, info):
        self.step_n += 1
        info['_n'] = self.step_n
        info['_elapsed_time'] = time.time() - self.start_time
        info['_extra_info'] = self.extra_info
        self.extra_info = []
        self.table.append(info)
        if time.time() - self.last_saved > self.autosave_freq:
            self.save()

    def require_directory(self):
        if self.created_directory:
            return
        self.created_directory = True
        self.dir_path = os.path.join(self.parent_dir, self.name_full)
        os.makedirs(self.dir_path)

    def save(self):
        self.last_saved = time.time()
        self.require_directory()
        with open(os.path.join(self.dir_path, 'log.pickle'), 'wb') as pickle_f:
            pickle.dump(self.table, pickle_f)

    def save_variables(self, session):
        # see https://www.tensorflow.org/guide/saved_model
        self.require_directory() # TODO

    def print_step(self):
        assert self.step_n > 0, 'Must step before printing a step.'
        val = [[k, v] for k, v in self.table[-1].items() if k != '_extra_info']
        val = sorted(val, key = lambda kv: kv[0])
        print(tabulate(val, tablefmt = 'fancy_grid', numalign = 'right'))

    def close(self):
        self.save()

if __name__ == '__main__':
    g = Grapher()
    g.add_all('maze-h3-ppo')
    g.add_all('maze-h1-pggae')
    g.add_all('maze-h3-pggae')
    g.plot('current maze size')
