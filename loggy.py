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

time_fmt_str = '%Y-%m-%dT%H:%M:%S'
time_str_len = len(time.strftime(time_fmt_str, time.localtime()))

class Grapher:
    # From http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
    colorblind_friendly = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    def __init__(self, names = [], parent_dir = './log'):
        self.parent_dir = parent_dir
        self.names = [os.path.join(parent_dir, name) for name in names]

    def _get_matches(self, name):
        match = os.path.join(self.parent_dir, name + '-*')
        return [s for s in glob.glob(match) if len(os.path.basename(s)) == 1 + len(name) + time_str_len]

    def add_all(self, name):
        self.names.extend(self._get_matches(name))

    def add_last(self, name):
        self.names.append(sorted(self._get_matches(name))[-1])

    def plot(self, y_name, x_name = '_n', match_name_colors = True):
        palette = {}
        palette_i = 0
        for name in self.names:
            with open(os.path.join(name, 'log.pickle'), 'rb') as pickle_f:
                table = pickle.load(pickle_f)
                xs, ys = [], []
                for step in table:
                    if x_name in step and y_name in step:
                        xs.append(step[x_name])
                        ys.append(step[y_name])
                if match_name_colors:
                    basename = os.path.basename(name)[:-(1 + time_str_len)]
                    if basename not in palette:
                        palette[basename] = palette_i
                        palette_i += 1
                        if palette_i >= len(self.colorblind_friendly):
                            self.colorblind_friendly.append(np.random.rand(3,1))
                        plt.plot(xs, ys, label = basename, color = self.colorblind_friendly[palette[basename]])
                    else:
                        plt.plot(xs, ys, color = self.colorblind_friendly[palette[basename]])
                else:
                    plt.plot(xs, ys, label = os.path.basename(name))
        
        plt.legend()
        plt.show()

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
        self.name_full = run_name + '-' + time.strftime(time_fmt_str, time.localtime(self.start_time))
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
        self.require_directory() # TODO

    def print_step(self):
        assert self.step_n > 0, 'Must step before printing a step.'
        val = [[k, v] for k, v in self.table[-1].items() if k != '_extra_info']
        print(tabulate(val, tablefmt = 'fancy_grid', numalign = 'right'))

    def close(self):
        self.save()

if __name__ == '__main__':
    g = Grapher()
    g.add_all('test-gae')
    g.add_all('test-vanilla')
    g.plot('average reward', 'simulation steps')
