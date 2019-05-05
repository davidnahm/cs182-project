from tabulate import tabulate
import time
import os
import dill
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import numpy as np
from datetime import datetime
from scipy.ndimage.filters import gaussian_filter1d

_time_fmt_str = '%Y-%m-%d_%H:%M:%S.%f'

class Grapher:
    # From http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
    colorblind_friendly = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    def __init__(self, names = [], parent_dir = './log'):
        self.parent_dir = parent_dir
        self.names = [os.path.join(parent_dir, name) for name in names]

    def _get_matches(self, name):
        return [s for s in glob.glob(os.path.join(self.parent_dir, name, '*'))]

    def add_all(self, name):
        self.names.extend(self._get_matches(name))

    def add_last(self, name, k = 1):
        self.names.extend(sorted(self._get_matches(name))[-k:])

    def _get_data(self, y_name, x_name, smooth_sigma):
        all_xs, all_ys = [], []
        for name in self.names:
            with open(os.path.join(name, 'log.dill'), 'rb') as dill_f:
                table = dill.load(dill_f)
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
                basename = os.path.split(name)[-2]
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
    def __init__(self, run_name, parent_dir = './log', autosave_freq = 30, autosave_vars_freq = 30,
                 continue_last_checkpoint = False, continuing = False, continuing_name = None):
        """
        Will save every autosave_freq seconds, unless autosave_freq == 0.
        If continuing_name is None we just restart the last created directory.
        """

        # These are the same whether we are loading or not
        current_time = time.time()
        self.extra_info = []
        self.autosave_freq = autosave_freq
        self.last_saved = current_time
        self.autosave_vars_freq = autosave_vars_freq
        self.last_saved_vars = current_time
        self.name = run_name
        self.parent_dir = os.path.join(parent_dir, run_name)
        self.tf_saver = None
        self.continuing = continuing

        if not continuing:
            self.start_time = current_time
            self.step_n = 0
            self.table = []
            self.params = {}

            self.current_name = datetime.fromtimestamp(self.start_time).strftime(_time_fmt_str)
            self.created_directory = False
        else:
            if continuing_name is None:
                last_found = sorted(list(glob.glob(os.path.join(parent_dir, run_name, '*'))))[-1]
                continuing_name = os.path.basename(last_found)
            self.current_name = continuing_name
            self.created_directory = True

            self.dir_path = os.path.join(self.parent_dir, self.current_name)
            with open(os.path.join(self.dir_path, 'log.dill'), 'rb') as dill_f:
                self.table = dill.load(dill_f)
            with open(os.path.join(self.dir_path, 'params.dill'), 'rb') as dill_f:
                self.params = dill.load(dill_f)

            self.step_n = len(self.table)
            # hack to make the elapsed time continue as if nothing happened
            self.start_time = current_time - self.table[-1]['_elapsed_time']

    def add_hyperparams(self, info):
        self.params = {**self.params, **info}

    def process_params(self, params):
        """
        Will replace parameters in params with their corresponding values in self.params
        """
        new_ps = {}
        for k,v in params.items():
            if k in self.params:
                new_ps[k] = self.params[k]
            else:
                new_ps[k] = v
        return new_ps

    def get_last(self, name, default):
        if len(self.table) == 0:
            return default
        if name not in self.table[-1]:
            return default
        return self.table[-1][name]

    def add_info(self, info):
        """
        This will include extra information in the upcoming step that will
        not be printed out. 
        """
        self.extra_info.append(info)

    def step(self, info, sess = None):
        self.step_n += 1
        info['_n'] = self.step_n
        info['_elapsed_time'] = time.time() - self.start_time
        info['_extra_info'] = self.extra_info
        self.extra_info = []
        self.table.append(info)
        if time.time() - self.last_saved > self.autosave_freq:
            self.save()
        if sess and time.time() - self.last_saved_vars > self.autosave_vars_freq:
            self.save_variables(sess)

    def require_directory(self):
        if self.created_directory:
            return
        self.created_directory = True
        self.dir_path = os.path.join(self.parent_dir, self.current_name)
        os.makedirs(self.dir_path)

    def save(self):
        self.last_saved = time.time()
        self.require_directory()
        with open(os.path.join(self.dir_path, 'log.dill'), 'wb') as dill_f:
            dill.dump(self.table, dill_f)
        with open(os.path.join(self.dir_path, 'params.dill'), 'wb') as dill_f:
            # 'self' is usually redundant
            self.params = {k:v for k,v in self.params.items() if dill.pickles(v) and k != 'self'}
            dill.dump(self.params, dill_f)

    def load_variables(self, session):
        if not self.tf_saver:
            self.tf_saver = tf.train.Saver()
        ckpt_name = 'model.ckpt-%d' % self.step_n
        self.tf_saver.restore(session, os.path.join(self.dir_path, ckpt_name))

    def save_variables(self, session):
        print("saving variables...")
        self.last_saved_vars = time.time()
        self.require_directory()
        if not self.tf_saver:
            self.tf_saver = tf.train.Saver()
        self.tf_saver.save(session, os.path.join(self.dir_path, 'model.ckpt'), global_step = self.step_n)

    def print_step(self):
        assert self.step_n > 0, 'Must step before printing a step.'
        val = [[k, v] for k, v in self.table[-1].items() if k != '_extra_info']
        val = sorted(val, key = lambda kv: kv[0])
        print(tabulate(val, tablefmt = 'fancy_grid', numalign = 'right'))

    def close(self, session = None):
        self.save()
        if session:
            self.save_variables(session)

if __name__ == '__main__':
    g = Grapher()
    g.add_all('maze-hyperparam-search')
    g.plot('average reward', 'simulation steps')
