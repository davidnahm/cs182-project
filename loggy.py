# should:
# save newest model
# log various statistics + timestamps
# log parameters
# copy code
# have a mechanism to compare / aggregate runs

from tabulate import tabulate
import time

class Grapher:
    def __init__(self):
        pass

    def select_all(self):
        pass

    def select_last(self):
        pass

    def plot_values(self, logs, values, with_markers):
        pass

class Log:
    def __init__(self, run_name, directory = './log', autosave = True,
                 continue_last_checkpoint = False):
        # TODO: if continue_last_checkpoint is set to be true, continue + load last
        # saved variables. Remove info logged after last checkpoint.
        # TODO: timestamped directory
        self.step_n = 0
        self.start_time = time.time()
        self.table = []
        self.extra_info = []
        self.globals = {}
        self.autosave = autosave

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
        if self.autosave:
            self.save()

    def save(self):
        pass # TODO

    def save_variables(self, session):
        pass # TODO

    def print_step(self):
        assert self.step_n > 0, 'Must step before printing a step.'
        val = [[k, v] for k, v in self.table[-1].items() if k != '_extra_info']
        print(tabulate(val, tablefmt = 'fancy_grid', numalign = 'right'))

    def close(self):
        pass # TODO
