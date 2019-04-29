import random
import matplotlib.pyplot as plt

# These are just some utilities to heuristically determine what makes
# a good schedule.

def gamma_run_length(trigger, percent_active, gamma):
    x = 0
    n = 0
    while x < trigger:
        n += 1
        x *= gamma
        if random.random() < percent_active:
            x += 1 - gamma
    return n

def gamma_dist(n = 100, trigger = 0.9, percent_active = 0.9, gamma = 0.95):
    ns = []
    for i in range(n):
        ns.append(gamma_run_length(trigger, percent_active, gamma))
    plt.hist(ns, bins = 20)
    plt.show()
    print('min: %d, max: %d' % (min(ns), max(ns)))
    print('avg:', sum(ns) / len(ns))
