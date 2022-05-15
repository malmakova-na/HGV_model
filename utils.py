import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt


def drop_duplicates(v1, v2):
    idxs = np.where(v1[:-1] == v1[1:])[0]
    v1_new = np.delete(v1, idxs)
    v2_new = np.delete(v2, idxs)
    return v1_new, v2_new

def plot_pair(v1, v2, save_path, xlabel, ylabel, title):
    v1, v2 = drop_duplicates(v1.reshape(-1), v2.reshape(-1))
    f = interpolate.interp1d(v1, v2, kind = 'cubic')
    v1_inter = np.linspace(v1[0], v1[-1], 100)
    plt.figure(figsize=(12, 7))
    plt.plot(v1_inter, f(v1_inter))
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=15)
    plt.savefig(save_path)