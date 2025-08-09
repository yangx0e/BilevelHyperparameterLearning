import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
plt.rcParams['font.size'] = 16
plt.rc('legend', fontsize=14)
plt.rcParams['lines.linewidth'] = 3
msz = 14
handlelength = 2.25     # 2.75
borderpad = 0.25     # 0.15

linestyle_tuples = {
     'solid':                 '-',
     'dashdot':               '-.',
     
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),
     
     'long dash with offset': (5, (10, 3)),
     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}

marker_list = ['o', 'd', 's', 'v', 'X', "*", "P", "^"]
style_list = ['-', '-', '-.',  linestyle_tuples['dotted'], linestyle_tuples['densely dashdotted'],
              linestyle_tuples['densely dashed'], linestyle_tuples['densely dashdotdotted']]


my_grid = np.load("my_grid.npy")
sweep_history = np.load("sweep_history.npy")


# idx_all = np.arange(sweep_history.shape[0])
idx_all = [0, 4, 9, 14, 19, 24, 29]

legs = [r"k=1", r"k=5", r"k=10", r"k=15",r"k=20", r"k=25", r"k=30"]
plt.close()
plt.figure(0)
plt.axvline(x=0.2, color='C7', ls="--")
for i, idx in enumerate(idx_all):
    print(idx)
    plt.loglog(my_grid, sweep_history[idx,...], ls=style_list[i], label=legs[i])
plt.grid(True, which="both")
plt.xlabel(r'Lengthscale')
plt.ylabel(r'Outer Objective')
plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.tight_layout()
plt.savefig("sweep_elliptic.pdf", format='pdf')