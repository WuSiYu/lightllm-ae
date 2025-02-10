import json
from pathlib import Path
from typing import List, Tuple
import sys
import matplotlib
from matplotlib import legend
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.axes
# plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.family"] = "Times new roman"
# plt.rcParams["font.family"] = "Verdana"

DUMPS = sys.stdin.read().strip().split('\n')

_m = lambda : {"con": [], "agg99": [], "our": []}


DATA = {
    "7b": {"o1": _m(), "long1": _m(), "long2": _m(), "long3": _m()},
    "13b": {"o1": _m(), "long1": _m(), "long2": _m(), "long3": _m()},
    "70b": {"o1": _m(), "long1": _m(), "long2": _m(), "long3": _m()},
}
for l in DUMPS:
    name, metric, val = l.split(':')
    if metric != "Overall token throughput":
        continue

    _, _, _, model, mode, dataset, clients, *_ = name.split('-')
    clients = int(clients)
    tokens_s = float(val.split(' ')[1])
    # print(model, mode, dataset, clients, tokens_s)
    DATA[model][dataset][mode].append((clients, tokens_s))

for d in DATA.values():
    for m in d.values():
        for t in m.values():
            t.sort()

print(DATA)

fig, axs = plt.subplots(len(DATA), len(next(iter(DATA.values()))), figsize=(10, 6))
axs: List[matplotlib.axes.Axes] = list(np.concatenate(axs))

xticks = {
    '7b': [10, 20, 30, 40, 60, 80, 100],
    '13b': [10, 20, 30, 40, 60, 80, 100],
    '70b': [100, 200, 300, 400, 500],
    # '70b': [200, 400, 600, 800, 1000],
}
ylims = [1200, 1200, 1200, 1200, 600, 600, 600, 600, 2500, 2500, 2500, 2500]
ds_name = {
    "o1": "Dataset: ShareGPT-o1\n(avg. input 381, avg. output 2160)",
    "long1": "Dataset: Distribution1\n(input 32~4k, output 2k~4k)",
    "long2": "Dataset: Distribution2\n(input 3k~5k, output 3k~5k)",
    "long3": "Dataset: Distribution3\n(input 2k~4k, output 32~4k)",
}
m_name = {
    '7b': "Llama2-7B-Chat",
    '13b': "Llama2-13B-Chat",
    '70b': "Llama2-70B-Chat",
}

plots = []
fig_i = 0

for model, datasets in DATA.items():
    axs[fig_i].set_ylabel("Goodput (token/s)")
    for j, (dataset, modes) in enumerate(datasets.items()):
        for mode, data in modes.items():
            title = m_name[model]
            if model == '7b':
                title = ds_name[dataset] + '\n\n' + title
            axs[fig_i].set_title(title)

            axs[fig_i].set_ylim(bottom=0, top=ylims[fig_i])
            axs[fig_i].grid(axis='y')
            data = [(c, d) for c, d in data if xticks[model][0] <= c and c <= xticks[model][-1]]
            plot, = axs[fig_i].plot([c for c, _ in data], [d for _, d in data], label=mode, marker=".")
            plots.append(plot)
            # axs[fig_i].legend()
            axs[fig_i].set_xticks(xticks[model])
        if fig_i >= 2*4:
            axs[fig_i].set_xlabel("# of clients")
        fig_i += 1


fig.legend(plots, [
    'Conservative',
    'Aggressive (water-mark=99%)',
    'Past-Future (our)',
    ],
    loc='lower center', ncol=3, fontsize=11)

fig.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig('fig7.png', dpi=300)
plt.savefig('fig7.pdf')
print(">>> figure saved to fig7.png and fig7.pdf")
