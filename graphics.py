import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

def plot_features(data, targets, t=None, fig=None, categorical=False):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        t = t.detach().cpu().numpy()


    fig2 = fig if fig else plt.figure(figsize=(8, 8))
    spec2 = gridspec.GridSpec(ncols=data.shape[1], nrows=data.shape[1], figure=fig2)

    for d in range(data.shape[1]):
        for d2 in range(data.shape[1]):
            if d > d2:
                continue
            sub_ax = fig2.add_subplot(spec2[d, d2])
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])

            if d == d2:
                if np.var(data[:, d]) == 0:
                    continue
                sns.histplot(data[:, d], kde=True, ax=sub_ax, stat="density", linewidth=0)
                sub_ax.set(ylabel=None)
            else:
                if t is not None:
                    sns.scatterplot(x=data[:, d], y=data[:, d2], hue=t, palette={0: 'blue', 1: 'red'}, ax=sub_ax, legend=False)
                else:
                    sns.scatterplot(x=data[:, d], y=data[:, d2], ax=sub_ax, legend=False)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()