from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_results(primary, compensation, output_root):
    out = Path(output_root)
    baseline = primary[primary.epsilon == 0].set_index('seed')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, key, label in zip(axes[0], ['intensity', 'extent', 'frequency'],
                              ['Intensity change (%)', 'Silhouette extent change (%)', 'Estimated frequency change (%)']):
        for seed, group in primary.groupby('seed'):
            ax.plot(group.epsilon, 100 * (group[key] / baseline.loc[seed, key] - 1), alpha=.65, marker='o', ms=3)
        ax.axhline(0, color='k', ls='--', lw=.6)
        ax.set(xlabel='Phase modulation (radians)', ylabel=label, ylim=(-.01, .01))
    for ax, key, label in zip(axes[1], ['ps', 'cycle_return_distance', 'spectral_concentration'],
                              ['All-window PS', 'Cycle-return distance', 'Spectral concentration']):
        for _, group in primary.groupby('seed'):
            ax.plot(group.epsilon, group[key], alpha=.65, marker='o', ms=3)
        ax.set(xlabel='Phase modulation (radians)', ylabel=label, ylim=(0, 1))
    pure = compensation.groupby('epsilon').compensation_ps_shared_period.mean()
    axes[1, 0].plot(np.r_[0, pure.index], np.r_[baseline.ps.mean(), pure.values], 'k--', lw=1.5,
                    label='Adjusted geometry, no modulation')
    axes[1, 0].legend(fontsize=7)
    for suffix in ['png', 'pdf']:
        fig.savefig(out / ('matched_controls.' + suffix), dpi=170)
    plt.close(fig)
