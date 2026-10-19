from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_sample_size(frame, output_root):
    root = Path(output_root)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.4), layout='constrained')
    measures = [('median_profile_error', 'Median profile error\n(fixed reference IQR)'),
                ('frequency_mae', 'Frequency MAE (Hz)'),
                ('ps_mae', 'PS MAE'),
                ('within_035', 'Targets within error 0.35\n(of 54)')]
    for method, label, marker in [('neural_decoder', 'Neural decoder', 'o'),
                                   ('retrieval', 'Retrieval', 's')]:
        group = frame[frame.method.eq(method)].sort_values('training_samples')
        for axis, (column, ylabel) in zip(axes.flat, measures):
            axis.plot(group.training_samples, group[column], marker=marker, label=label)
            axis.set_xlabel('Training clips')
            axis.set_ylabel(ylabel)
            axis.set_xticks(sorted(frame.training_samples.unique()))
            axis.grid(alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('More training data; fixed 451-parameter network and 54 test targets')
    fig.savefig(root / 'sample_size.png', dpi=180, bbox_inches='tight')
    fig.savefig(root / 'sample_size.pdf', bbox_inches='tight')
    plt.close(fig)
