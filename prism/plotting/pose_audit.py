
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from string import ascii_lowercase
from matplotlib.transforms import ScaledTranslation

def label_panels(fig, axes, fmt='({})', inside=False, dx=-0.35, dy=0.15, **kw):

    style = dict(fontsize=11, fontweight='bold', va='bottom', ha='left')
    style.update(kw)
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        letter = fmt.format(ascii_lowercase[i])
        if inside:
            ax.text(0.02, 0.98, letter, transform=ax.transAxes, **{**style, 'va': 'top'})
        else:
            offset = ScaledTranslation(dx, dy, fig.dpi_scale_trans)
            ax.text(0.0, 1.0, letter, transform=ax.transAxes + offset, **style)

def plot_audit(data, output_root):
    data=data.loc[data.threshold.eq(.5)]
    datasets=data.dataset.unique().tolist()
    metrics=[('coverage','Observed joints'),('displacement_gt_0.25','Large steps >.25 scales'),
             ('jump_return_fraction','Jump-and-return event fraction'),
             ('bone_change_gt_25pct','Bone-change candidates')]
    metrics=[(c,l) for c,l in metrics if c in data]
    fig,axes=plt.subplots(1,len(metrics),figsize=(4*len(metrics),3.5),squeeze=False)
    for ax,(column,label) in zip(axes[0],metrics):
        values=[data.loc[data.dataset.eq(name),column].dropna().to_numpy()*100 for name in datasets]
        ax.boxplot(values,showfliers=False)
        ax.set_xticks(range(1,len(datasets)+1),datasets,rotation=25,fontsize=7)
        ax.set_title(label,fontsize=9);ax.set_ylabel('%');ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    label_panels(fig, axes)
    for ext in ['pdf','png']:fig.savefig(Path(output_root)/('pose_quality_diagnostics.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
