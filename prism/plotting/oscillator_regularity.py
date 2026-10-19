from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def plot_results(data,root):
    for (behavior,family),subset in data.groupby(['behavior','family']):
        fig,axes=plt.subplots(1,2,figsize=(7,3),sharey=True)
        for ax,mode in zip(axes,['reference','estimated']):
            for seed,g in subset[subset.period_mode.eq(mode)].groupby('seed'):
                g=g.sort_values('irregularity');ax.plot(g.irregularity,g.prism_periodicity,'o-',label=f'Seed {seed}',lw=1,ms=3)
            ax.set_title(mode.capitalize()+' period');ax.set_xlabel('Disruption level' if family=='phase_disruption' else 'Secondary rhythm weight')
            ax.spines[['top','right']].set_visible(False)
        axes[0].set_ylabel('Production flow PS');axes[1].legend(frameon=False,fontsize=7)
        fig.suptitle(behavior.replace('_',' ')+' — '+family.replace('_',' '));fig.tight_layout()
        label_panels(fig, axes)
        for ext in ['pdf','png']:fig.savefig(Path(root)/(behavior+'__'+family+'.'+ext),dpi=250,bbox_inches='tight')
        plt.close(fig)
