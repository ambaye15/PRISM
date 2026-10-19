import re
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from string import ascii_lowercase
from matplotlib.transforms import ScaledTranslation

NAMES = ['intensity','extent','frequency','periodicity','duration','interbout']
LABELS = ['Intensity (body heights/s)','Extent (mask RMS)','Frequency (Hz)',
          'Periodicity','Duration (s)','IBI (s)']
COLORS = {'flap':'#4477AA','flap_rock':'#EE7733','rock':'#228833'}

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

def date_of(session):
    s=str(session)

    m=re.search(r'(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)',s)
    if m:return date.fromisoformat(m[1]).isoformat()
    m=re.search(r'(?<!\d)(\d{2})-(\d{2})-(\d{2})(?!\d)',s)
    return date(2000+int(m[3]),int(m[1]),int(m[2])).isoformat() if m else s


def plot_sessions(data, output_root, participants, minimum_n=3):
    data=data.loc[data.participant_id.isin(participants)].copy()
    fig,axes=plt.subplots(len(participants),6,figsize=(15,3.2*len(participants)),squeeze=False)
    for row,participant in enumerate(participants):
        g=data.loc[data.participant_id.eq(participant)].copy()
        parsed=g.session_id.map(date_of)
        if 'session_date' in g:
            parsed=parsed.where(parsed.ne(g.session_id.astype(str)),g.session_date.fillna(parsed))
        g['session_date']=parsed
        g=g.sort_values(['session_date','session_id'])
        sessions=g.session_id.drop_duplicates().tolist()
        lookup={name:i for i,name in enumerate(sessions)}
        dates=g.drop_duplicates('session_id').set_index('session_id').session_date
        repeated=dates.value_counts()
        labels=[]
        for session in sessions:
            label=str(dates[session])
            if repeated[dates[session]]>1:
                run=re.search(r'_(\d+)(?:_|$)',str(session))
                label+='\nrun '+run[1] if run else '\n'+str(session)
            labels.append(label)
        for col,measure in enumerate(NAMES):
            ax=axes[row,col]
            q=g.loc[g.measure.eq(measure) & g.n.ge(minimum_n)]
            for behavior,color in COLORS.items():
                z=q.loc[q.behavior.eq(behavior)].copy()

                groups=z.groupby('study_id') if 'study_id' in z else [('unknown',z)]
                for study,part in groups:
                    x=part.session_id.map(lookup).to_numpy()
                    y=part['median'].to_numpy()
                    laboratory=str(study).lower() in {'1','1.0','study1','study_1','lab','laboratory'}
                    marker='s' if laboratory else 'o'
                    ax.errorbar(x,y,yerr=np.vstack([y-part.q25,part.q75-y]),fmt=marker,
                                color=color,capsize=2,markersize=4,label=behavior)
                    if study!='unknown':ax.plot(x,y,color=color,lw=.9)

            ax.set_title(LABELS[col],fontsize=9)
            ax.set_xticks(range(len(sessions)),labels,rotation=55,fontsize=6)
            ax.tick_params(axis='y',labelsize=7);ax.spines[['top','right']].set_visible(False)
            if col==0:ax.set_ylabel(participant)
            if measure=='interbout':ax.set_yscale('symlog',linthresh=1)
            if not len(q):ax.text(.5,.5,'Insufficient eligible bouts',ha='center',transform=ax.transAxes,fontsize=7)
    handles,labels=axes[0,0].get_legend_handles_labels()
    unique=dict(zip(labels,handles))
    if unique:fig.legend(unique.values(),unique.keys(),loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,0,1,.94])
    label_panels(fig, axes)
    for suffix in ['pdf','png']:fig.savefig(Path(output_root)/('p001_p004_session_profiles.'+suffix),dpi=300,bbox_inches='tight')
    plt.close(fig)


def plot_pairs(selected, output_root):
    show=selected.groupby('anchor',sort=False).head(1)
    if show.empty:return
    fig,axes=plt.subplots(1,len(show),figsize=(3.4*len(show),3.2),squeeze=False)
    names=NAMES[:4];x=np.arange(4)
    for ax,(_,pair) in zip(axes[0],show.iterrows()):
        ax.bar(x-.16,[pair[n+'_rank_a'] for n in names],.3,color='#4477AA',label='A')
        ax.bar(x+.16,[pair[n+'_rank_b'] for n in names],.3,color='#CC6677',label='B')
        anchor=names.index(pair.anchor);ax.axvspan(anchor-.4,anchor+.4,color='#999999',alpha=.15)
        ax.set_xticks(x,names,rotation=35,fontsize=8);ax.set_ylim(0,1)
        ax.set_title('Matched '+pair.anchor,fontsize=10);ax.spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel('Within-participant/behavior percentile');axes[0,-1].legend(frameon=False)
    fig.tight_layout()
    label_panels(fig, axes)
    for suffix in ['pdf','png']:fig.savefig(Path(output_root)/('matched_coordinate_pairs.'+suffix),dpi=300,bbox_inches='tight')
    plt.close(fig)
