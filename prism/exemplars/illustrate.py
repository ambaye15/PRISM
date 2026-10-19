from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from .config import METRICS, FPS
from .simulation import render


def illustrate(results_dir, include_retrieval=False):
    out=Path(results_dir);data=pd.read_csv(out/'library_and_test.csv').set_index('condition_id')
    decoded=pd.read_csv(out/'decoded.csv').set_index('query_id')
    evaluation=pd.read_csv(out/'evaluation.csv')

    ids=['body_rocking__test__014','body_rocking__test__003']
    cases=[]
    for qid in ids:
        q=data.loc[qid].to_dict();q['condition_id']=qid
        n=decoded.loc[qid].to_dict();n['condition_id']=qid+'__decoded'
        rid=evaluation[(evaluation.query_id==qid)&(evaluation.method=='retrieval')].candidate_id.iloc[0]
        r=data.loc[rid].to_dict();r['condition_id']=rid
        selected=[('Held-out synthetic reference',q),('Neural example',n)]
        if include_retrieval: selected.append(('Retrieved example',r))
        for kind,row in selected:
            frames,masks=render(row)
            cases.append(dict(qid=qid,kind=kind,row=row,frames=frames,masks=masks))

    import hashlib
    for item in cases:
        h=hashlib.sha256(item['frames'].tobytes()+item['masks'].tobytes()).hexdigest()
        assert h==item['row']['image_sha256']
    font=ImageFont.truetype(__import__('matplotlib.font_manager',fromlist=['findfont']).findfont('DejaVu Sans'),18)
    small=ImageFont.truetype(__import__('matplotlib.font_manager',fromlist=['findfont']).findfont('DejaVu Sans'),14)
    title=ImageFont.truetype(__import__('matplotlib.font_manager',fromlist=['findfont']).findfont('DejaVu Sans'),22)
    columns=3 if include_retrieval else 2
    neural_cases=[item for item in cases if item['kind']=='Neural example']
    tile_w,tile_h=384,330
    frames_out=[]
    for i in range(min(len(x['frames']) for x in cases)):
        canvas=Image.new('RGB',(tile_w*columns,tile_h*2+92),'#f4f7fb');d=ImageDraw.Draw(canvas)
        d.text((18,12),'From a PRISM profile to illustrative synthetic movement',font=title,fill='#142c42')
        d.text((18,44),'Two held-out rocking profiles | every displayed example is measured through PRISM',font=small,fill='#475569')
        for j,item in enumerate(cases):
            x=(j%columns)*tile_w;y=75+(j//columns)*tile_h
            d.rectangle((x+5,y,x+tile_w-5,y+tile_h-7),fill='white')
            d.text((x+14,y+10),item['kind'],font=font,fill='#142c42')
            row=item['row'];v=[row[k] for k in METRICS]
            d.text((x+14,y+40),f'I {v[0]:.3f}   E {v[1]:.3f}   f {v[2]:.3f} Hz',font=small,fill='#334155')
            d.text((x+14,y+61),f'PS {v[3]:.3f}   |   t = {i/FPS:.2f} s',font=small,fill='#334155')
            frame=cv2.cvtColor(cv2.resize(item['frames'][i],(tile_w,216)),cv2.COLOR_BGR2RGB)
            canvas.paste(Image.fromarray(frame),(x,y+92))
        frames_out.append(np.asarray(canvas))
    video=out/'profile_to_motion.mp4'
    with imageio.get_writer(video,fps=FPS,codec='libx264',quality=8,macro_block_size=2) as w:
        for f in frames_out:w.append_data(f)
    Image.fromarray(frames_out[0]).save(out/'profile_to_motion_preview.png')

    fig,axs=plt.subplots(2,3,figsize=(7.2,3.5),layout='constrained')
    for rowidx,item in enumerate(neural_cases):
        masks=item['masks'];union=masks.any(axis=0);yy,xx=np.where(union)
        left,right=max(0,xx.min()-12),min(256,xx.max()+13)
        top,bottom=max(0,yy.min()-8),min(144,yy.max()+9)
        for col,sec in enumerate([0.,.25,.5]):
            ax=axs[rowidx,col];frame=item['frames'][int(round(sec*FPS))]
            ax.imshow(cv2.cvtColor(frame[top:bottom,left:right],cv2.COLOR_BGR2RGB));ax.axis('off')
            ax.set_title(f't = {sec:.2f} s',fontsize=8)
        q=data.loc[item['qid']];r=item['row']
        axs[rowidx,0].text(-.1,.5,f'Profile {"A" if rowidx==0 else "B"}',transform=axs[rowidx,0].transAxes,
                           rotation=90,ha='center',va='center',fontsize=9)
        label=f'Target f={q.frequency:.2f}, PS={q.ps:.2f}; example f={r["frequency"]:.2f}, PS={r["ps"]:.2f}'
        axs[rowidx,1].text(.5,-.06,label,transform=axs[rowidx,1].transAxes,ha='center',va='top',fontsize=7)
    fig.savefig(out/'profile_exemplars.pdf',bbox_inches='tight');fig.savefig(out/'profile_exemplars.png',dpi=170,bbox_inches='tight');plt.close(fig)

    from skeleton_lab.rig import BONES
    from skeleton_lab.simulate import simulate
    from skeleton_lab.spec import ScenarioSpec
    from .simulation import payload
    fig,axs=plt.subplots(1,2,figsize=(3.3,1.6),layout='constrained')
    for idx,item in enumerate(neural_cases):
        result=simulate(ScenarioSpec.model_validate(payload(item['row'])))
        uv=result.uv_clean
        for j,sec in enumerate([0.,.25,.5]):
            frame=int(round((2.+sec)*FPS))
            for parent,child,_ in BONES:
                a,b=uv[parent][frame],uv[child][frame]
                axs[idx].plot([a[0],b[0]],[a[1],b[1]],color=['#cbd5e1','#7ea3b8','#164b63'][j],lw=1.)
            axs[idx].add_patch(plt.Circle(tuple(uv['head_center'][frame]),18.9,fill=False,
                color=['#cbd5e1','#7ea3b8','#164b63'][j],lw=1.))
        axs[idx].set_aspect('equal');axs[idx].invert_yaxis();axs[idx].axis('off')
        row=item['row'];q=data.loc[item['qid']]
        axs[idx].set_title(f'Profile {"A" if idx==0 else "B"}',fontsize=8)
        axs[idx].text(.5,-.01,f'f {row["frequency"]:.2f} Hz | PS {row["ps"]:.2f}',transform=axs[idx].transAxes,
                      ha='center',va='top',fontsize=7)
    fig.savefig(out/'profile_compact.pdf',bbox_inches='tight');fig.savefig(out/'profile_compact.png',dpi=180,bbox_inches='tight');plt.close(fig)

    methods=['retrieval','without_ps','neural_decoder','random_100_mean']
    fig,ax=plt.subplots(figsize=(3.3,2.3),layout='constrained')
    values=[evaluation.loc[evaluation.method==m,'profile_error'].to_numpy() for m in methods]
    ax.boxplot(values,tick_labels=['4-metric\nretrieval','Without\nPS','Neural\ndecoder','Random\nbehavior'],showfliers=True)
    ax.set_ylabel('Profile error (library IQR)',fontsize=8);ax.tick_params(axis='both',labelsize=7)
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'profile_error.pdf',bbox_inches='tight');fig.savefig(out/'profile_error.png',dpi=170,bbox_inches='tight');plt.close(fig)
    saved=[dict(query_id=x['qid'],kind=x['kind'],condition_id=x['row']['condition_id'],
                 **{k:float(x['row'][k]) for k in METRICS}) for x in cases]
    (out/'illustrated_examples.json').write_text(json.dumps(saved,indent=2))
    return video


