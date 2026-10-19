import argparse
import csv
import hashlib
import json
import subprocess
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import aqsm_records, sam_records, BONES, PARTS

KINDS = ['jump_and_return', 'large_displacement', 'missingness',
         'score_drop', 'bone_change', 'acceleration']
JUDGMENTS = ['clear_pose_failure', 'genuine_movement', 'camera_or_projection',
             'uncertain', 'no_visible_issue', 'insufficient_video']

def candidates(p, c, scale, fps, threshold):

    n = len(p)
    valid = np.isfinite(p).all(-1) & np.isfinite(c) & (c >= threshold) & ~np.all(p == 0, -1)
    pair = valid[:-1] & valid[1:]
    masks = {k: np.zeros(n, bool) for k in KINDS}

    masks['missingness'] = valid.sum(1) < 8
    score_pair = np.isfinite(c[:-1]) & np.isfinite(c[1:])
    masks['score_drop'][1:] = (score_pair & ((c[:-1] - c[1:]) > .4)).any(1)
    if not np.isfinite(scale) or scale <= 0:
        return masks
    d = np.diff(p, axis=0) / scale
    step = np.linalg.norm(d, axis=-1)
    masks['large_displacement'][1:] = (pair & (step > .25)).any(1)
    triple = pair[:-1] & pair[1:]
    spike = triple & (step[:-1] > .15) & (step[1:] > .15) & (np.linalg.norm((p[2:] - p[:-2]) / scale, axis=-1) < .05)
    masks['jump_and_return'][1:-1] = spike.any(1)
    acc = np.linalg.norm(np.diff(d, axis=0), axis=-1) * fps ** 2
    masks['acceleration'][1:-1] = (triple & (acc > 50)).any(1)
    for a, b in BONES:
        good = valid[:, a] & valid[:, b]
        length = np.linalg.norm(p[:, a] - p[:, b], axis=1) / scale
        keep = good[:-1] & good[1:] & (length[:-1] > 1e-8)
        change = np.abs(np.diff(length)) / np.maximum(length[:-1], 1e-8)
        masks['bone_change'][1:] |= keep & (change > .25)
    return masks

def stable_seed(seed, key):
    return seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)

def prepare(args):
    out = Path(args.output_root)
    if (out/'review_key.csv').exists():
        raise ValueError('Review already exists. Choose a new output folder to preserve labels.')
    generators = []
    if args.aqsm_pickle:
        generators.append(aqsm_records(args.aqsm_pickle, 1.0))
    if args.sam_features:
        if not args.sam_tracks_root:
            raise ValueError('--sam-tracks-root is required with --sam-features')
        generators.append(sam_records(args.sam_features, args.sam_tracks_root))
    if not generators:
        raise ValueError('Provide AQSM pickle and/or SAM features with tracks.')
    pool = []
    settings = {k:v for k,v in vars(args).items() if k != 'func'}
    for records in generators:
        for p, c, scale, fps, meta in records:
            if meta['behavior'] not in ['rock', 'flap', 'flap_rock'] or len(p) < 3:
                continue
            if not np.isfinite(fps) or fps <= 0:
                continue
            valid=np.isfinite(p).all(-1)&np.isfinite(c)&(c>=args.threshold)&~np.all(p==0,-1)
            pairs=valid[:-1]&valid[1:]
            n_pairs=int(pairs.sum());n_triplets=int((pairs[:-1]&pairs[1:]).sum())
            if args.flag_kind in ['jump_and_return','acceleration'] and n_triplets==0:continue
            if args.flag_kind in ['large_displacement','bone_change'] and n_pairs==0:continue
            flags = candidates(p, c, scale, fps, args.threshold)
            selected = np.flatnonzero(flags[args.flag_kind])
            anyflag = np.logical_or.reduce(list(flags.values()))

            pool.append(dict(meta, fps=float(fps), n_frames=len(p),
                             is_flagged=bool(len(selected)),
                             frames=selected.tolist(), anyflag=anyflag.tolist(),
                             has_scale=bool(np.isfinite(scale) and scale > 0),
                             n_observed_joint_pairs=n_pairs,n_observed_joint_triplets=n_triplets))
    if not pool:
        raise ValueError('No eligible bouts found.')
    chosen = []
    eligible_counts = []
    for dataset, behavior in sorted({(r['dataset'], r['behavior']) for r in pool}):
        group = [r for r in pool if r['dataset']==dataset and r['behavior']==behavior]
        rng = np.random.default_rng(stable_seed(args.seed, dataset+'|'+behavior))

        if args.flag_kind in ['jump_and_return','large_displacement','bone_change','acceleration']:
            group = [r for r in group if r['has_scale']]
        flagged = [r for r in group if r['is_flagged']]
        controls = [r for r in group if not r['is_flagged']]
        if args.strict_controls:
            controls = [r for r in controls if not any(r['anyflag'])]
        eligible_counts.append({'dataset':dataset,'behavior':behavior,
                                'flagged_pool':len(flagged),'control_pool':len(controls)})
        rng.shuffle(flagged); rng.shuffle(controls)
        for sample_group, rows in [('flagged', flagged), ('control', controls)]:
            for row in rows[:args.per_stratum]:
                row = row.copy()
                centers = row['frames'] if sample_group=='flagged' else list(range(1,row['n_frames']-1))
                center = int(rng.choice(centers))
                row['center_local_frame'] = center
                row['center_source_frame'] = int(row['start_frame']) + center
                row['center_time_sec'] = row['center_source_frame']/row['fps']
                row['sample_group'] = sample_group
                row['flag_kind'] = args.flag_kind
                row['threshold'] = args.threshold
                row['context_sec'] = args.context_sec
                row.pop('frames');row.pop('anyflag')
                chosen.append(row)
    if not chosen:
        raise ValueError('No review items selected.')
    rng = np.random.default_rng(args.seed); rng.shuffle(chosen)
    out.mkdir(parents=True,exist_ok=True)
    (out/'tracks').mkdir(exist_ok=True)
    wanted = {(r['dataset'],r['unit_id']):r for r in chosen}
    for i, row in enumerate(chosen, 1):
        row['review_id'] = f'R{i:04d}'
        row['clip_path'] = 'clips/'+row['review_id']+'.mp4'

    generators = []
    if args.aqsm_pickle: generators.append(aqsm_records(args.aqsm_pickle, 1.0))
    if args.sam_features: generators.append(sam_records(args.sam_features,args.sam_tracks_root))
    for records in generators:
        for p,c,scale,fps,meta in records:
            row = wanted.get((meta['dataset'],meta['unit_id']))
            if row is None: continue
            radius = max(2,int(np.ceil(args.context_sec*fps)))
            center = row['center_local_frame']
            lo=max(0,center-radius); hi=min(len(p),center+radius+1)
            row['clip_start_source_frame']=int(meta['start_frame'])+lo
            row['clip_end_source_frame_exclusive']=int(meta['start_frame'])+hi
            row['center_clip_frame']=center-lo
            np.savez_compressed(out/'tracks'/(row['review_id']+'.npz'),xy=p[lo:hi],confidence=c[lo:hi])
    pd.DataFrame(chosen).to_csv(out/'review_key.csv',index=False)
    pd.DataFrame(eligible_counts).to_csv(out/'sampling_counts.csv',index=False)
    pd.DataFrame([{'review_id':r['review_id'],'judgment':'','failure_type':'','notes':''} for r in chosen]).to_csv(out/'ratings_template.csv',index=False)
    recordings=sorted({r['recording'] for r in chosen})
    pd.DataFrame([{'recording':r,'video_path':'','source_time_offset_sec':0,'source_time_scale':1} for r in recordings]).to_csv(out/'video_map.csv',index=False)
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    print(f'Selected {len(chosen)} items. Fill video_map.csv if filenames do not match recording IDs.')
    print(pd.DataFrame(eligible_counts).to_string(index=False))

def resolve_video(recording, root, mapping):
    hit = mapping[mapping.recording.astype(str)==str(recording)] if len(mapping) else pd.DataFrame()
    if len(hit)>1: raise ValueError(f'Ambiguous video-map rows: {recording}')
    offset,scale=0.,1.
    if len(hit):
        row=hit.iloc[0];offset=float(row.source_time_offset_sec);scale=float(row.source_time_scale)
        if not np.isfinite(scale) or scale<=0 or not np.isfinite(offset):raise ValueError('Invalid time mapping')
        if pd.notna(row.video_path) and str(row.video_path).strip():
            p=Path(str(row.video_path)).expanduser()
            if not p.is_absolute():p=Path(root)/p
            if not p.is_file():raise FileNotFoundError(p)
            return p,offset,scale
    hits=[p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in ['.avi','.mp4','.mov','.mkv','.m4v'] and p.stem==str(recording)]
    if len(hits)!=1:raise ValueError(f'{recording}: found {len(hits)} exact filename matches; fill video_map.csv')
    return hits[0],offset,scale

def overlay(frame, xy, confidence, threshold, normalized):
    import cv2
    im=frame.copy();h,w=im.shape[:2]
    points=xy.copy()
    if normalized:points*=np.array([w,h])
    finite=np.isfinite(points).all(-1)&~np.all(xy==0,-1)
    observed=finite&np.isfinite(confidence)&(confidence>=threshold)
    for a,b in BONES:
        if finite[a] and finite[b]:
            color=(190,190,190) if not(observed[a] and observed[b]) else (230,115,30)
            pts=np.clip(points[[a,b]],-10000,10000).astype(int)
            cv2.line(im,tuple(pts[0]),tuple(pts[1]),color,2,cv2.LINE_AA)
    for j in np.flatnonzero(finite):
        point=tuple(np.clip(points[j],-10000,10000).astype(int))
        cv2.circle(im,point,3,(230,115,30) if observed[j] else (190,190,190),-1,cv2.LINE_AA)
    return im

def render(args):
    import cv2
    out=Path(args.output_root);key=pd.read_csv(out/'review_key.csv')
    mapping=pd.read_csv(args.video_map or out/'video_map.csv')
    if not shutil.which('ffmpeg'):raise RuntimeError('ffmpeg required. On macOS: brew install ffmpeg')
    (out/'clips').mkdir(exist_ok=True);statuses=[]
    for _,row in key.iterrows():
        dest=out/row.clip_path
        try:
            path,offset,time_scale=resolve_video(row.recording,args.video_root,mapping)
            cap=cv2.VideoCapture(str(path));nativefps=cap.get(cv2.CAP_PROP_FPS)
            count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not cap.isOpened() or nativefps<=0:raise ValueError('Could not open source video')
            data=np.load(out/'tracks'/(row.review_id+'.npz'));xy=data['xy'];confidence=data['confidence']
            writer=None;temp=out/'clips'/(row.review_id+'_temp.avi')
            first=None;last=None
            for i in range(len(xy)):
                pose_time=(int(row.clip_start_source_frame)+i)/float(row.fps)
                source_time=offset+time_scale*pose_time
                frame_index=int(round(source_time*nativefps))
                if frame_index<0 or frame_index>=count:raise ValueError('Mapped timestamp outside source video; verify video_map.csv')
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);ok,frame=cap.read()
                if not ok:raise ValueError(f'Could not decode source frame {frame_index}')
                pose=overlay(frame,xy[i],confidence[i],row.threshold,row.dataset=='AQSM_MediaPipe')
                scale=min(1.,args.max_panel_width/frame.shape[1]);size=(int(frame.shape[1]*scale),int(frame.shape[0]*scale))
                size=(max(2,size[0]//2*2),max(2,size[1]//2*2))
                raw=cv2.resize(frame,size);pose=cv2.resize(pose,size)
                canvas=np.hstack([raw,pose]);canvas=cv2.copyMakeBorder(canvas,46,0,0,0,cv2.BORDER_CONSTANT,value=(250,250,250))
                text=f'{row.review_id} | source {source_time:.3f}s | pose {float(row.fps):.2f} fps | raw RGB / pose overlay'
                cv2.putText(canvas,text,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1,cv2.LINE_AA)
                if writer is None:
                    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'MJPG'),float(row.fps)/args.slowdown,(canvas.shape[1],canvas.shape[0]))
                    if not writer.isOpened():raise ValueError('Video writer failed')
                    first=source_time
                writer.write(canvas);last=source_time
            writer.release();cap.release()
            subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(temp),'-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(dest)],check=True)
            temp.unlink()
            native_dest=out/'clips'/(row.review_id+'_native.mp4')
            duration=last-first+time_scale/float(row.fps)
            subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(first),'-i',str(path),'-t',str(duration),'-an','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(native_dest)],check=True)
            statuses.append({'review_id':row.review_id,'status':'ok','video_path':str(path),'first_source_sec':first,'last_source_sec':last,'native_fps':nativefps,'source_width_px':cap.get(cv2.CAP_PROP_FRAME_WIDTH) if cap.isOpened() else frame.shape[1],'source_height_px':frame.shape[0],'source_aspect':frame.shape[1]/frame.shape[0],'pose_fps':row.fps,'slowdown':args.slowdown,'error':''})
        except Exception as exc:
            if 'writer' in locals() and writer is not None:writer.release()
            if 'cap' in locals():cap.release()
            dest.unlink(missing_ok=True)
            (out/'clips'/(row.review_id+'_native.mp4')).unlink(missing_ok=True)
            statuses.append({'review_id':row.review_id,'status':'failed','error':str(exc)})
        print(row.review_id,statuses[-1]['status'],statuses[-1].get('error',''),flush=True)
    pd.DataFrame(statuses).to_csv(out/'render_status.csv',index=False)
    write_html(out,[r for r in statuses if r['status']=='ok'])
    print('Open review.html in your browser. Save ratings.csv when finished.')
    if any(r['status']!='ok' for r in statuses):print('Some clips failed: see render_status.csv; do not count them as pose failures.')

def write_html(out, rendered):
    ids=[r['review_id'] for r in rendered]

    html='''<!doctype html><meta charset="utf-8"><title>Pose review</title>
<style>body{font:16px system-ui;max-width:1120px;margin:30px auto;padding:15px;background:#f5f7f9}video{width:100%;background:#222}button,select,input,textarea{font:inherit;padding:9px;margin:5px}textarea{display:block;width:95%}#status{color:#555}h1{font-size:24px}</style>
<h1>Pose observation review</h1><p>Compare source RGB with the unsmoothed pose overlay. Classify the visible evidence. Gray points are below the score threshold. Check synchronization before attributing a discrepancy to pose.</p>
<div id="status"></div><video id="video" controls loop></video><details><summary>Native-rate source video (no pose overlay)</summary><video id="native" controls loop></video></details>
<p><select id="judgment"><option value="">Choose judgment</option><option value="clear_pose_failure">Clear pose estimation/tracking failure</option><option value="genuine_movement">Genuine movement</option><option value="camera_or_projection">Camera or projection change</option><option value="uncertain">Uncertain / mixed causes</option><option value="no_visible_issue">No visible issue</option><option value="insufficient_video">Insufficient video / bad alignment</option></select></p>
<input id="failure_type" placeholder="If clear failure: jitter, wrong person, missing joint, jump, ..." size="65">
<textarea id="notes" rows="3" placeholder="Evidence and notes; avoid diagnosing hidden joints from skeleton alone"></textarea>
<button onclick="step(-1)">Previous</button><button onclick="step(1)">Save and next</button><button onclick="download()">Download ratings.csv</button>
<p>Ratings save in this browser as you navigate. Download the CSV before moving or deleting the folder. No video or ratings are transmitted.</p>
<script>
const ids=IDS;let index=0;const storage='STORAGE';let ratings={};try{ratings=JSON.parse(localStorage.getItem(storage)||'{}')}catch(e){}
function save(){if(!ids.length)return;ratings[ids[index]]={review_id:ids[index],judgment:document.getElementById('judgment').value,failure_type:document.getElementById('failure_type').value,notes:document.getElementById('notes').value};try{localStorage.setItem(storage,JSON.stringify(ratings))}catch(e){}}
function show(){if(!ids.length){document.getElementById('status').textContent='No rendered clips';return;}const id=ids[index],r=ratings[id]||{};document.getElementById('status').textContent=`${index+1} / ${ids.length}: ${id}`;document.getElementById('video').src='clips/'+id+'.mp4';document.getElementById('native').src='clips/'+id+'_native.mp4';for(const k of ['judgment','failure_type','notes'])document.getElementById(k).value=r[k]||'';}
function step(d){save();index=Math.max(0,Math.min(ids.length-1,index+d));show();}
function quote(v){return '"'+String(v||'').replaceAll('"','""')+'"'}
function download(){save();const keys=['review_id','judgment','failure_type','notes'];const rows=[keys.join(',')];for(const id of ids){const r=ratings[id]||{review_id:id};rows.push(keys.map(k=>quote(r[k])).join(','));}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{type:'text/csv'}));a.download='ratings.csv';a.click();URL.revokeObjectURL(a.href);}
show();
</script>'''
    runhash=hashlib.sha256((out/'review_key.csv').read_bytes()).hexdigest()[:16]
    (out/'review.html').write_text(html.replace('IDS',json.dumps(ids)).replace('STORAGE','pose_review_'+runhash))

def summarize(args):
    out=Path(args.output_root);key=pd.read_csv(out/'review_key.csv');ratings=pd.read_csv(args.ratings).fillna('')
    if ratings.review_id.duplicated().any():raise ValueError('Duplicate review IDs in ratings')
    if not set(ratings.review_id).issubset(set(key.review_id)):raise ValueError('Unknown review IDs')
    bad=set(ratings.judgment)-set(JUDGMENTS)-{''}
    if bad:raise ValueError('Unknown judgments: '+str(bad))
    merged=key.merge(ratings,on='review_id',how='left',validate='one_to_one').fillna('')
    merged.to_csv(out/'reviewed_items.csv',index=False)
    groups=[]
    for (dataset,behavior,group),z in merged.groupby(['dataset','behavior','sample_group']):
        reviewed=z[z.judgment!=''];evaluable=reviewed[reviewed.judgment!='insufficient_video']
        groups.append({'dataset':dataset,'behavior':behavior,'sample_group':group,'n_selected':len(z),'n_reviewed':len(reviewed),'n_evaluable':len(evaluable),**{j:int((reviewed.judgment==j).sum()) for j in JUDGMENTS},'clear_failure_fraction_reviewed_evaluable':float((evaluable.judgment=='clear_pose_failure').mean()) if len(evaluable) else np.nan})
    pd.DataFrame(groups).to_csv(out/'review_summary.csv',index=False)
    print(pd.DataFrame(groups).to_string(index=False))
    print('Fractions describe the sampled strata, not overall failure prevalence. One event per sampled bout; bouts are not independent across participant/session.')

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--aqsm-pickle');a.add_argument('--sam-features');a.add_argument('--sam-tracks-root');a.add_argument('--output-root',required=True);a.add_argument('--flag-kind',choices=KINDS,default='jump_and_return');a.add_argument('--threshold',type=float,default=.5);a.add_argument('--per-stratum',type=int,default=5);a.add_argument('--context-sec',type=float,default=1.5);a.add_argument('--seed',type=int,default=20261005);a.add_argument('--strict-controls',action='store_true');a.set_defaults(func=prepare)
    a=sub.add_parser('render');a.add_argument('--output-root',required=True);a.add_argument('--video-root',required=True);a.add_argument('--video-map');a.add_argument('--slowdown',type=float,default=1.);a.add_argument('--max-panel-width',type=int,default=640);a.set_defaults(func=render)
    a=sub.add_parser('summarize');a.add_argument('--output-root',required=True);a.add_argument('--ratings',required=True);a.set_defaults(func=summarize)
    args=parser.parse_args()
    if getattr(args,'per_stratum',1)<1 or getattr(args,'context_sec',1)<=0 or getattr(args,'slowdown',1)<=0:parser.error('Counts, context and slowdown must be positive')
    args.func(args)

if __name__=='__main__':main()
