import argparse
import json
import pickle
import warnings
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

COCO_FROM_MP = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]
BONES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
PARTS = {'wrists':[9,10], 'elbows':[7,8], 'torso':[5,6,11,12], 'legs':[13,14,15,16], 'head':[0,1,2,3,4]}
LABELS = {0:'none',1:'rock',2:'flap',3:'flap_rock'}

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name) in {('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','scalar'),('numpy._core.multiarray','scalar')}:
            return super().find_class(module,name)
        raise pickle.UnpicklingError(f'Unsupported pickle class: {module}.{name}')

def load_pickle(path):
    with open(path,'rb') as f:return RestrictedUnpickler(f).load()

def stat(x,quantile=.5):
    a=np.asarray(x);a=a[np.isfinite(a)]
    return float(np.median(a) if quantile==.5 else np.quantile(a,quantile)) if a.size else np.nan

def fraction(x):return float(np.mean(x)) if np.size(x) else np.nan

def runs(mask):
    e=np.diff(np.r_[False,mask,False].astype(int))
    return zip(np.flatnonzero(e==1),np.flatnonzero(e==-1))

def pose_scale(p,c):
    valid=np.isfinite(p).all(-1)&np.isfinite(c)&(c>=.5)&~np.all(p==0,-1)
    y=np.where(valid,p[:,:,1],np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        spans=np.nanmax(y,axis=1)-np.nanmin(y,axis=1)
    return stat(spans[(valid.sum(1)>=8)&(spans>1e-8)])

def preprocess(p,valid,fps,gap=0,sigma=0):
    x=p.copy();x[~valid]=np.nan
    for j in range(p.shape[1]):
        good=np.flatnonzero(valid[:,j])
        for a,b in zip(good[:-1],good[1:]):
            if 0<b-a-1<=int(np.floor(gap*fps+1e-9)):
                x[a:b+1,j]=np.linspace(x[a,j],x[b,j],b-a+1)
        if sigma:
            for a,b in runs(np.isfinite(x[:,j]).all(-1)):
                x[a:b,j]=gaussian_filter1d(x[a:b,j],sigma*fps,axis=0,mode='nearest')
    return x

def motion(p,valid,height,fps,exclude=None):
    pair=valid[:-1]&valid[1:]
    if exclude is not None:pair &= ~exclude
    velocity=np.diff(p,axis=0)*fps/height
    speed=np.linalg.norm(velocity,axis=-1)
    count=pair.sum(1);eligible=(count>=8)&np.isfinite(height)
    total=np.sum(np.where(pair,speed,0.),axis=1)/np.maximum(count,1)
    torso=PARTS['torso'];tv=np.where(pair[:,torso,None],velocity[:,torso],np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        ref=np.nanmedian(tv,axis=1)
    aspeed=np.linalg.norm(velocity-ref[:,None,:],axis=-1)
    articulated=np.sum(np.where(pair,aspeed,0.),axis=1)/np.maximum(count,1)
    arteligible=eligible&(pair[:,torso].sum(1)>=2)
    return {'intensity_total':stat(total[eligible]),'intensity_articulated':stat(articulated[arteligible]),'motion_pair_fraction':fraction(eligible)}

def audit(p,c,height,fps,threshold,meta):
    finite=np.isfinite(p).all(-1)&np.isfinite(c)
    zero=np.all(p==0,-1)
    valid=finite&~zero&(c>=threshold)
    pair=valid[:-1]&valid[1:]; triple=pair[:-1]&pair[1:]
    d=np.diff(p,axis=0)/height; dis=np.linalg.norm(d,axis=-1);speed=dis*fps
    acceleration=np.linalg.norm(np.diff(d,axis=0),axis=-1)*fps**2
    out={**meta,'threshold':threshold,'fps':fps,'n_frames':len(p),'scale':height,
         'nonfinite_fraction':fraction(~finite),'zero_coded_fraction':fraction(zero),
         'low_score_fraction':fraction(np.isfinite(c)&(c<threshold)),
         'coverage':fraction(valid),'pair_coverage':fraction(pair),'triplet_coverage':fraction(triple),
         'n_observed_joint_pairs':int(pair.sum()),'n_observed_joint_triplets':int(triple.sum()),'n_kinematic_joint_pairs':int(pair.sum()) if np.isfinite(height) else 0,'n_kinematic_joint_triplets':int(triple.sum()) if np.isfinite(height) else 0,
         'speed_p95':stat(speed[pair],.95),'acceleration_median':stat(acceleration[triple]),
         'acceleration_p95':stat(acceleration[triple],.95),'max_displacement':stat(dis[pair],1)}
    for x in [.15,.25,.5]:out[f'displacement_gt_{x:g}']=fraction(dis[pair]>x)
    for x in [2,5,10]:out[f'speed_gt_{x:g}']=fraction(speed[pair]>x)
    for x in [20,50,100]:out[f'acceleration_gt_{x:g}']=fraction(acceleration[triple]>x)
    gaps=[(b-a)/fps for j in range(17) for a,b in runs(~valid[:,j])]
    out['longest_gap_sec']=max(gaps,default=0.)
    out['score_drop_fraction']=fraction((c[:-1]-c[1:])[np.isfinite(c[:-1])&np.isfinite(c[1:])]>.4)
    for part,js in PARTS.items():
        out[part+'_frame_coverage']=fraction(valid[:,js].all(1))
        out[part+'_pair_coverage']=fraction(pair[:,js].all(1))

    spike=triple&(dis[:-1]>.15)&(dis[1:]>.15)&(np.linalg.norm((p[2:]-p[:-2])/height,axis=-1)<.05)
    out['jump_return_fraction']=fraction(spike[triple])
    robustflags=[]
    for j in range(17):
        a=acceleration[triple[:,j],j];median=stat(a);mad=1.4826*stat(np.abs(a-median))
        if len(a)>=8 and np.isfinite(mad):robustflags.extend(a>median+6*max(mad,1e-8))
    out['acceleration_mad6_fraction']=fraction(robustflags)
    bonecv=[];bonechanges=[];bonecollapse=[];bonestretch=[]
    for a,b in BONES:
        keep=valid[:,a]&valid[:,b];length=np.linalg.norm(p[:,a]-p[:,b],axis=1)/height
        good=length[keep];m=stat(good)
        if len(good)>=5 and np.mean(good)>1e-8:bonecv.append(np.std(good,ddof=1)/np.mean(good))
        if np.isfinite(m) and m>1e-8:
            bonecollapse.extend(good<.5*m);bonestretch.extend(good>1.5*m)
        bp=keep[:-1]&keep[1:]&(length[:-1]>1e-8)
        bonechanges.extend((np.abs(np.diff(length))[bp]/length[:-1][bp])>.25)
    out.update(bone_length_cv=stat(bonecv),bone_change_gt_25pct=fraction(bonechanges),bone_collapse_fraction=fraction(bonecollapse),bone_stretch_fraction=fraction(bonestretch))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        ref=np.nanmedian(np.where(pair[:,:,None],d,np.nan),axis=1)
    rn=np.linalg.norm(ref,axis=1)
    cosine=np.sum(d*ref[:,None,:],axis=-1)/np.maximum(np.linalg.norm(d,axis=-1)*rn[:,None],1e-12)
    align=np.sum(pair&(cosine>np.cos(np.pi/6)),axis=1)/np.maximum(pair.sum(1),1)
    out['coherent_global_jump_fraction']=fraction(((rn>.25)&(align>=.75))[pair.sum(1)>=8])
    swaps=[]
    for a,b in [(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]:
        keep=pair[:,a]&pair[:,b]
        direct=(np.linalg.norm(p[1:,a]-p[:-1,a],axis=1)+np.linalg.norm(p[1:,b]-p[:-1,b],axis=1))/height
        crossed=(np.linalg.norm(p[1:,b]-p[:-1,a],axis=1)+np.linalg.norm(p[1:,a]-p[:-1,b],axis=1))/height
        swaps.extend((crossed[keep]<.5*direct[keep])&(direct[keep]>.3))
    out['left_right_swap_candidate_fraction']=fraction(swaps)
    base=motion(p,valid,height,fps);out.update(base)
    if not np.isfinite(height):
        for key in list(out):
            if key.startswith(('displacement_gt','speed_gt','acceleration_gt')) or key in ['speed_p95','acceleration_median','acceleration_p95','max_displacement','jump_return_fraction','acceleration_mad6_fraction','bone_length_cv','bone_change_gt_25pct','bone_collapse_fraction','bone_stretch_fraction','coherent_global_jump_fraction','left_right_swap_candidate_fraction','intensity_total','intensity_articulated']:
                out[key]=np.nan
        out['motion_pair_fraction']=0.
    flagged=pair&(dis>.25)
    for t,j in np.argwhere(spike):flagged[t,j]=True;flagged[t+1,j]=True
    filtered=motion(p,valid,height,fps,flagged)
    for k in ['intensity_total','intensity_articulated']:
        out[k+'_after_flag_exclusion']=filtered[k]
        out[k+'_change_after_flag_exclusion']=(filtered[k]-base[k])/base[k] if base[k]>1e-8 else np.nan
    out['pair_fraction_after_flag_exclusion']=filtered['motion_pair_fraction']
    events=[]

    for t,j in np.argwhere(pair&(dis>.25)):
        events.append({**meta,'threshold':threshold,'frame':int(t+1),'time_sec':(t+1)/fps,'joint':int(j),'kind':'large_displacement','value':float(dis[t,j])})
    for t,j in np.argwhere(spike):
        events.append({**meta,'threshold':threshold,'frame':int(t+1),'time_sec':(t+1)/fps,'joint':int(j),'kind':'jump_and_return','value':float(max(dis[t,j],dis[t+1,j]))})
    return out,events

def aqsm_records(path,aspect):
    for entry in load_pickle(path):
        arr=np.asarray(entry['keypoints'],float)
        if arr.ndim!=3 or arr.shape[1]!=33:raise ValueError('AQSM requires frames x 33 x (x,y,visibility)')
        p=arr[:,COCO_FROM_MP,:2].copy();p[:,:,0]*=aspect;c=arr[:,COCO_FROM_MP,2]
        labels=np.asarray(entry['annotations']).reshape(-1)
        if len(labels)!=len(p):raise ValueError('AQSM annotations and poses differ in length')
        name=str(entry['name']);fps=float(entry['fps']);height=pose_scale(p,c)
        for a,b in runs(np.ones(len(labels),bool)):

            edges=np.r_[a,np.flatnonzero(np.diff(labels)!=0)+1,b]
            for lo,hi in zip(edges[:-1],edges[1:]):
                if hi-lo<1:continue
                label=labels[lo];meta={'dataset':'AQSM_MediaPipe','recording':name,'unit_id':f'{name}__f{lo}-{hi}','behavior':LABELS.get(int(label),str(label)),'start_frame':int(lo),'scale_source':'fixed_recording_pose_vertical_span','coordinates':'normalized_image_xy_aspect_'+str(aspect),**{f'native33_coverage_{t:g}':fraction(np.isfinite(arr[lo:hi,:,:2]).all(-1)&(arr[lo:hi,:,2]>=t)) for t in [.3,.5,.7]}}
                yield p[lo:hi],c[lo:hi],height,fps,meta

def sam_records(features,root):
    for row in pd.read_csv(features).to_dict('records'):
        name=str(row['episode_id']);path=Path(root)/(name+'.npz')
        if not path.exists():raise FileNotFoundError(path)
        with np.load(path,allow_pickle=False) as f:p=f['xy'].copy();c=f['confidence'].copy()
        end=row.get('core_end_frame',row.get('core_end_frame_exclusive'))
        lo=int(row['core_start_frame'])-int(row['analysis_start_frame']);hi=int(end)-int(row['analysis_start_frame'])
        if not 0<=lo<hi<=len(p):raise ValueError(f'Core bounds outside saved pose track: {name}')
        height=float(row['pose_median_mask_height_px']);scale_source='saved_median_mask_height'

        if not np.isfinite(height) or height<=0:height=pose_scale(p,c);scale_source='fallback_pose_vertical_span'
        yield p[lo:hi],c[lo:hi],height,float(row['fps']),{'dataset':'AQSM_SAM_crop','recording':row['session_id'],'unit_id':name,'behavior':row['behavior'],'start_frame':int(row['core_start_frame']),'scale_source':scale_source,'coordinates':'pixel_xy'}

def external_records(path,processing_per_label,seed):
    data=load_pickle(path)
    labels={}
    for i,entry in enumerate(data['annotations']):labels.setdefault(entry['action_name'],[]).append(i)
    rng=np.random.default_rng(seed);selected=set()
    for indices in labels.values():selected.update(rng.choice(indices,min(len(indices),processing_per_label),replace=False).tolist())
    for i,entry in enumerate(data['annotations']):
        p=np.asarray(entry['keypoint'],float);c=np.asarray(entry['keypoint_score'],float)

        height=pose_scale(p,c)
        yield p,c,height,float(entry['fps']),{'dataset':'ASDMotion','recording':str(entry['identifier']),'unit_id':f'{i}__{entry["identifier"]}','behavior':entry['action_name'],'start_frame':0,'scale_source':'fixed_clip_pose_vertical_span','coordinates':'pixel_xy','processing_selected':i in selected}

def summarize(df,out):
    metrics=[c for c in df.select_dtypes('number') if c not in ['start_frame','threshold','fps','n_frames','scale']]
    keys=['dataset','behavior','threshold'];rows=[]
    for key,g in df.groupby(keys):
        row=dict(zip(keys,key));row['n_units']=len(g)
        for c in metrics:
            row[c+'_median']=stat(g[c]);row[c+'_q25']=stat(g[c],.25);row[c+'_q75']=stat(g[c],.75);row[c+'_n']=int(g[c].notna().sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'summary_by_behavior.csv',index=False)
    overall=[]
    for key,g in df.groupby(['dataset','threshold']):
        row={'dataset':key[0],'threshold':key[1],'n_units':len(g)}
        for c in metrics:row[c+'_median']=stat(g[c]);row[c+'_n']=int(g[c].notna().sum())
        overall.append(row)
    pd.DataFrame(overall).to_csv(out/'summary_overall.csv',index=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aqsm-pickle');parser.add_argument('--aqsm-image-aspect',type=float,default=1.)
    parser.add_argument('--sam-features');parser.add_argument('--sam-tracks-root');parser.add_argument('--asdmotion-pickle')
    parser.add_argument('--output-root',required=True);parser.add_argument('--thresholds',type=float,nargs='+',default=[.3,.5,.7])
    parser.add_argument('--limit',type=int);parser.add_argument('--skip-processing',action='store_true');parser.add_argument('--processing-per-label',type=int,default=50);parser.add_argument('--seed',type=int,default=20261005);args=parser.parse_args();out=Path(args.output_root);out.mkdir(parents=True,exist_ok=True)
    generators=[]
    if args.aqsm_pickle:generators.append(aqsm_records(args.aqsm_pickle,args.aqsm_image_aspect))
    if args.sam_features:generators.append(sam_records(args.sam_features,args.sam_tracks_root))
    if args.asdmotion_pickle:generators.append(external_records(args.asdmotion_pickle,args.processing_per_label,args.seed))
    if not generators:parser.error('Provide at least one input dataset')
    rows=[];events=[];errors=[];processing=[]
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    for generator in generators:
        for number,(p,c,h,fps,meta) in enumerate(generator,1):
            if args.limit and number>args.limit:break
            if meta['behavior']=='none' and meta['dataset']=='AQSM_MediaPipe':continue
            try:
                if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid FPS')
                if not np.isfinite(h) or h<=0:errors.append({**meta,'error':'No usable height: coverage retained; kinematic metrics unavailable'});h=np.nan
                for threshold in args.thresholds:
                    row,ee=audit(p,c,h,fps,threshold,meta);rows.append(row)
                    if threshold==.5:events.extend(sorted(ee,key=lambda x:x['value'],reverse=True)[:8])

                if args.skip_processing or not np.isfinite(h) or not meta.get('processing_selected',True) or meta['behavior']=='none':
                    if number%500==0:print(meta['dataset'],number,flush=True)
                    continue
                for threshold in args.thresholds:
                    valid=np.isfinite(p).all(-1)&np.isfinite(c)&~np.all(p==0,-1)&(c>=threshold)
                    base=motion(p,valid,h,fps)
                    widths = [0,1/fps,2/fps] if meta['dataset']=='ASDMotion' else [0,.1,.2]
                    gaps = [0] if meta['dataset']=='ASDMotion' else [0,.2]
                    for gap,sigma in product(gaps,widths):
                        x=preprocess(p,valid,fps,gap,sigma);m=motion(x,np.isfinite(x).all(-1),h,fps)
                        processing.append({**meta,'threshold':threshold,'gap_sec':gap,'sigma_sec':sigma,'sigma_frames':sigma*fps,**m,'baseline_intensity_total':base['intensity_total'],'total_change_from_same_threshold':(m['intensity_total']-base['intensity_total'])/base['intensity_total'] if base['intensity_total']>1e-8 else np.nan})
            except (ValueError,IndexError) as e:errors.append({**meta,'error':str(e)})
            if number%500==0:print(meta['dataset'],number,flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'unit_quality.csv',index=False);summarize(df,out)
    pd.DataFrame(events).to_csv(out/'review_events.csv',index=False)
    pd.DataFrame(errors).to_csv(out/'unavailable_units.csv',index=False)
    pd.DataFrame(processing).to_csv(out/'processing_sensitivity.csv',index=False)
    settings={**vars(args),'n_audited_units':len(df[df.threshold==.5]),'n_scale_unavailable':len(errors),'interpretation':'Flags are review diagnostics, not verified errors. AQSM and SAM are the same cohort. Valid denominators accompany flags. No interpolation/smoothing before diagnostics. Raw coordinates include camera motion. Height normalization differs across sources. AQSM normalized x/y aspect ratio must be verified. COCO mapping on external data requires joint-layout verification. Core intervals only for SAM.'}
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    from prism.plotting.pose_audit import plot_audit
    plot_audit(df,out)
    print('Completed',len(df),'threshold-unit rows; unavailable scales',len(errors),flush=True)
if __name__=='__main__':main()
