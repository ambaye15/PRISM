import argparse
import copy
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

PRESETS={'body_rocking':('body_rocking.json','rock'),
         'arm_flapping':('arm_flapping.json','flap'),
         'hand_flapping':('hand_flapping.json','flap')}


def frequency_track(duration, fps, frequency, level, seed, correlation_sec, warmup):

    if not 0 <= level < 1:raise ValueError('phase levels must satisfy 0 <= level < 1')


    times=np.linspace(0,duration,int(round(duration*fps*8))+1)
    noise=gaussian_filter1d(np.random.default_rng(seed).normal(size=len(times)),
                           max(correlation_sec*fps*8,1),mode='reflect')
    core=times>=warmup
    noise-=np.trapz(noise[core],times[core])/(times[-1]-times[core][0])
    noise/=max(float(np.abs(noise).max()),1e-12)
    values=frequency*(1+level*noise)
    return {'type':'keyframes','points':np.c_[times,values].tolist(),
            'interp':'linear','time_unit':'sec'}


def scenario(specs_root, behavior, duration, fps, frequency, level, seed,
             family, correlation_sec, warmup, azimuth):
    name,drive_id=PRESETS[behavior]
    p=json.loads((Path(specs_root)/name).read_text())
    p.update(duration_sec=duration+warmup,fps=fps,seed=seed)
    p['nuisance']['sway']['enabled']=False
    p['nuisance']['micro_motion']['enabled']=False
    p['nuisance']['fidget']['enabled']=False
    p['camera'].update(azimuth_deg=azimuth,px_per_m=210.,perspective=0.)
    p['camera']['shake']['sigma_px']=0.
    p['camera']['pan_dx']={'type':'const','value':0.}
    p['camera']['pan_dy']={'type':'const','value':0.}
    p['camera']['zoom']={'type':'const','value':1.}
    p['render'].update(width=640,height=360,overlay_truth=False)
    p['measurement']['jitter']['sigma_px']={'type':'const','value':0.}
    p['measurement']['dropout']['p_enter']=0.
    p['measurement']['swap']['p_swap']=0.
    selected=next(d for d in p['drives'] if d['id']==drive_id)
    amplitude=float(selected['amplitude']['tracks'][1]['value'])

    for drive in p['drives']:
        drive['amplitude']={'type':'const','value':0.}
        drive['sigma_phi']={'type':'const','value':0.}
        drive['sigma_r']={'type':'const','value':0.}
        drive['coupling']=[]
    selected['amplitude']={'type':'const','value':amplitude}
    selected['waveform']={'shape':'sine','skew':0.,'contact':1.}
    selected['onset_rate']=8.
    if family=='phase_disruption':
        selected['frequency']=frequency_track(duration+warmup,fps,frequency,level,
                                              seed,correlation_sec,warmup)
    else:
        selected['frequency']={'type':'const','value':frequency}

        selected['amplitude']['value']=amplitude/np.sqrt(1+level**2)
        second=copy.deepcopy(selected);second['id']='secondary_rhythm'
        second['frequency']={'type':'const','value':frequency*np.sqrt(2)}
        second['amplitude']['value']*=level
        second['phi0_deg']=45.
        p['drives'].append(second)

        for binding in list(p['bindings']):
            if binding['drive']==drive_id:
                extra=copy.deepcopy(binding);extra['drive']='secondary_rhythm'
                extra['rest_offset']={'type':'const','value':0.}
                p['bindings'].append(extra)
    return p,drive_id


def write_video(frames,fps,path,caption):
    h,w=frames.shape[1:3]
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h+42))
    if not writer.isOpened():raise RuntimeError('MP4 writer could not open '+str(path))
    for i,frame in enumerate(frames):
        canvas=np.full((h+42,w,3),245,np.uint8);canvas[42:]=frame
        cv2.putText(canvas,caption,(7,17),cv2.FONT_HERSHEY_SIMPLEX,.38,(30,30,30),1,cv2.LINE_AA)
        cv2.putText(canvas,f't = {i/fps:.2f} s',(7,34),cv2.FONT_HERSHEY_SIMPLEX,.38,(30,30,30),1,cv2.LINE_AA)
        writer.write(canvas)
    writer.release()


def render_result(result,start,max_dimension):
    from skeleton_lab.render import Renderer

    flow_renderer=Renderer(result,variant='skeleton')
    mask_renderer=Renderer(result,variant='simple')
    frames=[];masks=[]
    for i in range(start,result.n_frames):
        frame=flow_renderer.frame(i);silhouette=mask_renderer.frame(i)
        mask=np.max(cv2.absdiff(silhouette,mask_renderer.background),axis=2)>1
        ratio=min(1.,max_dimension/max(frame.shape[:2]))
        size=(int(round(frame.shape[1]*ratio)),int(round(frame.shape[0]*ratio)))
        frames.append(cv2.resize(frame,size,interpolation=cv2.INTER_AREA))
        masks.append(cv2.resize(mask.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0)
    return np.array(frames),np.array(masks)


def generate(args):
    from skeleton_lab.spec import ScenarioSpec
    from skeleton_lab.simulate import simulate
    root=Path(args.output_root);root.mkdir(parents=True,exist_ok=True)
    bundles=root/'bundles';bundles.mkdir(exist_ok=True)
    videos=root/'videos';videos.mkdir(exist_ok=True)
    rows=[]
    families=['phase_disruption'] if args.no_quasiperiodic else ['phase_disruption','quasiperiodic']
    for behavior in args.behaviors:
      for family in families:
       for seed in args.seeds:
        for level in args.levels:
            cid=f'{behavior}__{family}__r{level:g}__seed{seed}'
            payload,drive_id=scenario(args.specs_root,behavior,args.duration_sec,args.fps,
                args.frequency_hz,level,seed,family,args.correlation_sec,args.warmup_sec,args.azimuth_deg)
            payload['name']=cid
            result=simulate(ScenarioSpec.model_validate(payload))
            start=int(round(args.warmup_sec*args.fps))
            frames,masks=render_result(result,start,args.max_dimension)
            drive=result.drives[drive_id]
            phase=np.asarray(drive.phi)[start:]
            instantaneous=np.asarray(drive.f_inst)[start:]
            joints=np.stack(list(result.joints_world.values()),axis=1)[start:]
            displacement=np.linalg.norm(np.diff(joints,axis=0),axis=-1)*args.fps
            phase_times=np.asarray(result.t)[start:]
            cycles=np.arange(np.ceil(phase[0]/(2*np.pi)),np.floor(phase[-1]/(2*np.pi))+1)*2*np.pi
            crossings=np.interp(cycles,phase,phase_times)
            intervals=np.diff(crossings)
            dof=np.asarray(result.dof_values[next(b['target'] for b in payload['bindings'] if b['drive']==drive_id)])[start:]
            path=bundles/(cid+'.npz')
            np.savez_compressed(path,frames=frames,masks=masks,fps=args.fps,
                                phase=phase,instantaneous_frequency_hz=instantaneous,
                                joints_world=joints,cycle_intervals_sec=intervals,primary_dof=dof)
            spec_path=bundles/(cid+'.spec.json');spec_path.write_text(json.dumps(payload,indent=2)+'\n')
            row=dict(condition_id=cid,bundle_path='bundles/'+path.name,behavior=behavior,
                family=family,seed=seed,irregularity=level,reference_frequency_hz=args.frequency_hz,
                mean_primary_frequency_hz=float((phase[-1]-phase[0])/(2*np.pi*(phase_times[-1]-phase_times[0]))),
                cycle_duration_cv=float(np.std(intervals)/np.mean(intervals)) if len(intervals)>1 else np.nan,
                primary_dof_peak_to_peak=float(np.ptp(dof)),primary_dof_rms=float(np.sqrt(np.mean((dof-dof.mean())**2))),
                world_joint_speed_median=float(np.median(displacement.mean(axis=1))),fps=args.fps,n_frames=len(frames))
            rows.append(row)
            if args.videos:
                write_video(frames,args.fps,videos/(cid+'.mp4'),f'{behavior.replace("_"," ")} | {family.replace("_"," ")} {level:g}')
            print('Generated',cid,flush=True)
    pd.DataFrame(rows).to_csv(root/'manifest.csv',index=False)
    (root/'generation_settings.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    if args.videos:

        for behavior in args.behaviors:
          for family in families:
            group=[r for r in rows if r['behavior']==behavior and r['family']==family and r['seed']==args.seeds[0]]
            sequences=[np.load(root/r['bundle_path'])['frames'] for r in group]
            panels=[]
            for i in range(min(map(len,sequences))):
                tiles=[]
                for r,seq in zip(group,sequences):
                    h,w=seq.shape[1:3];tile=np.full((h+28,w,3),245,np.uint8);tile[28:]=seq[i]
                    cv2.putText(tile,f'level {r["irregularity"]:g}',(7,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1,cv2.LINE_AA)
                    tiles.append(tile)
                panels.append(np.concatenate(tiles,axis=1))
            write_video(np.array(panels),args.fps,videos/(behavior+'__'+family+'__comparison.mp4'),
                        behavior.replace('_',' ')+' | '+family.replace('_',' '))
    return root/'manifest.csv'


def evaluate(args):
    from prism.optical_flow import OpticalFlowSettings,analyze_optical_flow_arrays
    from prism.topology import TopologySettings,analyze_flow_states
    root=Path(args.output_root);data=pd.read_csv(root/'manifest.csv');rows=[]
    for _,condition in data.iterrows():
        path=root/condition.bundle_path
        with np.load(path) as bundle:
            frames=bundle['frames'];masks=bundle['masks'];fps=float(bundle['fps'])
        timestamps=np.arange(len(frames))/fps
        flow=analyze_optical_flow_arrays(frames,masks,timestamps,settings=OpticalFlowSettings(
            camera_correction='none',torso_correction=True,minimum_valid_pixels=25,torso_minimum_pixels=12))
        states=flow['flow_states']
        for mode in ['reference','estimated']:
            row=condition.to_dict();row['period_mode']=mode
            row.update(prism_periodicity=np.nan,prism_frequency_hz=np.nan,spectral_concentration=np.nan)
            row['prism_intensity']=flow['measurements'].get('optical_flow_pixel_p75_torso_corrected_body_heights_per_sec_median')
            try:
                settings=TopologySettings(d=args.d,n_points=args.n_points,min_cycles=0,
                    min_period_sec=.4,max_period_sec=5.,
                    period_sec=1/condition.reference_frequency_hz if mode=='reference' else None)
                analysis=analyze_flow_states(states['total'],states['confidence'],states['timestamps_sec'],
                                            pair_indices=states['pair_indices'],settings=settings)
                summary=analysis['summary']
                row.update(prism_periodicity=summary['normalized'].get('periodicity_score'),
                           prism_frequency_hz=summary['period']['frequency_hz'],
                           spectral_concentration=summary['period'].get('spectral_concentration'),
                           analysis_error='; '.join(x['error'] for x in summary.get('issues',[])))
            except ValueError as error:row['analysis_error']=str(error)
            rows.append(row)
        print('Evaluated',condition.condition_id,flush=True)
        pd.DataFrame(rows).to_csv(root/'results.csv',index=False)
    data=pd.DataFrame(rows);summaries=[]
    for keys,g in data.groupby(['behavior','family','seed','period_mode']):
        ps=pd.to_numeric(g.prism_periodicity,errors='coerce');valid=ps.notna()
        rho=np.nan
        if valid.sum()>=3 and ps[valid].nunique()>1:
            rho=float(spearmanr(-g.loc[valid,'irregularity'],ps[valid]).statistic)
        summaries.append(dict(zip(['behavior','family','seed','period_mode'],keys),
                              n=len(g),n_available=int(valid.sum()),rho_ps_negative_level=rho))
    pd.DataFrame(summaries).to_csv(root/'regularity_summary.csv',index=False)
    from prism.plotting.oscillator_regularity import plot_results
    plot_results(data,root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',default='output/oscillator_regularity')
    p.add_argument('--specs-root',default='examples/specs')
    p.add_argument('--behaviors',nargs='+',choices=list(PRESETS),default=['body_rocking'])
    p.add_argument('--levels',type=float,nargs='+',default=[0.,.25,.5,.75])
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--fps',type=float,default=20.)
    p.add_argument('--duration-sec',type=float,default=12.)
    p.add_argument('--warmup-sec',type=float,default=2.)
    p.add_argument('--frequency-hz',type=float,default=1.)
    p.add_argument('--correlation-sec',type=float,default=.25)
    p.add_argument('--azimuth-deg',type=float,default=45.)
    p.add_argument('--max-dimension',type=int,default=256)
    p.add_argument('--d',type=int,default=23);p.add_argument('--n-points',type=int,default=300)
    p.add_argument('--videos',action='store_true');p.add_argument('--no-quasiperiodic',action='store_true')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--generate-only',action='store_true');mode.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args()
    if args.duration_sec<=0 or args.fps<=0 or args.frequency_hz<=0 or args.correlation_sec<=0:p.error('Duration, FPS, frequency and correlation time must be positive')
    if args.warmup_sec<0 or any(not 0<=r<1 for r in args.levels):p.error('Warmup must be nonnegative and levels must be in [0,1)')
    if len(set(args.levels))!=len(args.levels) or len(set(args.seeds))!=len(args.seeds):p.error('Levels and seeds must be unique')
    if not args.evaluate_only:generate(args)
    if not args.generate_only:evaluate(args)

if __name__=='__main__':main()
