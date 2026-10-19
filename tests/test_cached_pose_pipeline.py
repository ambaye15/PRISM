import json
import numpy as np
import pandas as pd
from prism.validation.pose_baseline.analysis import run_pose_baseline


def test_cached_pose_reaggregates_flow_without_loading_model_or_rgb(tmp_path):
    runs=tmp_path/'runs';output=tmp_path/'pose';tracks=output/'pose_tracks'
    tracks.mkdir(parents=True);rows=[]
    for i in range(4):
        name='bout_%d'%i;run=runs/name;run.mkdir(parents=True)
        np.savez(run/'sam2_masks_fixed_crop.npz',masks=np.ones((8,100,30),bool),
                 core_mask=np.ones(8,bool),mask_valid=np.ones(8,bool),timestamps_sec=np.arange(8)/10)
        (run/'subject_dino_summary.json').write_text(json.dumps({'source_cached_video':'/missing/video.mp4'}))
        xy=np.zeros((8,17,2));xy[:,:,0]=np.arange(8)[:,None]*(i+1)
        xy[:,:,1]=np.linspace(10,90,17)
        confidence=np.ones((8,17));confidence[-1]=0
        np.savez(tracks/(name+'.npz'),xy=xy,confidence=confidence,mask_box_iou=np.ones(8))
        total=np.r_[np.full(6,(i+1)/20),1000]
        pd.DataFrame({'frame_index':np.arange(7),'flow_pair_valid':True,'is_core_pair':True,
                      'body_height_pixels':200.,'flow_pixel_p75_body_heights_per_sec':total,
                      'flow_pixel_p75_torso_corrected_body_heights_per_sec':total/2,
                      'flow_torso_translation_body_heights_per_sec':total/2}).to_csv(run/'optical_flow_frames.csv',index=False)
        rows.append({'episode_id':name,'participant_id':'P001','behavior':'rock'})
    features=tmp_path/'features.csv';pd.DataFrame(rows).to_csv(features,index=False)
    merged,agreement,_=run_pose_baseline(features,runs,output,maximum_interpolation_gap=0,
                                        bootstrap_replicates=10)
    assert merged.flow_total_shared_pose_pairs_n.eq(6).all()
    assert np.allclose(merged.flow_total_shared_pose_pairs,[.05,.1,.15,.2])
    assert merged.pose_cache_status.eq('reused').all()
    assert len(agreement)>0
