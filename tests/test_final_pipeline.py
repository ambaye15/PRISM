import json
import numpy as np
import pandas as pd
import pytest
from prism.analysis.real_video import add_interbout_intervals, apply_qc, matched_pairs
from prism.topology import TopologySettings, analyze_flow_states
from prism.validation.pose_baseline.measurements import measure_pose_bout


def test_timing_preserves_all_annotations_and_never_crosses_files():
    data=pd.DataFrame({'episode_id':['c','a','b','d'], 'session_id':['one','one','one','two'],
                       'start_sec':[10,0,4,20],'duration_sec':[2,5,2,1]})
    result=add_interbout_intervals(data)
    assert result.loc[1,'interbout_status']=='first_bout'
    assert result.loc[2,'interbout_status']=='overlap'
    assert np.isnan(result.loc[2,'prism_interbout_interval_sec'])
    assert result.loc[0,'prism_interbout_interval_sec']==4
    assert result.loc[3,'interbout_status']=='first_bout'


def test_frequency_is_retained_when_windows_cannot_fit():
    n=20;t=np.arange(n)/10
    states=np.zeros((n,2,2,2));states[...,0]=np.sin(2*np.pi*t)[:,None,None]
    states[...,1]=np.cos(2*np.pi*t)[:,None,None]
    result=analyze_flow_states(states,np.ones((n,2,2)),t,
                              settings=TopologySettings(period_sec=5,n_points=30))
    assert result['summary']['period']['frequency_hz']==.2
    assert result['summary']['normalized']=={}
    assert result['summary']['issues'][0]['stage']=='periodicity'


def test_qc_does_not_treat_string_false_as_true():
    data=pd.DataFrame({'prism_frequency_hz':[1,1], 'fps':[10,10],
                      'topo_flow_estimated_cycles':[5,5],'topo_flow_valid_mask_fraction':[1,1],
                      'topo_flow_period_reliable':['False','True']})
    assert apply_qc(data).frequency_eligible.tolist()==[False,True]


def test_pose_reports_actual_used_pairs_and_uses_frozen_height():
    xy=np.zeros((4,17,2));xy[:,:,0]=np.arange(4)[:,None]*2
    xy[:,:,1]=np.linspace(10,90,17)
    confidence=np.ones((4,17));confidence[2]=0
    result=measure_pose_bout(xy,confidence,np.ones((4,100,40),bool),10,
                            maximum_interpolation_gap=0,pair_heights=np.full(3,200),return_pair_indices=True)
    assert result['total_pair_indices']==[0]
    assert result['pose_total_motion_p75_mask_heights_per_sec']==pytest.approx(.1)



def test_matches_require_same_recording_and_do_not_use_ineligible_frequency():
    data=pd.DataFrame({'episode_id':['a','b','c'], 'participant_id':['P001']*3,
                      'session_id':['one','one','two'],'behavior':['rock']*3,
                      'prism_intensity':[1,1,1], 'prism_extent':[1,3,5],
                      'prism_frequency_hz':[1,2,2],'prism_periodicity':[.1,.5,.9],
                      'duration_sec':[1,1,1],'prism_interbout_interval_sec':[1,2,3],
                      'frequency_eligible':[False,True,True]})
    pairs,selected=matched_pairs(data)
    assert not pairs.anchor.eq('frequency').any()
    assert set(pairs.episode_b)=={'b'}


def test_mask_baseline_failure_does_not_discard_production_coordinates(monkeypatch):
    from prism.validation.skeleton_lab import measurements as m
    monkeypatch.setattr(m,'framewise_rms_extent',lambda masks:{'extent_frame_rms':.4})
    monkeypatch.setattr(m,'analyze_flow_states',lambda *a,**kw:{'summary':{'period':{'frequency_hz':1.,'period_sec':1.},'normalized':{'periodicity_score':.5}}})
    def fail(*a,**kw):raise ValueError('no mask period')
    monkeypatch.setattr(m,'_topology_summary',fail)
    cached={'flow_states':{'total':None,'confidence':None,'timestamps_sec':None,'pair_indices':None}}
    result=m.topology_measurements(np.zeros((10,2,2,3)),np.ones((10,2,2)),10,flow_output=cached)
    assert result['prism_extent']==.4
    assert result['prism_frequency_hz']==1
    assert result['prism_periodicity']==.5
    assert not result['prism_frequency_welch_hz_available']
