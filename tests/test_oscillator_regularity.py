import numpy as np
from prism.validation.skeleton_lab.oscillator_regularity import frequency_track, scenario


def test_frequency_modulation_is_positive_matched_and_mean_preserving():
    clean=frequency_track(14,20,1,0,0,.25,2)
    low=frequency_track(14,20,1,.25,0,.25,2)
    high=frequency_track(14,20,1,.75,0,.25,2)
    p=np.array(high['points']);q=np.array(low['points']);base=np.array(clean['points'])
    assert np.all(p[:,1]>0)
    np.testing.assert_allclose((q[:,1]-1)*3,p[:,1]-1,atol=1e-12)
    core=p[:,0]>=2
    assert abs(np.trapz(p[core,1],p[core,0])/12-1)<1e-12
    assert np.all(base[:,1]==1)


def test_quasiperiodic_bindings_have_no_duplicate_rest_offset():
    p,primary=scenario('examples/specs','body_rocking',12,20,1,.5,0,'quasiperiodic',.25,2,45)
    extra=[b for b in p['bindings'] if b['drive']=='secondary_rhythm']
    original=[b for b in p['bindings'] if b['drive']==primary]
    assert len(extra)==len(original)>0
    assert all(b['rest_offset']['value']==0 for b in extra)
    assert all(d['sigma_phi']['value']==0 for d in p['drives'])
