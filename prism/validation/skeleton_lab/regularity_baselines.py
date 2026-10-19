import numpy as np
from scipy.signal import periodogram
from scipy.spatial.distance import pdist
from prism.topology.flow import _longest_consecutive_run, _robust_clip, _features


def prepare_baseline_states(states, confidence, timestamps, pair_indices, clip_mad=6.):

    keep=_longest_consecutive_run(pair_indices)
    time=np.asarray(timestamps,dtype=float)[keep]
    clipped,_=_robust_clip(np.asarray(states)[keep],np.asarray(confidence)[keep],clip_mad)
    features=_features(clipped,np.asarray(confidence)[keep])
    if not np.isfinite(features).all() or not np.isfinite(time).all():
        raise ValueError('Baseline features and timestamps must be finite')
    if len(time)<8 or np.any(np.diff(time)<=0):raise ValueError('Insufficient ordered flow states')
    dt=float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time),dt,rtol=1e-4,atol=1e-8):
        raise ValueError('Spectral baseline requires uniformly sampled consecutive flow states')

    grid,power=periodogram(features,fs=1/dt,axis=0,window='hann',detrend='constant',scaling='spectrum')
    energy=power.sum(axis=1)
    distances=pdist(features)
    scale=float(np.median(distances))
    return {'features':features,'timestamps':time,'frequency_grid':grid,'energy':energy,
            'distance_scale':scale,'duration_sec':len(time)*dt,'n_states':len(time)}


def scores(prepared, period_sec, fractional_bandwidth=.1, harmonics=2):

    if not np.isfinite(period_sec) or period_sec<=0:raise ValueError('Period must be positive')
    if fractional_bandwidth<=0 or harmonics<1:raise ValueError('Invalid spectral settings')
    time=prepared['timestamps'];features=prepared['features']
    grid=prepared['frequency_grid'];energy=prepared['energy'];frequency=1/period_sec
    positive=grid>0;total=float(energy[positive].sum())
    resolution=1/prepared['duration_sec']
    width=max(fractional_bandwidth*frequency,resolution)
    fundamental=positive & (np.abs(grid-frequency)<=width+1e-12)
    combined=fundamental.copy()
    for harmonic in range(2,harmonics+1):
        if harmonic*frequency<=grid[-1]:
            combined |= positive & (np.abs(grid-harmonic*frequency)<=width+1e-12)
    available=bool(total>1e-20 and frequency<=grid[-1])
    spectral=float(energy[fundamental].sum()/total) if available else np.nan
    harmonic=float(energy[combined].sum()/total) if available else np.nan
    targets=time+period_sec
    keep=np.flatnonzero(targets<=time[-1]+1e-10)
    return_error=np.nan;consistency=np.nan
    if len(keep) and prepared['distance_scale']>1e-12:
        targets=np.minimum(targets[keep],time[-1])
        right=np.clip(np.searchsorted(time,targets,side='right'),1,len(time)-1)
        left=right-1
        weight=(targets-time[left])/(time[right]-time[left])
        interpolated=(1-weight[:,None])*features[left]+weight[:,None]*features[right]
        return_error=float(np.median(np.linalg.norm(interpolated-features[keep],axis=1))/prepared['distance_scale'])
        consistency=float(np.exp(-return_error))
    return {'flow_spectral_concentration':spectral,'flow_harmonic_concentration':harmonic,
            'flow_cycle_return_consistency':consistency,'flow_cycle_return_relative_error':return_error,
            'baseline_return_pairs':len(keep),'baseline_n_states':prepared['n_states'],
            'spectral_half_bandwidth_hz':width,'spectral_harmonics':harmonics}
