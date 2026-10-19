import json
import hashlib
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
from skeleton_lab.render import Renderer
from skeleton_lab.simulate import simulate
from skeleton_lab.spec import ScenarioSpec
from prism.optical_flow import OpticalFlowSettings, analyze_optical_flow_arrays
from prism.topology import TopologySettings, analyze_flow_states, framewise_rms_extent
from threadpoolctl import threadpool_limits
from .config import SPEC_DIR, FPS, DURATION, WARMUP, N_WINDOWS, DRIVES, METRICS

def payload(row):
    p = json.loads((SPEC_DIR / (row['behavior'] + '.json')).read_text())
    seed = int(row['seed']); rng = np.random.default_rng(seed)
    duration = DURATION + WARMUP
    p.update(name=row['condition_id'], duration_sec=duration, fps=FPS, seed=seed)
    for key in ['sway', 'micro_motion', 'fidget']: p['nuisance'][key]['enabled'] = False
    p['camera'].update(azimuth_deg=45., px_per_m=210., perspective=0.)
    p['camera']['shake']['sigma_px'] = 0.
    for key, value in [('pan_dx', 0.), ('pan_dy', 0.), ('zoom', 1.)]:
        p['camera'][key] = {'type': 'const', 'value': value}
    p['render'].update(width=640, height=360, overlay_truth=False)
    p['measurement']['jitter']['sigma_px'] = {'type': 'const', 'value': 0.}
    p['measurement']['dropout']['p_enter'] = 0.
    p['measurement']['swap']['p_swap'] = 0.
    selected = next(d for d in p['drives'] if d['id'] == DRIVES[row['behavior']])
    amp = float(selected['amplitude']['tracks'][1]['value'])
    for drive in p['drives']:
        for field in ['amplitude', 'sigma_r', 'sigma_phi']:
            drive[field] = {'type': 'const', 'value': 0.}
        drive['coupling'] = []
    selected['amplitude'] = {'type': 'const', 'value': amp * row['amplitude_scale']}
    selected['waveform'] = {'shape': 'sine', 'skew': 0., 'contact': 1.}
    selected['onset_rate'] = 8.
    selected['phi0_deg'] = float(rng.uniform(0, 360))
    times = np.linspace(0, duration, int(round(duration * FPS * 8)) + 1)
    noise = gaussian_filter1d(rng.normal(size=len(times)), .25 * FPS * 8, mode='reflect')
    core = times >= WARMUP
    noise -= trapezoid(noise[core], times[core]) / (times[-1] - times[core][0])
    noise /= max(float(abs(noise).max()), 1e-12)
    values = row['frequency_input'] * (1 + row['disruption'] * noise)
    selected['frequency'] = {'type': 'keyframes', 'points': np.c_[times, values].tolist(),
                             'interp': 'linear', 'time_unit': 'sec'}
    return p

def render(row):
    cv2.setNumThreads(1)
    result = simulate(ScenarioSpec.model_validate(payload(row)))
    skeleton = Renderer(result, variant='skeleton'); capsule = Renderer(result, variant='simple')
    frames, masks = [], []
    for i in range(int(round(WARMUP * FPS)), result.n_frames):
        frame = skeleton.frame(i); silhouette = capsule.frame(i)
        mask = np.max(cv2.absdiff(silhouette, capsule.background), axis=2) > 1
        frames.append(cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA))
        masks.append(cv2.resize(mask.astype(np.uint8), (256, 144), interpolation=cv2.INTER_NEAREST) > 0)
    return np.asarray(frames), np.asarray(masks, dtype=bool)

@threadpool_limits.wrap(limits=1)
def measure(frames, masks, *, camera='none', profile_power=2):
    if profile_power not in (1, 2):
        raise ValueError('profile_power must be 1 or 2')
    cv2.setNumThreads(1)
    t0 = time.perf_counter()
    flow = analyze_optical_flow_arrays(frames, masks, np.arange(len(frames)) / FPS,
        settings=OpticalFlowSettings(camera_correction=camera, torso_correction=True,
             minimum_valid_pixels=25, torso_minimum_pixels=12))
    t1 = time.perf_counter()
    extent = framewise_rms_extent(masks)['extent_frame_rms']
    t2 = time.perf_counter()
    state = flow['flow_states']
    temporal = analyze_flow_states(state['total'], state['confidence'], state['timestamps_sec'],
        pair_indices=state['pair_indices'], settings=TopologySettings(d=23, n_points=N_WINDOWS,
            min_cycles=0, min_period_sec=.2, max_period_sec=5.,
            flow_lag_profile_power=profile_power))
    t3 = time.perf_counter()
    summary = temporal['summary']
    return dict(intensity=flow['measurements'].get('optical_flow_pixel_p75_torso_corrected_body_heights_per_sec_median'),
                extent=extent, frequency=summary['period']['frequency_hz'],
                ps=summary['normalized'].get('periodicity_score'),
                windows_retained=summary['embedding']['n_windows_retained'],
                reliable=summary['period']['reliable'],
                flow_valid_pairs=len(state['timestamps_sec']),
                flow_sec=t1-t0, extent_sec=t2-t1, temporal_sec=t3-t2, total_sec=t3-t0,
                state_bytes=sum(state[k].nbytes for k in ['total', 'confidence', 'timestamps_sec', 'pair_indices']),
                frame_bytes=frames.nbytes, mask_bytes=masks.nbytes)
