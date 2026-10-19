import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

MEASURES = {'intensity': 'prism_intensity', 'extent': 'prism_extent',
            'frequency': 'prism_frequency_hz', 'periodicity': 'prism_periodicity',
            'duration': 'duration_sec', 'interbout': 'prism_interbout_interval_sec'}


def truthy(values):
    return values.astype(str).str.lower().isin({'true', '1', 'yes'})


def add_interbout_intervals(frame):

    out = frame.copy()
    needed = {'session_id', 'start_sec', 'duration_sec'}
    missing = needed - set(out)
    if missing:
        raise ValueError('Timing requires annotation metadata: ' + ', '.join(sorted(missing)))
    if out.get('episode_id', pd.Series(dtype=str)).duplicated().any():
        raise ValueError('Duplicate episode_id')
    out['start_sec'] = pd.to_numeric(out.start_sec, errors='raise')
    out['end_sec'] = (pd.to_numeric(out.end_sec, errors='raise') if 'end_sec' in out
                      else out.start_sec + pd.to_numeric(out.duration_sec, errors='raise'))
    if out[['start_sec', 'end_sec']].isna().any().any() or (out.end_sec < out.start_sec).any():
        raise ValueError('Missing or reversed annotation bounds')
    out['duration_sec'] = out.end_sec - out.start_sec

    group = 'source_video_path' if 'source_video_path' in out and out.source_video_path.notna().all() else 'session_id'
    if out[group].isna().any():
        raise ValueError('Missing recording identifier')
    sort_columns = [group, 'start_sec', 'end_sec']
    if 'episode_id' in out:
        sort_columns.append('episode_id')
    order = out.sort_values(sort_columns, kind='stable')
    previous = order.groupby(group, sort=False).end_sec.shift()
    gaps = order.start_sec - previous
    order['prism_interbout_signed_interval_sec'] = gaps
    order['interbout_overlap'] = gaps.lt(0)
    order['prism_interbout_interval_sec'] = gaps.where(gaps.ge(0))
    order['interbout_status'] = np.select([previous.isna(), gaps.lt(0)],
                                          ['first_bout', 'overlap'], default='available')
    return order.reindex(frame.index)


def apply_qc(frame, min_cycles=2.5, minimum_mask_valid_fraction=.8, nyquist_fraction=.8):
    out = frame.copy()
    frequency = pd.to_numeric(out.prism_frequency_hz, errors='coerce')
    cycles = pd.to_numeric(out.topo_flow_estimated_cycles, errors='coerce')
    coverage = pd.to_numeric(out.topo_flow_valid_mask_fraction, errors='coerce')
    fps = pd.to_numeric(out.fps, errors='coerce')
    reliable = truthy(out.topo_flow_period_reliable)
    checks = [('missing_or_nonpositive', frequency.notna() & frequency.gt(0)),
              ('unreliable_period', reliable), ('too_few_cycles', cycles.ge(min_cycles)),
              ('insufficient_masks', coverage.ge(minimum_mask_valid_fraction)),
              ('nyquist', fps.gt(0) & frequency.le(nyquist_fraction * fps / 2))]
    out['frequency_eligible'] = True
    out['frequency_exclusion_reason'] = 'eligible'
    for reason, passed in checks:
        first = out.frequency_eligible & ~passed
        out.loc[first, 'frequency_exclusion_reason'] = reason
        out['frequency_eligible'] &= passed
    return out


def participant_name(value):
    value = str(value)
    match = re.fullmatch(r'(?:P)?0*(\d+)', value, flags=re.I)
    return 'P%03d' % int(match[1]) if match else value


def measure_values(group, name, column):
    if name == 'frequency':
        group = group.loc[group.frequency_eligible]
    return pd.to_numeric(group[column], errors='coerce').dropna()


def summaries(frame, keys):
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        for name, column in MEASURES.items():
            x = measure_values(group, name, column)
            mean = x.mean()
            rows.append({**dict(zip(keys, key)), 'measure': name, 'n': len(x),
                         'n_bouts': len(group), 'median': x.median(),
                         'q25': x.quantile(.25), 'q75': x.quantile(.75),
                         'iqr': x.quantile(.75) - x.quantile(.25),
                         'cv': x.std(ddof=1) / mean if len(x) > 1 and mean > 0 else np.nan})
    return pd.DataFrame(rows)


def matched_pairs(frame, tolerance=.03, minimum_difference=.1, pairs_per_measure=3):
    work = frame.copy()
    for name, column in list(MEASURES.items())[:4]:
        value = pd.to_numeric(work[column], errors='coerce')
        if name == 'frequency':
            value = value.where(work.frequency_eligible)
        work[column + '_rank'] = value.groupby([work.participant_id, work.behavior]).rank(pct=True)
    rows = []
    recording='source_video_path' if 'source_video_path' in work and work.source_video_path.notna().all() else 'session_id'
    for key, g in work.groupby(['participant_id', 'behavior', recording]):
        g = g.sort_values('episode_id')
        for anchor, column in list(MEASURES.items())[:4]:
            ranked = g[column + '_rank'].to_numpy()
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if not np.isfinite(ranked[i] + ranked[j]) or abs(ranked[i] - ranked[j]) > tolerance:
                        continue
                    a, b = g.iloc[i], g.iloc[j]
                    differences = {name: abs(a[col + '_rank'] - b[col + '_rank'])
                                   for name, col in list(MEASURES.items())[:4] if name != anchor}
                    finite = {name: d for name, d in differences.items() if np.isfinite(d)}
                    if not finite:
                        continue
                    other = max(finite, key=finite.get)
                    row = {'participant_id': key[0], 'behavior': key[1], 'session_id': g.iloc[0].session_id, 'source_recording':key[2],
                           'anchor': anchor, 'episode_a': a.episode_id, 'episode_b': b.episode_id,
                           'anchor_rank_difference': abs(ranked[i] - ranked[j]),
                           'largest_other_difference': finite[other], 'different_measure': other,
                           'contrast': finite[other] >= minimum_difference}
                    for name, col in MEASURES.items():
                        row[name + '_a'] = a[col]; row[name + '_b'] = b[col]
                        if name in ['intensity', 'extent', 'frequency', 'periodicity']:
                            row[name + '_rank_a'] = a[col + '_rank'];row[name + '_rank_b'] = b[col + '_rank']
                    rows.append(row)
    columns = ['anchor', 'contrast', 'largest_other_difference', 'episode_a', 'episode_b']
    pairs = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    selected = (pairs.loc[pairs.contrast.astype(bool)].sort_values(
        ['anchor', 'largest_other_difference', 'episode_a', 'episode_b'],
        ascending=[True, False, True, True]).groupby('anchor').head(pairs_per_measure))
    return pairs, selected


def export_pair_videos(selected, frame, destination, video_root):
    destination = Path(destination);destination.mkdir(parents=True, exist_ok=True)
    sources = frame.set_index('episode_id')
    root = Path(video_root)
    for number, (_, row) in enumerate(selected.iterrows(), 1):
        for suffix in ['a', 'b']:
            bout = sources.loc[row['episode_' + suffix]]
            declared = Path(str(bout.source_video_path))
            hits = [declared] if declared.is_file() else [p for p in root.rglob('*')
                if p.is_file() and p.name == declared.name]
            if len(hits) != 1:
                raise ValueError('Expected one source video for ' + str(declared) + '; found ' + str(len(hits)))
            output = destination / ('pair_%02d_%s_%s.mp4' % (number, row.anchor, suffix.upper()))
            subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', str(bout.start_sec),
                            '-i', str(hits[0]), '-t', str(bout.duration_sec), '-an',
                            '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(output)], check=True)


def write_profile_table(profiles, path):
    lines=[r"\begin{tabular}{lcccccc}",r"\toprule",
           r"Behavior & Intensity & Extent & Frequency & Periodicity & Duration & Interval \\",r"\midrule"]
    for behavior,g in profiles.groupby('behavior'):
        cells=[]
        for name in MEASURES:
            row=g.loc[g.measure.eq(name)].iloc[0]
            cells.append(('%.3f [%.3f, %.3f]' % (row['median'],row.q25,row.q75)) if row.n else '--')
        label=str(behavior).replace('_',r'\_')
        lines.append(label+' & '+' & '.join(cells)+r' \\')
    lines += [r"\bottomrule",r"\end{tabular}"]
    Path(path).write_text('\n'.join(lines)+'\n')


def run_analysis(input_path, output_root, participants=('P001', 'P004'),
                 tolerance=.06, minimum_difference=.25, pairs_per_measure=3,
                 min_cycles=2.5, minimum_mask_valid_fraction=.8, nyquist_fraction=.8,
                 minimum_session_bouts=3, export_videos=False, video_root=None):
    data = pd.read_csv(input_path, dtype={'episode_id': str, 'participant_id': str, 'session_id': str})
    data.participant_id = data.participant_id.map(participant_name)
    data.behavior = data.behavior.astype(str).str.lower().str.replace('-', '_')
    data = apply_qc(add_interbout_intervals(data), min_cycles, minimum_mask_valid_fraction, nyquist_fraction)
    out = Path(output_root);out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / 'analysis_ready.csv', index=False)
    profiles = summaries(data, ['behavior'])
    profiles.to_csv(out / 'behavior_profiles.csv', index=False)
    write_profile_table(profiles,out/'behavior_profiles.tex')
    within = summaries(data, ['participant_id', 'behavior'])
    within.to_csv(out / 'within_participant.csv', index=False)
    within.loc[within.n >= minimum_session_bouts].groupby(['behavior','measure'])[['iqr','cv']].median().to_csv(out / 'within_participant_summary.csv')
    session_keys = ['participant_id', 'session_id', 'behavior']
    for c in ['study_id', 'session_date']:
        if c in data:
            session_keys.append(c)
    session_data = data.copy()
    split_recordings = {
        '004-01-17-08_0_study1',
        '004-01-17-08_2_study1'
    }
    same_session = (
        session_data.participant_id.eq('P004')
        & session_data.session_id.isin(split_recordings)
    )
    session_data.loc[same_session, 'session_id'] = '004-01-17-08_study1'
    sessions = summaries(session_data, session_keys)

    sessions['display_eligible'] = sessions.n >= minimum_session_bouts
    sessions.to_csv(out / 'session_profiles.csv', index=False)
    pairs, selected = matched_pairs(data, tolerance, minimum_difference, pairs_per_measure)
    pairs.to_csv(out / 'all_matched_pairs.csv', index=False)
    selected.to_csv(out / 'selected_pairs.csv', index=False)
    pairs.groupby('anchor').contrast.agg(['count', 'mean']).to_csv(out / 'matched_pair_summary.csv')
    availability = []
    for behavior, g in data.groupby('behavior'):
        for name, col in MEASURES.items():
            availability.append({'behavior':behavior,'measure':name,'n':len(measure_values(g,name,col)),'total':len(g)})
    pd.DataFrame(availability).to_csv(out/'availability.csv',index=False)
    data.groupby(['behavior','frequency_exclusion_reason']).size().rename('n').to_csv(out/'frequency_qc.csv')
    from prism.plotting.case_studies import plot_sessions, plot_pairs
    plot_sessions(sessions, out, participants, minimum_session_bouts)
    plot_pairs(selected, out)
    if export_videos:
        if video_root is None:raise ValueError('--video-root is required for export')
        export_pair_videos(selected, data, out/'matched_bout_videos', video_root)
    settings = {'participants': list(participants), 'tolerance':tolerance,
               'minimum_difference':minimum_difference, 'pairs_per_measure':pairs_per_measure,
               'minimum_session_bouts':minimum_session_bouts,'min_cycles':min_cycles,
               'minimum_mask_valid_fraction':minimum_mask_valid_fraction,'nyquist_fraction':nyquist_fraction}
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--participants',nargs='+',default=['P001','P004'])
    p.add_argument('--match-percentile-tolerance',type=float,default=.06)
    p.add_argument('--minimum-other-difference',type=float,default=.25)
    p.add_argument('--pairs-per-measure',type=int,default=3)
    p.add_argument('--minimum-session-bouts',type=int,default=3)
    p.add_argument('--min-cycles',type=float,default=2.5)
    p.add_argument('--minimum-mask-valid-fraction',type=float,default=.8)
    p.add_argument('--nyquist-fraction',type=float,default=.8)
    p.add_argument('--export-videos',action='store_true');p.add_argument('--video-root')
    a=p.parse_args()
    run_analysis(a.input,a.output_root,a.participants,a.match_percentile_tolerance,
                 a.minimum_other_difference,a.pairs_per_measure,a.min_cycles,
                 a.minimum_mask_valid_fraction,a.nyquist_fraction,a.minimum_session_bouts,a.export_videos,a.video_root)

if __name__ == "__main__":
    main()
