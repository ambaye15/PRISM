import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from .config import (BEHAVIORS, METRICS, PARAMETERS, design, scales_for, signature, workspace, manifest)
from .model import PortableDecoder, inputs, train_decoder


def worker(row, results_dir, profile_power, pipeline_signature, force=False):
    path = Path(results_dir)/"records"/(row["condition_id"]+".json")
    row_hash = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    if path.exists() and not force:
        cached = json.loads(path.read_text())
        if (cached.get("pipeline_signature") == pipeline_signature and
                cached.get("input_row_sha256") == row_hash and cached.get("status") == "ok"):
            return cached
    record = dict(row, pipeline_signature=pipeline_signature, input_row_sha256=row_hash,
                  flow_lag_profile_power=profile_power)
    try:
        from .simulation import render, measure
        frames, masks = render(row)
        record["image_sha256"] = hashlib.sha256(frames.tobytes()+masks.tobytes()).hexdigest()
        record.update(measure(frames, masks, profile_power=profile_power))
        record["status"] = "ok"
        if any(record[x] is None or not np.isfinite(record[x]) for x in METRICS):
            record["status"] = "unavailable"
    except Exception as exc:
        record.update(status="error", error=type(exc).__name__+": "+str(exc))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return record


def batch(rows, results_dir, profile_power, workers=4, force=False):
    if workers < 1 or not rows:
        raise ValueError("workers must be positive and the design nonempty")
    code_signature = signature(profile_power)
    out = []
    if workers == 1:
        for row in rows:
            item = worker(row, results_dir, profile_power, code_signature, force)
            out.append(item)
            print(f'{len(out)}/{len(rows)} {item["condition_id"]} {item["status"]}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, row, str(results_dir), profile_power,
                                   code_signature, force) for row in rows]
            for future in as_completed(futures):
                item = future.result()
                out.append(item)
                if len(out) % 12 == 0 or len(out) == len(rows):
                    print(f'{len(out)}/{len(rows)} {item["condition_id"]} {item["status"]}', flush=True)
    return pd.DataFrame(out).sort_values("condition_id")


def run_library(results_dir, workers=4, profile_power=None, force=False):
    root, meta = workspace(results_dir, profile_power)
    rows = design()
    (root/"design.json").write_text(json.dumps(rows, indent=2))
    data = batch(rows, root, meta["flow_lag_profile_power"], workers, force)
    data.to_csv(root/"library_and_test.csv", index=False)
    meta.update(archived_reference=False, source="rendered with installed PRISM",
                pipeline_signature=signature(meta["flow_lag_profile_power"]))
    (root/"experiment.json").write_text(json.dumps(meta, indent=2))
    return data


def run_decode(results_dir, workers=4, retrain=False, force=False):
    root = Path(results_dir)
    meta = manifest(root)
    data = pd.read_csv(root/"library_and_test.csv")
    rows = evaluate(data, root, retrain)
    decoded = batch(rows, root, meta["flow_lag_profile_power"], workers, force)
    decoded.to_csv(root/"decoded.csv", index=False)
    summarize(data, decoded, root)
    return decoded

def evaluate(data, results_dir, retrain=False):
    root = Path(results_dir)
    library = data[(data.split == 'library') & (data.status == 'ok')].copy()
    test = data[data.split == 'test'].copy(); scales = scales_for(library)
    assert not set(library.image_sha256) & set(test.image_sha256.dropna())
    assert not set(library.seed) & set(test.seed)
    if retrain or not (root/'decoder.npz').exists():
        train_decoder(root)
    decoder = PortableDecoder.load(root)
    rows, decoded = [], []
    rng = np.random.default_rng(20261017)
    for query in test.to_dict('records'):
        if query['status'] != 'ok': continue
        group = library[library.behavior == query['behavior']].reset_index(drop=True)
        scale = scales[query['behavior']]; target = np.array([query[k] for k in METRICS])
        delta = (group[METRICS].to_numpy(float) - target) / scale
        dist = np.sqrt(np.mean(delta**2, axis=1)); order = np.argsort(dist, kind='stable')

        three = np.sqrt(np.mean(delta[:, :3]**2, axis=1))
        choices = [('retrieval', int(order[0])), ('without_ps', int(np.argmin(three)))]
        for method, idx in choices:
            candidate = group.iloc[idx]
            rows.append(dict(query_id=query['condition_id'], behavior=query['behavior'], method=method,
                candidate_id=candidate.condition_id, profile_error=float(dist[idx]),
                **{k+'_abs_error':float(abs(candidate[k]-query[k])) for k in METRICS},
                **{k+'_scaled_error':float(abs(candidate[k]-query[k])/scale[j]) for j,k in enumerate(METRICS)}))
        random_idx = rng.integers(0,len(group),size=100)
        rows.append(dict(query_id=query['condition_id'], behavior=query['behavior'], method='random_100_mean',
            candidate_id='', profile_error=float(dist[random_idx].mean()),
            **{k+'_abs_error':float(abs(group.iloc[random_idx][k].to_numpy()-query[k]).mean()) for k in METRICS},
            **{k+'_scaled_error':float(abs(group.iloc[random_idx][k].to_numpy()-query[k]).mean()/scale[j]) for j,k in enumerate(METRICS)}))
        predicted = decoder.predict(inputs(pd.DataFrame([query])))[0]
        clipped = np.clip(predicted, [.75,.6,0.], [2.,1.4,.75])
        decoded.append(dict(condition_id=query['condition_id']+'__decoded', query_id=query['condition_id'],
            behavior=query['behavior'], split='decoded', seed=3000+len(decoded),
            frequency_input=float(clipped[0]), amplitude_scale=float(clipped[1]), disruption=float(clipped[2]),
            parameter_clipped=bool(np.any(clipped != predicted))))
    pd.DataFrame(rows).to_csv(root/'retrieval_evaluation.csv',index=False)
    (root/'decoded_design.json').write_text(json.dumps(decoded,indent=2))
    (root/'scales.json').write_text(json.dumps({k:v.tolist() for k,v in scales.items()},indent=2))
    return decoded

def summarize(data, decoded, results_dir):
    root = Path(results_dir)
    scores = pd.read_csv(root/'retrieval_evaluation.csv')
    scales = json.loads((root/'scales.json').read_text())
    additions = []
    for r in decoded.to_dict('records'):
        if r['status'] != 'ok': continue
        q = data.set_index('condition_id').loc[r['query_id']]
        delta = np.array([abs(r[k]-q[k]) for k in METRICS]); scale=np.array(scales[r['behavior']])
        additions.append(dict(query_id=r['query_id'],behavior=r['behavior'],method='neural_decoder',
             candidate_id=r['condition_id'], profile_error=float(np.sqrt(np.mean((delta/scale)**2))),
             **{k+'_abs_error':float(delta[j]) for j,k in enumerate(METRICS)},
             **{k+'_scaled_error':float(delta[j]/scale[j]) for j,k in enumerate(METRICS)}))
    scores=pd.concat([scores,pd.DataFrame(additions)],ignore_index=True)
    scores.to_csv(root/'evaluation.csv',index=False)
    summary=scores.groupby('method').agg(n=('profile_error','size'),profile_error=('profile_error','median'),
       intensity_mae=('intensity_abs_error','mean'), extent_mae=('extent_abs_error','mean'),
       frequency_mae=('frequency_abs_error','mean'),ps_mae=('ps_abs_error','mean'))
    summary.to_csv(root/'summary.csv');print(summary.to_string(),flush=True)
