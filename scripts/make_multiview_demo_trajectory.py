#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--frequency", type=float, default=1.5)
    args = parser.parse_args()

    time = np.arange(int(args.fps * args.duration)) / args.fps
    base = np.array([
        [0.0, 1.75, 0.0], [0.0, 1.45, 0.0],
        [-0.22, 1.42, 0.0], [0.22, 1.42, 0.0],
        [-0.45, 1.18, 0.0], [0.45, 1.18, 0.0],
        [-0.55, 0.95, 0.0], [0.55, 0.95, 0.0],
        [-0.16, 0.90, 0.0], [0.16, 0.90, 0.0],
        [-0.18, 0.45, 0.0], [0.18, 0.45, 0.0],
        [-0.20, 0.02, 0.0], [0.20, 0.02, 0.0],
    ])
    joints = np.repeat(base[None], len(time), axis=0)
    phase = 2.0 * np.pi * args.frequency * time
    # Arm flapping plus a smaller anterior-posterior body rock.
    joints[:, 4:8, 1] += 0.20 * np.sin(phase)[:, None]
    joints[:, 4:8, 2] += 0.12 * np.cos(phase)[:, None]
    joints[:, :10, 2] += 0.08 * np.sin(phase)[:, None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        joints_3d=joints,
        fps=args.fps,
        condition_id="demo_flap_rock",
        behavior="flap_rock",
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

