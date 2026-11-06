"""Portable launcher using the recorded robot-specific training profiles."""
import argparse
import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('robot',choices=['g1','rp1','tonypi'])
    p.add_argument('--gpu',default=os.environ.get('GPU','0'));p.add_argument('--run-dir',type=Path)
    p.add_argument('--dry-run',action='store_true');args,extra=p.parse_known_args()
    profile=OmegaConf.load(ROOT/f'configs/experiments/{args.robot}.yaml')
    run=args.run_dir or Path('runs')/f'{args.robot}_{datetime.datetime.now():%Y%m%d_%H%M%S}'
    target=run if run.is_absolute() else ROOT/run
    if target.exists() and not args.dry_run:p.error(f'Run directory exists: {target}; use a new --run-dir')
    code='from unisupermimic.scripts.train_supermimic import hydra_main; hydra_main()'
    cmd=[sys.executable,'-c',code,f'task={profile.task}',f'training.log_dir={run}',*list(profile.overrides),*extra]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),MUJOCO_GL=os.environ.get('MUJOCO_GL','egl'),
             OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS','4'),PYTHONUNBUFFERED='1',
             PYTHONPATH=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    print(f'GPU={args.gpu} cwd={ROOT}\n'+shlex.join(cmd),flush=True)
    if not args.dry_run:raise SystemExit(subprocess.call(cmd,cwd=ROOT,env=env))

if __name__=='__main__':main()
