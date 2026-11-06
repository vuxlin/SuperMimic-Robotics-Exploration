"""Dispatch deterministic full-library evaluation to the correct robot adapter."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('robot',choices=['g1','rp1','tonypi'])
    p.add_argument('--ckpt',type=Path,required=True);p.add_argument('--gpu',default='0')
    args,extra=p.parse_known_args();ckpt=args.ckpt.resolve()
    if not ckpt.is_file():p.error(f'Checkpoint not found: {ckpt}')
    module={'g1':'g1','rp1':'rp1','tonypi':'tonypi'}[args.robot]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,MUJOCO_GL='egl',
        PYTHONPATH=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    command=[sys.executable,'-m',f'unisupermimic.tasks.motion_tracking.{module}.supermimic.evaluate',
             '--ckpt',str(ckpt),*extra]
    raise SystemExit(subprocess.call(command,cwd=ROOT,env=env))

if __name__=='__main__':main()
