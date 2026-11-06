"""Copy one indexed checkpoint and its run configuration, verifying the content hash."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('robot',choices=['g1','rp1','tonypi'])
    p.add_argument('--iteration',type=int,choices=[2000,6000,12000],default=12000)
    args=p.parse_args();items=json.loads((ROOT/'results/checkpoints.json').read_text())['checkpoints']
    row=next(x for x in items if x['robot']==args.robot and x['iteration']==args.iteration)
    src=Path(row['source_path']);dst=ROOT/'checkpoints'/args.robot/src.name
    if not src.is_file():p.error('Original checkpoint is unavailable on this machine; obtain the file identified by its manifest SHA256.')
    if digest(src)!=row['sha256']:p.error('Source checkpoint hash differs from the recorded experiment')
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if digest(dst)!=row['sha256']:p.error(f'Refusing to overwrite different file: {dst}')
    else:shutil.copy2(src,dst)
    shutil.copy2(ROOT/'results'/args.robot/'run_config.json',dst.parent/'run_config.json')
    print(dst)

if __name__=='__main__':main()
