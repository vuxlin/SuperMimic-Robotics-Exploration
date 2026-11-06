"""Evaluate only manifest-approved TonyPi composition-validation clips."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--ckpt',type=Path,required=True);p.add_argument('--gpu',default='0');args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu;os.environ.setdefault('MUJOCO_GL','egl');os.chdir(ROOT)
    import torch
    import unisupermimic
    from hydra import initialize_config_dir,compose
    from unilab.training import ensure_registries
    from unilab.base.config_adapter import create_env
    from unisupermimic.scripts.train_supermimic import build_env_override
    from unisupermimic.tasks.motion_tracking.tonypi.supermimic.evaluate import evaluate
    folder=ROOT/'src/unisupermimic/assets/motions/tonypi';m=json.loads((folder/'manifest.json').read_text())
    clips=[x for x in m['items'] if x['accepted'] and x['split']=='validation'];assert len(clips)==27
    with initialize_config_dir(version_base='1.3',config_dir=str(Path(unisupermimic.__file__).parent/'conf/supermimic')):
        cfg=compose(config_name='config',overrides=['task=tonypi_super_mimic/mjwarp','training.play_only=true'])
    cfg.env.commands.motion.motion_files=[str((folder/x['file']).relative_to(ROOT)) for x in clips]
    ensure_registries();env=create_env(cfg,num_envs=27,env_cfg_override=build_env_override(cfg,eval_mode=True,learner_device='cuda:0'))
    try:
        checkpoint=torch.load(args.ckpt.resolve(),map_location='cuda:0',weights_only=True)
        result=evaluate(env,checkpoint)
        target=args.ckpt.resolve().with_name(args.ckpt.stem+'_validation_eval.json')
        target.write_text(json.dumps(result,indent=2));print(target)
    finally:env.close();env.backend.close()

if __name__=='__main__':main()
