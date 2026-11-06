"""Render saved TonyPi trajectories with a world-fixed, metric display grid.

No simulation steps or policy inference are performed. Physical XML and weights
are unchanged; render decorations are appended only to the visualization scene.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL','egl')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from unisupermimic.tasks.motion_tracking.comparison_render import comparison_xml

ROOT=Path(__file__).resolve().parents[1]
JOBS=[(12000,'train_sequence_0000','videos/tonypi',False),(2000,'train_sequence_0000','videos/tonypi_002000',False),(6000,'train_sequence_0000','videos/tonypi_006000',False)]


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load_scene():
    scene=ROOT/'src/unisupermimic/assets/robots/tonypi/scene.xml'
    xml=ET.fromstring(comparison_xml(scene));asset=xml.find('asset')
    if asset is None:asset=ET.SubElement(xml,'asset')
    if not any(e.get('type')=='skybox' for e in asset.findall('texture')):
        ET.SubElement(asset,'texture',dict(name='showcase_sky',type='skybox',builtin='gradient',
             rgb1='0.89 0.93 0.97',rgb2='0.98 0.99 1',width='512',height='512'))
    with tempfile.NamedTemporaryFile(mode='w',suffix='.xml',dir=scene.parent,delete=False) as f:
        f.write(ET.tostring(xml,encoding='unicode'));path=Path(f.name)
    try:model=mujoco.MjModel.from_xml_path(str(path))
    finally:path.unlink()
    model.geom_rgba[model.geom('floor').id]=[.92,.94,.96,1]
    return model

def grid(scene,center):
    # Connector endpoints are in world coordinates, never relative to the camera.
    x0,y0=np.floor(np.asarray(center)/.25)*.25
    for axis in (0,1):
        for i in range(-30,31):
            offset=i*.05
            if axis==0:a=[x0+offset,y0-1.5,.0007];b=[x0+offset,y0+1.5,.0007]
            else:a=[x0-1.5,y0+offset,.0007];b=[x0+1.5,y0+offset,.0007]
            major=i%5==0
            g=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),
                               np.array([.46,.52,.58,.9]) if major else np.array([.65,.70,.75,.75]))
            mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,1.2 if major else .65,a,b)
            scene.ngeom+=1

def label(image,lines,font):
    draw=ImageDraw.Draw(image)
    for y,text in lines:
        box=draw.textbbox((14,y),text,font=font)
        draw.rectangle((box[0]-5,box[1]-3,box[2]+5,box[3]+3),fill=(249,251,253))
        draw.text((14,y),text,font=font,fill=(28,47,63))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=1);parser.add_argument('--force',action='store_true');parser.add_argument('--only')
    args=parser.parse_args();model=load_scene();data=mujoco.MjData(model)
    names=json.loads((ROOT/'assets/metadata/tonypi_012000.json').read_text())['joint_names']
    qa=np.array([int(model.joint(n).qposadr[0]) for n in names]);rq=np.array([int(model.joint('ref_'+n).qposadr[0]) for n in names])
    assert model.nq==46 and sorted(qa)==list(range(7,23)) and sorted(rq)==list(range(30,46))
    width,height=960,600;model.vis.global_.offwidth=width;model.vis.global_.offheight=height
    model.vis.headlight.ambient[:]=.5;model.vis.headlight.diffuse[:]=.5
    opt=mujoco.MjvOption();opt.geomgroup[3:]=0
    feet_opt=mujoco.MjvOption();feet_opt.geomgroup[3:]=0;feet_opt.geomgroup[1]=0
    cam=mujoco.MjvCamera();cam.azimuth=110;cam.elevation=-20
    feet_cam=mujoco.MjvCamera();feet_cam.distance=.42;feet_cam.azimuth=110;feet_cam.elevation=-22
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    with mujoco.Renderer(model,width=width,height=height) as renderer:
        for iteration,name,relative,feet in JOBS[args.worker::args.workers]:
            if args.only and name!=args.only:continue
            output=ROOT/'assets'/(relative+'.mp4');status=ROOT/'assets/metadata'/(output.stem+'_render.json')
            if output.exists() and status.exists() and not args.force:continue
            source=ROOT/'assets/trajectories'/f'tonypi_{iteration:06d}_{name}.npz'
            arrays=np.load(source);policy=arrays['policy'];reference=arrays['reference']
            rows=json.loads((ROOT/f'assets/metadata/tonypi_{iteration:06d}.json').read_text())['items']
            meta=next(x for x in rows if x['name']==name)
            assert np.isfinite(policy).all() and np.isfinite(reference).all()
            xy=np.concatenate((policy[:,:2],reference[:,:2]));center=(xy.min(0)+xy.max(0))/2
            cam.distance=.95+float(np.ptp(xy,axis=0).max())*.55;cam.lookat[:]=[center[0],center[1],.16]
            temp=output.with_suffix('.partial.mp4');started=time.monotonic();count=0
            with imageio.get_writer(temp,fps=25,codec='libx264',quality=None,macro_block_size=2,
                  ffmpeg_params=['-crf','21','-preset','veryfast','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart']) as writer:
                for index in range(0,len(reference),2):
                    p=policy[min(index,len(policy)-1)];r=reference[index]
                    data.qpos[:7]=p[:7];data.qpos[qa]=p[7:]
                    data.qpos[23:30]=r[:7];data.qpos[rq]=r[7:];data.qpos[23:25]=p[:2]
                    mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_camlight(model,data)
                    renderer.update_scene(data,camera=cam,scene_option=opt);grid(renderer.scene,center)
                    frame=Image.fromarray(renderer.render())
                    top=f'TonyPi | {name} | Policy {iteration:,}'
                    timing=f'{index/50:.2f} / {meta["target_seconds"]:.2f} s | 1x real time'
                    if meta['failed'] and index>=len(policy):timing+=' | TERMINATED - policy frozen'
                    label(frame,[(12,top),(39,timing),(height-32,'Solid: policy | Blue: reference (XY aligned) | Grid: 5 cm / 25 cm')],font)
                    if feet:
                        feet_cam.lookat[:]=[p[0],p[1],.055]
                        renderer.update_scene(data,camera=feet_cam,scene_option=feet_opt)
                        grid(renderer.scene,p[:2]);close=Image.fromarray(renderer.render())
                        err=np.linalg.norm(p[:2]-r[:2])*100
                        label(close,[(12,'Policy feet | camera follows robot; grid stays in world'),(39,f'Global XY error: {err:.1f} cm'),(height-32,'Minor grid: 5 cm | Major grid: 25 cm')],font)
                        combined=Image.new('RGB',(2*width,height));combined.paste(frame,(0,0));combined.paste(close,(width,0));frame=combined
                    writer.append_data(np.asarray(frame));count+=1
                    if index==(len(reference)//3)//2*2:
                        frame.save(ROOT/'assets/images'/(output.stem+'.jpg'),quality=92)
            temp.replace(output)
            record=dict(robot='tonypi',iteration=iteration,clip=name,video=output.relative_to(ROOT).as_posix(),
                poster='assets/images/'+output.stem+'.jpg',frames=count,fps=25,seconds=count/25,
                reference_seconds=meta['target_seconds'],executed_seconds=meta['executed_seconds'],failed=meta['failed'],
                width=width*(2 if feet else 1),height=height,source_trajectory=source.relative_to(ROOT).as_posix(),
                source_sha256=digest(source),sha256=digest(output),grid_minor_m=.05,grid_major_m=.25,
                reference_xy_aligned=True,world_fixed_main_camera=True,policy_freezes_after_termination=True,
                physics_steps_performed=0,render_seconds=time.monotonic()-started)
            status.write_text(json.dumps(record,indent=2));print('RENDERED',output.name,count,round(record['render_seconds'],1),flush=True)

if __name__=='__main__':main()
