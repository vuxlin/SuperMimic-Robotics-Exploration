"""Collect corrected TonyPi reference motions through continuous native physics.

Raw IDs follow the MJCF servo annotations, while CSV columns retain the original
canonical named-joint order. Original calibration, assets and data stay intact.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from model_utils import build_model, ROOT, NAMES, xml_servo_names
sys.path.insert(0, str(ROOT/'control'))
from servo_map import pulse_to_sim, ZERO, SIGN, PULSE_PER_RAD

DT=.004
FPS=50
G={}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as c:
        return np.array(c.execute('SELECT Time,'+','.join(f'Servo{i}' for i in range(1,17))+' FROM ActionGroup ORDER BY [Index]').fetchall(),dtype=float)


def make_model(out):
    folder=out/'model';m=build_model(folder);d=mujoco.MjData(m)
    ids=xml_servo_names();qa=np.array([int(m.joint(n).qposadr[0]) for n in ids]);aa=np.array([m.actuator(n).id for n in ids])
    target=pulse_to_sim(read_rows(ROOT/'action/stand.d6a')[-1,1:])
    assert np.all(target>=m.actuator_ctrlrange[aa,0]) and np.all(target<=m.actuator_ctrlrange[aa,1])
    mujoco.mj_resetDataKeyframe(m,d,m.key('home').id);d.qpos[qa]=target;d.ctrl[aa]=target
    tilt=[]
    for i in range(2500):
        mujoco.mj_step(m,d)
        if i%50==0:tilt.append(float(np.degrees(np.arccos(np.clip(d.xmat[m.body('torso').id].reshape(3,3)[2,2],-1,1)))))
    mujoco.mj_forward(m,d)
    assert max(tilt[-25:])<10 and d.site_xpos[m.site('imu').id,2]>.22
    tree=ET.parse(folder/'robot.xml');keys=tree.getroot().find('keyframe')
    for key in list(keys):
        if key.get('name')=='stand':keys.remove(key)
    ET.SubElement(keys,'key',dict(name='stand',qpos=' '.join(map(str,d.qpos)),ctrl=' '.join(map(str,d.ctrl))))
    tree.write(folder/'robot.xml',encoding='utf-8',xml_declaration=True)
    qa_canonical=[int(m.joint(n).qposadr[0]) for n in NAMES]
    np.save(out/'default_joint_pos.npy',d.qpos[qa_canonical])
    return dict(joint_names=NAMES,servo_id_to_joint=ids,ZERO=ZERO,SIGN=SIGN,
                pulse_per_rad=PULSE_PER_RAD,model_sha256=digest(folder/'robot.xml'),
                source_model_sha256=digest(ROOT/'model/robot.xml'),
                calibration_source_sha256=digest(ROOT/'control/servo_map.py'),
                stand_last_2s_max_tilt_deg=max(tilt[-25:]),stand_imu_height=float(d.site_xpos[m.site('imu').id,2]),
                validation_scope='MuJoCo simulation; no hardware calibration claim')


def inventory(out):
    m=mujoco.MjModel.from_xml_path(str(out/'model/scene.xml'));ids=xml_servo_names()
    aa=[m.actuator(n).id for n in ids];bounds=m.actuator_ctrlrange[aa]
    stand=pulse_to_sim(read_rows(ROOT/'action/stand.d6a')[-1,1:]);seen={};items=[]
    for path in sorted((ROOT/'action').glob('*.d6a')):
        item=dict(name=path.stem,file=str(path),source_sha256=digest(path))
        try:
            rows=read_rows(path);assert rows.ndim==2 and rows.shape[1]==17 and len(rows)>0
            assert np.isfinite(rows).all() and np.all((rows[:,1:]>=0)&(rows[:,1:]<=1000)) and np.all(rows[:,0]>0)
            key=hashlib.sha256(rows.tobytes()).hexdigest();item.update(semantic_sha256=key,source_seconds=float(rows[:,0].sum()/1000),rows=len(rows))
            if key in seen:item.update(excluded='duplicate_body_commands',duplicate_of=seen[key]);items.append(item);continue
            seen[key]=path.stem
            if path.stem.startswith('tonypi_'):item['excluded']='legacy_custom_IK_servo_convention_not_verified'
            elif path.stem.startswith(('climb_stairs','down_floor')) or path.stem=='hurdles':item['excluded']='requires_unmodeled_stairs_or_obstacle'
            if 'excluded' in item:items.append(item);continue
            q=pulse_to_sim(rows[:,1:]);delta=q-stand;upper=bounds[:,1]-.015;lower=bounds[:,0]+.015;alpha=1.
            for j in range(16):
                if delta[:,j].max()>0:alpha=min(alpha,float((upper[j]-stand[j])/delta[:,j].max()))
                if delta[:,j].min()<0:alpha=min(alpha,float((lower[j]-stand[j])/delta[:,j].min()))
            item.update(feasible_amplitude=float(np.clip(alpha*.98 if alpha<1 else 1,.0,1)),
                        max_original_limit_violation_deg=float(np.degrees(np.maximum(bounds[:,0]-q,0).max()+np.maximum(q-bounds[:,1],0).max())))
            if item['feasible_amplitude']<.25:item['excluded']='requires_amplitude_below_0.25'
        except Exception as e:item['excluded']='invalid_source';item['error']=repr(e)
        items.append(item)
    (out/'source_inventory.json').write_text(json.dumps(items,indent=2))
    return items


def init_worker(out, sources):
    out=Path(out);m=mujoco.MjModel.from_xml_path(str(out/'model/scene.xml'));ids=xml_servo_names()
    qa=np.array([int(m.joint(n).qposadr[0]) for n in NAMES]);va=np.array([int(m.joint(n).dofadr[0]) for n in NAMES])
    ia=np.array([int(m.joint(n).qposadr[0]) for n in ids]);aa=np.array([m.actuator(n).id for n in ids])
    G.update(out=out,m=m,qa=qa,va=va,ia=ia,aa=aa,ids=ids,
             rows={x['name']:read_rows(x['file']) for x in sources if 'excluded' not in x},
             stand=pulse_to_sim(read_rows(ROOT/'action/stand.d6a')[-1,1:]),
             imu=m.site('imu').id,torso=m.body('torso').id)


def run(spec):
    m=G['m'];d=mujoco.MjData(m);qa=G['qa'];va=G['va'];aa=G['aa'];ia=G['ia']
    initial=spec.get('initial','stand');mujoco.mj_resetDataKeyframe(m,d,m.key(initial).id)
    if initial!='stand':d.ctrl[aa]=d.qpos[ia]
    for _ in range(125):mujoco.mj_step(m,d)
    records=[];velocities=[];targets=[];heights=[];tilts=[];penetrations=[];events=[];step=0;clip_amount=0.
    def sample():
        mujoco.mj_forward(m,d)
        records.append(np.r_[d.qpos[:3],d.qpos[4:7],d.qpos[3],d.qpos[qa]])
        velocities.append(np.r_[d.qvel[:6],d.qvel[va]]);targets.append(d.ctrl[aa].copy())
        heights.append(float(d.site_xpos[G['imu'],2]));tilts.append(float(np.degrees(np.arccos(np.clip(d.xmat[G['torso']].reshape(3,3)[2,2],-1,1)))))
        penetrations.append(float(min(0.,np.min(d.contact.dist[:d.ncon]))) if d.ncon else 0.)
    sample()
    for command in spec['commands']:
        name=command['name'];start=step/250
        if name=='@hold':
            rows=[(float(command.get('seconds',.25)),None)]
        else:
            alpha=command.get('amplitude',spec.get('amplitude',1.));tempo=command.get('tempo',spec.get('tempo',1.))
            source=G['rows'][name]
            rows=[(max(.02,r[0]/1000*tempo),G['stand']+alpha*(pulse_to_sim(r[1:])-G['stand'])) for r in source]
            if name=='stand':rows=[(max(command.get('seconds',.8),rows[-1][0]),rows[-1][1])]
        for duration,goal in rows:
            count=max(5,int(round(duration/DT)));q0=d.qpos[ia].copy()
            if goal is None:goal=d.ctrl[aa].copy()
            bounds=m.actuator_ctrlrange[aa];clipped=np.clip(goal,*bounds.T);clip_amount=max(clip_amount,float(np.max(np.abs(clipped-goal))))
            for k in range(count):
                d.ctrl[aa]=q0+(clipped-q0)*((k+1)/count);mujoco.mj_step(m,d);step+=1
                if step%5==0:sample()
        events.append(dict(action=name,start_seconds=start,end_seconds=step/250,
                           amplitude=command.get('amplitude',spec.get('amplitude',1.)),tempo=command.get('tempo',spec.get('tempo',1.))))
    arr=np.asarray(records);vel=np.asarray(velocities)
    for k in range(1,len(arr)):
        if arr[k-1,3:7]@arr[k,3:7]<0:arr[k,3:7]*=-1
    reasons=[];kind=spec.get('kind','upright')
    if not np.isfinite(arr).all() or not np.isfinite(vel).all():reasons.append('nonfinite')
    if clip_amount>.025:reasons.append('command_limit_violation')
    if min(penetrations)<-.025:reasons.append('penetration_gt_2.5cm')
    if np.max(np.linalg.norm(np.diff(arr[:,:3],axis=0),axis=1))>.06:reasons.append('root_step_gt_6cm')
    if np.max(np.abs(np.diff(arr[:,7:],axis=0)))>.6:reasons.append('joint_step_gt_0.6rad')
    if np.max(np.abs(vel[:,6:]))>30:reasons.append('joint_speed_gt_30rad_s')
    if kind=='upright':
        if max(tilts)>50:reasons.append('unexpected_tilt_gt_50deg')
        if min(heights)<.15:reasons.append('unexpected_imu_below_15cm')
    if kind in ['upright','recovery']:
        if tilts[-1]>22 or heights[-1]<.21:reasons.append('not_upright_at_end')
    # All state samples come from consecutive physics steps; no root/FK stitching.
    target=G['out']/('train' if spec.get('split')=='train' else 'validation')/(spec['name']+'.csv')
    if reasons:target=G['out']/'quarantine'/(spec['name']+'.csv')
    np.savetxt(target,arr,delimiter=',',fmt='%.9g')
    np.savez_compressed(G['out']/'states'/(spec['name']+'.npz'),reference=arr,qvel=vel,position_targets=np.asarray(targets),fps=50.,joint_names=np.asarray(NAMES))
    result=dict(spec,accepted=not reasons,reasons=reasons,frames=len(arr),seconds=(len(arr)-1)/50,
                max_tilt_deg=max(tilts),final_tilt_deg=tilts[-1],min_imu_height=min(heights),final_imu_height=heights[-1],
                max_penetration_m=-min(penetrations),root_xy_distance=float(np.linalg.norm(arr[-1,:2]-arr[0,:2])),
                max_joint_speed=float(np.max(np.abs(vel[:,6:]))),max_joint_step=float(np.max(np.abs(np.diff(arr[:,7:],axis=0)))),
                max_root_step=float(np.max(np.linalg.norm(np.diff(arr[:,:3],axis=0),axis=1))),
                command_clip_max_rad=clip_amount,events=events,sha256=digest(target),file=target.relative_to(G['out']).as_posix())
    (G['out']/'status'/(spec['name']+'.json')).write_text(json.dumps(result,indent=2))
    return result


def commands_for(name, available):
    predecessor={'stand_slow':'squat','put_down':'move_up','put_down2':'move_up','put_down3':'move_up',
                 'squat_up':'squat_down','lift_down':'lift_up',
                 'grab_squat_up_left':'grab_squat_left','grab_squat_up_right':'grab_squat_right'}
    commands=[]
    if name in predecessor and predecessor[name] in available:commands.append(dict(name=predecessor[name]))
    commands.append(dict(name=name))
    return commands


def collect(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert not (out/'manifest.json').exists(),'Refusing to overwrite a completed dataset'
    for folder in ['train','validation','quarantine','states','status']:(out/folder).mkdir(exist_ok=True)
    binding=make_model(out);(out/'servo_binding.json').write_text(json.dumps(binding,indent=2))
    items=inventory(out);sources=[x for x in items if 'excluded' not in x];available={x['name']:x for x in sources}
    specs=[]
    floor={'creep_forward'};falls={'fall_forward','fall_back'};recovery={'stand_up_front':'prone','stand_up_back':'supine','sit_ups':'supine'}
    for item in sources:
        name=item['name'];cmds=commands_for(name,available)
        alpha=min(available[c['name']]['feasible_amplitude'] for c in cmds)
        kind='floor' if name in floor|falls else 'recovery' if name in recovery else 'upright'
        if kind!='floor':cmds += [dict(name='stand',seconds=1.25),dict(name='@hold',seconds=.5)]
        else:cmds += [dict(name='@hold',seconds=.6)]
        # Keep immutable source provenance; amplitudes/tempo are explicit augmentations.
        for variant,mult in [('base',1.),('soft',.8)]:
            if name in ['stand','0'] and variant=='soft':continue
            specs.append(dict(name=f'single_{name}_{variant}',family=name,kind=kind,initial=recovery.get(name,'prone' if name in floor else 'stand'),
                              split='train',amplitude=alpha*mult,tempo=1.,commands=cmds,stage='single'))
    results=[]
    def progress(stage,done,total):
        (out/'pipeline_status.json').write_text(json.dumps(dict(stage=stage,done=done,total=total,accepted=sum(x['accepted'] for x in results),updated_at=time.time()),indent=2))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(str(out),sources)) as pool:
        futures={pool.submit(run,s):s for s in specs};progress('screening_sources',0,len(specs))
        for done,f in enumerate(as_completed(futures),1):
            result=f.result();results.append(result);progress('screening_sources',done,len(specs))
            if done%10==0 or not result['accepted']:print('SOURCE',done,len(specs),result['name'],result['accepted'],','.join(result['reasons']),flush=True)
        failed_families={x['family'] for x in results if x['kind']=='upright'}-{x['family'] for x in results if x['accepted']}
        rescue=[]
        for family in sorted(failed_families):
            original=next(s for s in specs if s['family']==family)
            for index,(mult,tempo) in enumerate([(.6,1.2),(.45,1.3),(.35,1.5)]):
                spec=dict(original);spec.update(name=f'rescue_{family}_{index}',amplitude=original['amplitude']*mult,tempo=tempo,stage='single_rescue')
                rescue.append(spec)
        futures={pool.submit(run,s):s for s in rescue};progress('rescuing_unstable_sources',0,len(rescue))
        for done,f in enumerate(as_completed(futures),1):
            result=f.result();results.append(result);progress('rescuing_unstable_sources',done,len(rescue))
            print('RESCUE',result['name'],result['accepted'],','.join(result['reasons']),flush=True)
        bank=[x for x in results if x['accepted'] and x['kind']=='upright' and x['family'] not in ['stand','0','stand_slow'] and x['seconds']>1.8]
        assert len({x['family'] for x in bank})>=12,'Too few distinct stable sources; inspect mapping before training'
        rng=random.Random(20261104);mix=[];signatures=set()
        for i in range(args.sequences+args.validation_sequences):
            split='train' if i<args.sequences else 'validation';target=rng.uniform(22,52);duration=0.;cmds=[];families=[]
            while duration<target:
                choices=[b for b in bank if b['family'] not in families[-2:] and (not b['family'].isdigit() or duration<5)]
                block=rng.choice(choices);tempo=rng.choice([.9,1.,1.12]);families.append(block['family'])
                for c in block['commands']:
                    cmd=dict(c);cmd.update(amplitude=block['amplitude'],tempo=tempo)
                    if c['name']=='stand':cmd['seconds']=rng.uniform(.7,1.25)
                    if c['name']=='@hold':cmd['seconds']=rng.uniform(.12,.35)
                    cmds.append(cmd)
                duration+=block['seconds']*tempo
            signature=json.dumps(cmds,sort_keys=True);assert signature not in signatures;signatures.add(signature)
            mix.append(dict(name=f'{split}_sequence_{i:04d}',family='mixed',source_families=families,kind='upright',initial='stand',
                            split=split,amplitude=1.,tempo=1.,commands=cmds,stage='continuous_sequence'))
        # Gait loops keep the natural phase; no stand reset between individual cycles.
        for name,start,end in [('go_forward','go_forward_start','go_forward_end'),('go_forward_fast','go_forward_start_fast','go_forward_end'),
                               ('back','back_start','back_end'),('back_fast','back_start','back_end'),
                               ('stepping',None,None),('left_move',None,None),('right_move',None,None),
                               ('turn_left_fast',None,None),('turn_right_fast',None,None)]:
            if name not in available:continue
            participating=[n for n in [start,name,end] if n in available];alpha=min(available[n]['feasible_amplitude'] for n in participating)
            for repeat in [8,16,24]:
                cmds=([dict(name=start)] if start in available else [])+[dict(name=name)]*repeat+([dict(name=end)] if end in available else [])
                cmds += [dict(name='stand',seconds=1.25),dict(name='@hold',seconds=.5)]
                mix.append(dict(name=f'gait_{name}_{repeat:02d}',family=name,kind='upright',initial='stand',split='train',amplitude=alpha,tempo=1.,commands=cmds,stage='continuous_gait'))
        futures={pool.submit(run,s):s for s in mix};progress('collecting_sequences',0,len(mix))
        for done,f in enumerate(as_completed(futures),1):
            result=f.result();results.append(result);progress('collecting_sequences',done,len(mix))
            if done%10==0 or not result['accepted']:print('SEQUENCE',done,len(mix),result['name'],result['accepted'],','.join(result['reasons']),flush=True)
    # Holdout audit: a different tempo is not an unseen full primitive order.
    train_orders={tuple(x.get('source_families', [])) for x in results
                  if x['accepted'] and x['split']=='train' and x['stage']=='continuous_sequence'}
    heldout_orders=set()
    for item in results:
        if item['accepted'] and item['split']=='validation':
            order=tuple(item.get('source_families', []))
            if order in train_orders or order in heldout_orders:
                item['split']='diagnostic_not_heldout'
                item['holdout_exclusion']='Full primitive order overlaps training or another validation sequence'
            else:heldout_orders.add(order)
    train=[x for x in results if x['accepted'] and x['split']=='train'];validation=[x for x in results if x['accepted'] and x['split']=='validation']
    assert len(train)>=40 and len(validation)>=8,'Insufficient validated library'
    assert sum(x['seconds'] for x in train)>600,'Insufficient useful duration'
    manifest=dict(schema_version=2,dataset='TonyPi corrected physical continuous sequences v2',fps=50.,quaternion_order='xyzw',
                  joint_names=NAMES,servo_binding=binding,model_sha256=binding['model_sha256'],source_inventory=items,items=results,
                  training_clips=len(train),training_frames=sum(x['frames'] for x in train),training_seconds=sum(x['seconds'] for x in train),
                  validation_clips=len(validation),validation_frames=sum(x['frames'] for x in validation),validation_seconds=sum(x['seconds'] for x in validation),
                  accepted_source_families=sorted({x['family'] for x in results if x['accepted'] and x['stage'] in ['single','single_rescue']}),
                  independent_holdout='Unseen sequence orders/tempo combinations of shared source primitives; not unseen skills',
                  generation='Every sequence is one uninterrupted MuJoCo rollout; no qpos/root CSV stitching',
                  original_files_modified=False,locked_channels='Head and grippers fixed; object-related clips provide body pose only')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));progress('complete',len(results),len(results))
    print(json.dumps({k:manifest[k] for k in ['training_clips','training_frames','training_seconds','validation_clips','validation_seconds','accepted_source_families']}),flush=True)


if __name__=='__main__':
    multiprocessing.freeze_support();p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'data_build/tonypi')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--sequences',type=int,default=160);p.add_argument('--validation-sequences',type=int,default=32)
    collect(p.parse_args())
