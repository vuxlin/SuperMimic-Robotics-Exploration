"""Portable, named-joint model preparation for the corrected TonyPi collector."""
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/'src/unisupermimic/assets/motions/tonypi/source'
NAMES=[f'{s}_{j}' for s in ('l','r') for j in
       ('ankle_roll','ankle_pitch','knee','hip_pitch','hip_roll','elbow','shoulder_roll','shoulder_pitch')]

def xml_servo_names():
    text=(ROOT/'model/robot.xml').read_text(encoding='utf-8',errors='replace')
    pairs=re.findall(r'<position name="([^"]+)"[^>]*>\s*<!-- servo (\d+) -->',text)
    mapping={int(i):n for n,i in pairs}
    assert set(range(1,17)).issubset(mapping)
    return [mapping[i] for i in range(1,17)]

def build_model(folder):
    folder.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'model/assets',folder/'assets',dirs_exist_ok=True)
    original=mujoco.MjModel.from_xml_path(str(ROOT/'model/robot.xml'))
    text=(ROOT/'model/robot.xml').read_text(encoding='utf-8',errors='replace')
    tree=ET.ElementTree(ET.fromstring(re.sub(r'<!--.*?-->','',text,flags=re.S)));doc=tree.getroot()
    locked={'head_pan','head_tilt','l_gripper','r_gripper'}
    keep_q=list(range(7))+[int(original.jnt_qposadr[i]) for i in range(original.njnt) if original.joint(i).name in NAMES]
    keep_a=[i for i in range(original.nu) if original.actuator(i).name in NAMES]
    for parent in doc.iter():
        for child in list(parent):
            if (child.tag=='joint' and child.get('name') in locked) or child.get('joint') in locked:parent.remove(child)
    for key in doc.findall('keyframe/key'):
        q=np.fromstring(key.get('qpos',''),sep=' ')
        if len(q):key.set('qpos',' '.join(map(str,q[keep_q])))
        for attr in ('qvel','act'):key.attrib.pop(attr,None)
        control=np.fromstring(key.get('ctrl',''),sep=' ')
        if len(control):key.set('ctrl',' '.join(map(str,control[keep_a])))
    for mesh in doc.findall('asset/mesh'):mesh.set('file','assets/'+Path(mesh.get('file')).name)
    doc.find('compiler').attrib.pop('meshdir',None)
    tree.write(folder/'robot.xml',encoding='utf-8',xml_declaration=True)
    (folder/'scene.xml').write_text('<mujoco model="tonypi_supermimic"><include file="robot.xml"/>'
        '<worldbody><light pos="0 0 2"/><geom name="floor" type="plane" size="0 0 .05" '
        'friction="0.9 0.005 0.0001" condim="3" contype="1" conaffinity="15"/></worldbody></mujoco>')
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
    assert model.nq==23 and model.nu==16
    return model
