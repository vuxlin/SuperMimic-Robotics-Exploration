"""Build display-only comparison scenes for robot-matched policy/reference qpos."""
from pathlib import Path
import copy,tempfile,xml.etree.ElementTree as ET
import mujoco
def comparison_xml(scene: Path) -> str:
    scene=Path(scene).resolve()
    model=mujoco.MjModel.from_xml_path(str(scene))
    with tempfile.NamedTemporaryFile(suffix='.xml',prefix='.compare_export_',dir=scene.parent,delete=False) as f:tmp=Path(f.name)
    try:
        mujoco.mj_saveLastXML(str(tmp),model)
        root=ET.parse(tmp).getroot()
    finally:tmp.unlink(missing_ok=True)
    world=root.find('worldbody');bodies=world.findall('body')
    if len(bodies)!=1:raise ValueError('Comparison expects one free-root robot')
    ghost=copy.deepcopy(bodies[0])
    groups={}
    for default in root.findall('.//default'):
        geom=default.find('geom')
        if default.get('class') and geom is not None:groups[default.get('class')]=int(geom.get('group',0))
    for parent in ghost.iter():
        for geom in list(parent.findall('geom')):
            name=geom.get('name')
            group=int(model.geom(name).group.item()) if name else int(geom.get('group',groups.get(geom.get('class'),0)))
            if group>=3:
                geom.set('contype','0');geom.set('conaffinity','0');geom.set('group','3');continue
            geom.attrib.pop('material',None)
            geom.set('rgba','0.22 0.56 1 0.26');geom.set('contype','0');geom.set('conaffinity','0');geom.set('group','1')
    renames={x.get('name'):'ref_'+x.get('name') for x in ghost.iter() if x.get('name')}
    for element in ghost.iter():
        if element.get('name'):element.set('name',renames[element.get('name')])
        for attr in ['target','body','joint','site']:
            if element.get(attr) in renames:element.set(attr,renames[element.get(attr)])
    world.append(ghost)
    key=root.find('keyframe')
    if key is not None:root.remove(key)
    return ET.tostring(root,encoding='unicode')
def load_comparison_model(scene: Path):
    scene=Path(scene).resolve()
    with tempfile.NamedTemporaryFile(mode='w',suffix='.xml',prefix='.comparison_',dir=scene.parent,delete=False,encoding='utf-8') as f:
        f.write(comparison_xml(scene));tmp=Path(f.name)
    try:return mujoco.MjModel.from_xml_path(str(tmp))
    finally:tmp.unlink(missing_ok=True)

