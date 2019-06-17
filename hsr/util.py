from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import re
import tempfile
from typing import List
import xml.etree.ElementTree as ET

from gym.spaces import Box

from hsr.env import get_xml_filepath
from rl_utils import parse_space, parse_vector


def add_env_args(parser):
    parser.add_argument(
        '--image-dims',
        type=parse_vector(length=2, delim=','),
        default='800,800')
    parser.add_argument('--block-space', type=parse_space(dim=4))
    parser.add_argument('--min-lift-height', type=float, default=None)
    parser.add_argument('--obs-type', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render-freq', type=int, default=None)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--record-separate-episodes', action='store_true')
    parser.add_argument('--record-freq', type=int, default=None)
    parser.add_argument('--record-path', type=Path, default=None)
    parser.add_argument('--steps-per-action', type=int, required=True)


def add_wrapper_args(parser):
    parser.add_argument('--xml-file', type=Path, default='models/world.xml')
    parser.add_argument('--set-xml', type=xml_setter, action='append')
    parser.add_argument('--use-dof', type=str, action='append', default=[])
    parser.add_argument('--geofence', type=float, required=True)
    parser.add_argument('--n-blocks', type=int, default=0)
    parser.add_argument(
        '--goal-space', type=parse_space(dim=3), required=True)  # TODO


def xml_setter(arg: str):
    return XMLSetter(*arg.split(','))
    # setters = [XMLSetter(*v.split(',')) for v in arg]
    # mirroring = [XMLSetter(p.replace('_l_', '_r_'), v)
    #              for p, v in setters if '_l_' in p] \
    #             + [XMLSetter(p.replace('_r_', '_l_'), v)
    #                for p, v in setters if '_r_' in p]
    # return [s._replace(path=s.path) for s in setters + mirroring]


def env_wrapper(func):
    @wraps(func)
    def _wrapper(set_xml, use_dof, n_blocks, goal_space, xml_file, geofence,
                 env_args: dict, **kwargs):
        xml_filepath = get_xml_filepath(xml_file)
        if set_xml is None:
            set_xml = []
        site_size = ' '.join([str(geofence)] * 3)
        path = Path('worldbody', 'body[@name="goal"]', 'site[@name="goal"]',
                    'size')
        set_xml += [XMLSetter(path=f'./{path}', value=site_size)]
        with mutate_xml(
                changes=set_xml,
                dofs=use_dof,
                n_blocks=n_blocks,
                goal_space=goal_space,
                xml_filepath=xml_filepath) as temp_path:
            env_args.update(
                geofence=geofence,
                xml_file=temp_path,
                goal_space=goal_space,
            )

            return func(env_args=env_args, **kwargs)

    def new_function(wrapper_args, **kwargs):
        return _wrapper(**wrapper_args, **kwargs)

    return new_function


XMLSetter = namedtuple('XMLSetter', 'path value')


@contextmanager
def mutate_xml(changes: List[XMLSetter], dofs: List[str], goal_space: Box,
               n_blocks: int, xml_filepath: Path):
    def rel_to_abs(path: Path):
        return Path(xml_filepath.parent, path)

    def mutate_tree(tree: ET.ElementTree):

        worldbody = tree.getroot().find("./worldbody")
        rgba = [
            "0 1 0 1",
            "0 0 1 1",
            "0 1 1 1",
            "1 0 0 1",
            "1 0 1 1",
            "1 1 0 1",
            "1 1 1 1",
        ]

        if worldbody:
            for i in range(n_blocks):
                pos = ' '.join(map(str, goal_space.sample()))
                name = f'block{i}'

                body = ET.SubElement(
                    worldbody, 'body', attrib=dict(name=name, pos=pos))
                ET.SubElement(
                    body,
                    'geom',
                    attrib=dict(
                        name=name,
                        type='box',
                        mass='1',
                        size=".05 .025 .017",
                        rgba=rgba[i],
                        condim='6',
                        solimp="0.99 0.99 "
                        "0.01",
                        solref='0.01 1'))
                ET.SubElement(
                    body, 'freejoint', attrib=dict(name=f'block{i}joint'))

        for change in changes:
            parent = re.sub('/[^/]*$', '', change.path)
            element_to_change = tree.find(parent)
            if isinstance(element_to_change, ET.Element):
                print('setting', change.path, 'to', change.value)
                name = re.search('[^/]*$', change.path)[0]
                element_to_change.set(name, change.value)

        for actuators in tree.iter('actuator'):
            for actuator in list(actuators):
                if actuator.get('joint') not in dofs:
                    print('removing', actuator.get('name'))
                    actuators.remove(actuator)
        for body in tree.iter('body'):
            for joint in body.findall('joint'):
                if not joint.get('name') in dofs:
                    print('removing', joint.get('name'))
                    body.remove(joint)

        parent = Path(temp[xml_filepath].name).parent

        for include_elt in tree.findall('*/include'):
            original_abs_path = rel_to_abs(include_elt.get('file'))
            tmp_abs_path = Path(temp[original_abs_path].name)
            include_elt.set('file', str(tmp_abs_path.relative_to(parent)))

        for compiler in tree.findall('compiler'):
            abs_path = rel_to_abs(compiler.get('meshdir'))
            compiler.set('meshdir', str(abs_path))

        return tree

    included_files = [
        rel_to_abs(e.get('file'))
        for e in ET.parse(xml_filepath).findall('*/include')
    ]

    temp = {
        path: tempfile.NamedTemporaryFile(suffix='.xml')
        for path in (included_files + [xml_filepath])
    }
    try:
        for path, f in temp.items():
            tree = ET.parse(path)
            mutate_tree(tree)
            tree.write(f)
            f.flush()

        yield Path(temp[xml_filepath].name)
    finally:
        for f in temp.values():
            f.close()
