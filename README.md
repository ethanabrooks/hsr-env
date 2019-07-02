Install `mujoco-py` from source: https://github.com/openai/mujoco-py

```
pip install -e .
python3 hsr/control.py --block-space "(0,0)(0,0)(0.418,0.418)(1,1)(0,0)(0,0)(0,0)" --steps-per-action=300 --geofence=.5 --goal-space "(0,0)(0,0)(.418,.418)"  --use-dof arm_flex_joint --use-dof hand_l_proximal_joint --use-dof hand_r_proximal_joint --use-dof wrist_flex_joint --use-dof arm_roll_joint --use-dof wrist_roll_joint --render --n-blocks=1

```
