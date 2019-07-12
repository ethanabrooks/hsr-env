Install `mujoco-py` from source: https://github.com/openai/mujoco-py

```
pip install -e .
python3 hsr/control.py --block-space "(-0.16,0.16)(-.24,.24)(0.422,0.422)(0,1)(0,0)(0,0)(0,1)" --steps-per-action=300 --geofence=.5 --goal-space "(0,0)(0,0)(.422,.422)"  --use-dof arm_flex_joint --use-dof hand_l_proximal_joint --use-dof hand_r_proximal_joint --use-dof wrist_flex_joint --use-dof arm_roll_joint --use-dof wrist_roll_joint --use-dof slide_x --use-dof slide_y --render --n-blocks=4


```
