Install `mujoco-py` from source: https://github.com/openai/mujoco-py

```
pip install -e .
python hsr/control.py --block-space (0,0)(0,0)(0,0)(0,0) --steps-per-action=300 --geofence=.5 --goal-space (0,0)(0,0)(0,0) --use-dof slide_x --use-dof slide_y --render
```
