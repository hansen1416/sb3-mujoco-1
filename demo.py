import time
import mujoco
import mujoco.viewer
from transforms3d import quaternions
import numpy as np

MJCF = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300" mark="none"/>
    <material name="grid" texture="grid" texrepeat="1 1"
     texuniform="true" reflectance=".2"/>
  </asset>

  <worldbody>
    <light name="light" pos="0 0 2"/>

    <geom name="pole" type="cylinder" fromto=".3 0 -.5 .3 0 -.1" size=".04"/>
    <body name="bat" pos=".3 0 -.1">
      <joint name="swing" type="hinge" damping="1" axis="0 0 1"/>
      <geom name="bat" type="capsule" fromto="0 0 .04 0 -.3 .04"
       size=".04" rgba="0 0 1 1"/>
    </body>

  </worldbody>

</mujoco>
"""

model = mujoco.MjModel.from_xml_string(MJCF)
data = mujoco.MjData(model)


# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True


mujoco.mj_kinematics(model, data)
mujoco.mj_forward(model, data)


# print(model.ngeom)
print(data.qvel)
print(data.qpos)

with mujoco.viewer.launch_passive(model, data) as viewer:

    # set camera
    viewer.cam.lookat[0] = 0  # x position
    viewer.cam.lookat[1] = 0  # y position
    viewer.cam.lookat[2] = 0  # z position
    viewer.cam.distance = 3  # distance from the target
    viewer.cam.elevation = 0  # elevation angle
    viewer.cam.azimuth = 0  # azimuth angle

    # random initial rotational velocity:
    mujoco.mj_resetData(model, data)
    # data.qvel[3:6] = 5*np.random.randn(3)

    # data.qvel[0] = 10
    # data.qpos[0] = 10

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()

    while viewer.is_running() and time.time() - start < 6:

        step_start = time.time()

        # data.qvel[0] = 10

        mujoco.mj_step(model, data)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
