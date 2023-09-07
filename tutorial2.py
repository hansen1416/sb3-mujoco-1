
import time
import itertools
import numpy as np
import mujoco
import mujoco.viewer

tippe_top = """
<mujoco model="tippe top">
  <option integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="top" pos="0 0 .02">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" />
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
       contype="0" conaffinity="0" group="3"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(tippe_top)
data = mujoco.MjData(model)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True


print('positions', data.qpos)
print('velocities', data.qvel)

mujoco.mj_kinematics(model, data)
mujoco.mj_forward(model, data)

timevals = []
angular_velocity = []
stem_height = []


with mujoco.viewer.launch_passive(model, data) as viewer:

    # set camera
    viewer.cam.lookat[0] = 0  # x position
    viewer.cam.lookat[1] = 0  # y position
    viewer.cam.lookat[2] = 0  # z position
    viewer.cam.distance = 0.2  # distance from the target
    viewer.cam.elevation = -30  # elevation angle
    viewer.cam.azimuth = 0  # azimuth angle

    mujoco.mj_resetData(model, data)

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()

    while viewer.is_running() and time.time() - start < 2:

        step_start = time.time()

        mujoco.mj_step(model, data)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        timevals.append(data.time)
        angular_velocity.append(data.qvel[3:6].copy())
        stem_height.append(data.geom_xpos[2, 2])

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


print(np.array(timevals))
print(np.array(angular_velocity))
print(np.array(stem_height))
