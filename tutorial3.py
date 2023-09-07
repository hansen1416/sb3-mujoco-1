import time
import itertools
import numpy as np
import mujoco
import mujoco.viewer


chaotic_pendulum = """
<mujoco>
  <option timestep=".001">
    <flag energy="enable" contact="disable"/>
  </option>

  <default>
    <joint type="hinge" axis="0 -1 0"/>
    <geom type="capsule" size=".02"/>
  </default>

  <worldbody>
    <light pos="0 -.4 1"/>
    <camera name="fixed" pos="0 -1 0" xyaxes="1 0 0 0 0 1"/>
    <body name="0" pos="0 0 .2">
      <joint name="root"/>
      <geom fromto="-.2 0 0 .2 0 0" rgba="1 1 0 1"/>
      <geom fromto="0 0 0 0 0 -.25" rgba="1 1 0 1"/>
      <body name="1" pos="-.2 0 0">
        <joint/>
        <geom fromto="0 0 0 0 0 -.2" rgba="1 0 0 1"/>
      </body>
      <body name="2" pos=".2 0 0">
        <joint/>
        <geom fromto="0 0 0 0 0 -.2" rgba="0 1 0 1"/>
      </body>
      <body name="3" pos="0 0 -.25">
        <joint/>
        <geom fromto="0 0 0 0 0 -.2" rgba="0 0 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(chaotic_pendulum)
data = mujoco.MjData(model)


# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True


mujoco.mj_kinematics(model, data)
mujoco.mj_forward(model, data)

PERTURBATION = 1e-7
SIM_DURATION = 10  # seconds
NUM_REPEATS = 8


with mujoco.viewer.launch_passive(model, data) as viewer:

    # set camera
    viewer.cam.lookat[0] = 0  # x position
    viewer.cam.lookat[1] = 0  # y position
    viewer.cam.lookat[2] = 0  # z position
    viewer.cam.distance = 3  # distance from the target
    viewer.cam.elevation = 0  # elevation angle
    viewer.cam.azimuth = 90  # azimuth angle

    mujoco.mj_resetData(model, data)
    # set initial state
    data.joint('root').qvel = 10
    # perturb initial velocities
    data.qvel[:] += PERTURBATION * np.random.randn(model.nv)

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()

    while viewer.is_running() and time.time() - start < 20:

        step_start = time.time()

        mujoco.mj_step(model, data)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
