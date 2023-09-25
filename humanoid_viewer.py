import time
import os

import numpy as np
import mujoco
import mujoco.viewer

# model_path = os.path.join('assets', 'humanoidstandup.xml')
model_path = os.path.join('assets', 'humanoid.xml')

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)


# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True


mujoco.mj_kinematics(model, data)
mujoco.mj_forward(model, data)

PERTURBATION = 1e-7
SIM_DURATION = 10  # seconds
NUM_REPEATS = 8

# Close the viewer automatically after 30 wall-seconds.
start = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:

    # set camera
    viewer.cam.lookat[0] = 0  # x position
    viewer.cam.lookat[1] = 0  # y position
    viewer.cam.lookat[2] = 0  # z position
    viewer.cam.distance = 3  # distance from the target
    viewer.cam.elevation = 0  # elevation angle
    viewer.cam.azimuth = 90  # azimuth angle

    mujoco.mj_resetData(model, data)

    step_start = time.time()

    while viewer.is_running() and time.time() - start < 20:

        step_start = time.time()

        mujoco.mj_step(model, data)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
