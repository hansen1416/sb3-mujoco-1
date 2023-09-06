import time
import mujoco
import mujoco.viewer
from transforms3d import quaternions
import numpy as np

r1 = quaternions.axangle2quat([1, 0, 0], 0)


print(r1)

# https://github.com/deepmind/mujoco/blob/main/doc/XMLreference.rst

xml = """
<mujoco>
  <option gravity="0 0 9.81"/>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" pos="0 0 0" size="10. .01 10." rgba="1 0 0 1" quat="{0} {1} {2} {3}"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
""".format(*r1)


model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)


# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True


print('Total number of DoFs in the model:', model.nv)
print('Generalized positions:', data.qpos)
print('Generalized velocities:', data.qvel)


with mujoco.viewer.launch_passive(model, data) as viewer:

    # set camera
    viewer.cam.lookat[0] = 0  # x position
    viewer.cam.lookat[1] = 0  # y position
    viewer.cam.lookat[2] = 0  # z position
    viewer.cam.distance = 60  # distance from the target
    viewer.cam.elevation = 0  # elevation angle
    viewer.cam.azimuth = 0  # azimuth angle

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 5:
        step_start = time.time()

        # is this line necessary?
        mujoco.mj_forward(model, data)

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
