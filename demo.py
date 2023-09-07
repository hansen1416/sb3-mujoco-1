import time
import mujoco
import mujoco.viewer
from transforms3d import quaternions
import numpy as np

q_left = quaternions.axangle2quat([0, 0, 1], np.pi / 2)
q_back = quaternions.axangle2quat([1, 0, 0], 0)
q_right = quaternions.axangle2quat([0, 0, 1], -np.pi / 2)
q_top = quaternions.axangle2quat([0, 1, 0], np.pi / 2)
q_bottom = quaternions.axangle2quat([0, 1, 0], -np.pi / 2)

"""
free joint, which doesn't limit the sphere's movement in any way 
and doesn't require an actuator to move the sphere:

contype is a bit mask that specifies the contact type of the geom, 
and conaffinity is a bit mask that specifies the contact affinity of the geom. 
Two geoms can collide only if the bitwise AND of their contype and conaffinity is nonzero1. 
For example, if you have two geoms with contype=“1” and conaffinity=“2”, 
they will not collide, because 1 & 2 = 0. But if you have two geoms with contype=“3” 
and conaffinity=“2”, they will collide, because 3 & 2 = 2.
"""


# https://github.com/deepmind/mujoco/blob/main/doc/XMLreference.rst

xml = """
<mujoco>
    <option gravity="0 0 -9.81">
      <flag contact="enable"/>
    </option>
    <worldbody>
        <light name="light" pos="0 -5 5"/>
        <body name="box" euler="0 0 0">
            <geom name="left" type="box" pos="5. 5.0 0" size=".1 5. 5." rgba="1 0 0 1" quat="{0} {1} {2} {3}"/>
            <geom name="back" type="box" pos="10. 0. 0" size=".1 5. 5." rgba="1 0 0 1" quat="{4} {5} {6} {7}"/>
            <geom name="right" type="box" pos="5. -5. 0" size=".1 5. 5." rgba="1 0 0 1" quat="{8} {9} {10} {11}"/>
            <geom name="top" type="box" pos="5 0 5." size=".1 5. 5." rgba="1 0 0 1" quat="{12} {13} {14} {15}"/>
            <geom name="bottom" type="box" pos="5 0 -5" size=".1 5. 5." rgba="1 0 0 1" quat="{16} {17} {18} {19}"/>
        </body>
        <body name="ball" euler="0 0 0">
            <joint name="ball_joint" type="free"/>
            <geom name="green_sphere" pos="0 0 0" size="0.2" mass="1" rgba="0 1 0 1" contype="1" conaffinity="1"/>
        </body>
    </worldbody>
    <actuator>
        <general joint="ball_joint" dyntype="none" dynprm="1" forcerange="-1 1" forcelimited="true"/>
    </actuator>
</mujoco>
""".format(*q_left, *q_back, *q_right, *q_top, *q_bottom)


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
    viewer.cam.distance = 30  # distance from the target
    viewer.cam.elevation = 0  # elevation angle
    viewer.cam.azimuth = 0  # azimuth angle

    # model.set_velocity("green_sphere", linear=[1, 0, 0], angular=[0, 0, 0])

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 50:
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
