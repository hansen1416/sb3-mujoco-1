import time
import mujoco
import mujoco.viewer


xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)


try:
    model.geom('green_sphere')

    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'green_sphere')

    model.geom_rgba[id, :]

    c = model.geom('green_sphere').rgba

    print('id of "green_sphere": ', model.geom('green_sphere').id)
    print('name of geom 1: ', model.geom(1).name)
    # the 0th body is always the world. It cannot be renamed.
    print('name of body 0: ', model.body(0).name)

    # The id and name attributes are useful in Python comprehensions:
    res = [model.geom(i).name for i in range(model.ngeom)]

    print(res)
    # contains data of all geom in the model
    data = mujoco.MjData(model)

    print(data.geom_xpos)
except KeyError as e:
    print(e)


m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

print('default gravity', model.opt.gravity)
print('default timestep', model.opt.timestep)

ct = 0

while data.time < 1:
    mujoco.mj_step(model, data)
    # print(data.geom_xpos)
    ct += 1

print('ct', ct)

# with mujoco.viewer.launch_passive(m, d) as viewer:
#     # Close the viewer automatically after 30 wall-seconds.
#     start = time.time()
#     while viewer.is_running() and time.time() - start < 30:
#         step_start = time.time()

#         # mj_step can be replaced with code that also evaluates
#         # a policy and applies a control signal before stepping the physics.
#         mujoco.mj_step(m, d)

#         # Example modification of a viewer option: toggle contact points every two seconds.
#         with viewer.lock():
#             viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
#                 d.time % 2)

#         # Pick up changes to the physics state, apply perturbations, update options from GUI.
#         viewer.sync()

#         # Rudimentary time keeping, will drift relative to wall clock.
#         time_until_next_step = m.opt.timestep - (time.time() - step_start)
#         if time_until_next_step > 0:
#             time.sleep(time_until_next_step)
