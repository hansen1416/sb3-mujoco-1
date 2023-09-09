from lib.BaseRender import BaseRender


ball_joint = """
<mujoco>
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
        rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
    </asset>
    <worldbody>
        <light pos="0 0 .6"/>
        <geom size=".3 .3 .01" type="plane" material="grid"/>
        <body name="bat" pos="0 0 0">
            <joint name="ball" type="ball" pos="0 0 0"/>
            <geom name="bat" type="capsule" fromto="0 0 .04 0 -.3 .04"
            size=".04" rgba="1 1 1 1"/>
        </body>
    </worldbody>


    <actuator>
        <velocity name="my_motor1" joint="ball" gear="1 1 1"/>
    </actuator>
</mujoco>
"""


def init_callback(mode, data):
    # The slide type creates a sliding or prismatic joint with one translational degree of freedom.
    # data.qpos[0] = math.pi/4

    print(data.qpos)  # [0.]
    print(data.qvel)  # [0.]

    print(data.ctrl)


def step_callback(mode, data):
    # along the axis of the slide joint
    # data.qvel[0] = 0

    data.ctrl[0] = 300
    # data.ctrl[1] = 10
    # data.ctrl[2] = 10
    # data.qvel[2] = 1


if __name__ == "__main__":
    rnder = BaseRender(ball_joint)

    rnder.run(init_callback=init_callback,
              step_callback=step_callback)
