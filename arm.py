from lib.BaseRender import BaseRender


xml = """
<mujoco>
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
        rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
    </asset>
    <worldbody>
        <light pos="0 0 .6"/>
        <geom size="1 1 .01" type="plane" material="grid"/>
        <!-- Shoulder joint -->
        <body pos="0 0 0" gravcomp="1">
            <joint name="shoulder" type="ball"/>
            <geom name="upper_arm" type="cylinder" fromto="0 0 0 0 0 .4" size=".1" rgba="0.8 0.6 0.4 1"/>
            
            <!-- Elbow joint -->
            <body pos="0. 0. 0" gravcomp="1">
                <joint name="elbow" type="hinge" pos="0 0 0" axis="1 0 0"/>
                <geom name="lower_arm" type="cylinder"  fromto="0 0 0.4 0 0 .7" size=".08" 
                pos="0 0 0" rgba="0.8 0.6 0.4 1"/>
                
                <!-- Wrist joint -->
                <body pos="0. 0. 0" gravcomp="1">
                    <joint name="wrist" type="hinge" pos="0 0 0" axis="1 0 0"/>
                    <geom name="hand" type="cylinder" fromto="0 0 0.7 0 0 .9" 
                    size="0.09" pos="0 0 0" rgba="0.8 0.6 0.4 1"/>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>
"""


def init_callback(mode, data):
    print(data.qpos)
    print(data.qvel)

    data.qpos[4] = 0.1


def step_callback(mode, data):
    pass


if __name__ == "__main__":
    rnder = BaseRender(xml)

    rnder.run(init_callback=init_callback,
              step_callback=step_callback)
