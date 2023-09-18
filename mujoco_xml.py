tall = 1.6

left_shouder = -0.4
right_shouder = 0.4

target_x = -1.1
target_y = 0.52
target_z = 1.4

upper_arm_l = 0.4
lower_arm_l = 0.4
hand_l = 0.14

skin_color = '0.8 0.6 0.4 1'

string_length = 0.4
anchor_z = target_z + string_length

arm_xml = """
<mujoco>
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
        rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
    </asset>
    <worldbody>
        
        <light pos="0 0 2."/>
        
        <geom size="1 1 .01" type="plane" material="grid"/>
        
        <site name="anchor" pos="{target_x} {target_y} {anchor_z}" size=".01"/>

        <body pos="0 0 0">
            <geom name="spine" type="box" fromto="0 0 0. 0 0 {tall}" size=".1" 
                pos="0 0 0" rgba="1 1 1 1"/>
            <geom name="shoulder" type="box" fromto="0 {left_shouder} {tall} 0 {right_shouder} {tall}" size=".1" 
                pos="0 0 0" rgba="1 1 1 1"/>

            <!-- Shoulder joint -->
            <body pos="0. 0.55 {tall}" gravcomp="1">
                <joint name="shoulder" type="ball" pos="0 0 0"/>
                <geom name="upper_arm" type="cylinder" fromto="0 0. 0. 0 0. -{upper_arm_l}" size="0.1" 
                rgba="{skin_color}"/>

                <!-- Elbow joint -->
                <body pos="0. 0. -0.6" gravcomp="1">
                    <joint name="elbow" type="hinge" pos="0 0 0.1" axis="0 1 0"/>
                    <geom name="lower_arm" type="cylinder"  fromto="0 0 0. 0 0 -{lower_arm_l}" size=".08" 
                    pos="0 0 0" rgba="{skin_color}"/>

                    <!-- Hand -->
                    <body pos="0. 0. -{lower_arm_l}" gravcomp="1">
                        <geom name="hand" type="cylinder" fromto="0 0 0.0 0 0 -{hand_l}" size="0.09" 
                        pos="0 0 0" rgba="{skin_color}"/>
                    </body>
                </body>
            </body>
        </body>

        <body name="target" pos="{target_x} {target_y} {target_z}">
            <joint name="free" type="free"/>
            <geom name="ball" type="sphere" size=".06" rgba="1 0 0 1"/>
            <site name="hook" pos="0. 0. 0." size=".01"/>
        </body>

    </worldbody>

    <tendon>
        <spatial name="wire" limited="true" range="0 {string_length}" width="0.003" stiffness="0.5" damping="0.5">
        <site site="anchor"/>
        <site site="hook"/>
        </spatial>
    </tendon>
</mujoco>
""".format(tall=tall, left_shouder=left_shouder, right_shouder=right_shouder,
           upper_arm_l=upper_arm_l, lower_arm_l=lower_arm_l, hand_l=hand_l, skin_color=skin_color,
           target_x=target_x, target_y=target_y, target_z=target_z,
           anchor_z=anchor_z, string_length=string_length)
