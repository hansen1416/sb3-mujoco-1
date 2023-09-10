import math
import time

import numpy as np
import mujoco
import mujoco.viewer
from transforms3d import quaternions

from lib.BaseRender import BaseRender


xml = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300" mark="none"/>
    <material name="grid" texture="grid" texrepeat="1 1"
     texuniform="true" reflectance=".2"/>
  </asset>

  <worldbody>
    <light name="light" pos="0 0 1"/>

    <geom name="floor" type="plane" pos="0 0 -.5" size="2 2 .1" material="grid"/>
    <site name="anchor" pos="0 0 .3" size=".01"/>

    <!-- <geom name="pole" type="cylinder" fromto=".3 0 -.5 .3 0 -.1" size=".04"/>
    <body name="bat" pos=".3 0 -.1">
      <joint name="swing" type="hinge" damping="1" axis="0 0 1"/>
      <geom name="bat" type="capsule" fromto="0 0 .04 0 -.3 .04"
       size=".04" rgba="0 0 1 1"/>
    </body> -->

    <body name="box_and_sphere" pos="0 0 0">
      <joint name="free" type="free"/>
      <geom name="red_sphere" type="sphere" size=".06" rgba="1 0 0 1"/>
      <!-- <geom name="green_sphere"  size=".06" pos=".1 .1 .1" rgba="0 1 0 1"/> -->
      <site name="hook" pos="0. 0. 0." size=".01"/>
      <!-- <site name="IMU"/> -->
    </body>
  </worldbody>

  <tendon>
    <spatial name="wire" limited="true" range="0 0.4" width="0.003">
      <site site="anchor"/>
      <site site="hook"/>
    </spatial>
  </tendon>

  <!-- <actuator>
    <motor name="my_motor" joint="swing" gear="1"/>
  </actuator> -->

  <!-- <sensor>
    <accelerometer name="accelerometer" site="IMU"/>
  </sensor> -->
</mujoco>
"""

if __name__ == "__main__":

    render = BaseRender(xml)

    render.run()
