import math
import time
import os

import numpy as np
import mujoco
import mujoco.viewer
from transforms3d import quaternions
import matplotlib.pyplot as plt
# from dm_control.mujoco.wrapper.mjbindings import mjlib

from mujoco_xml import arm_xml

def plot_force(times, forces):
    lines = plt.plot(times, forces)
    plt.title('contact force')
    plt.ylabel('Newton')
    plt.legend(iter(lines), ('normal z', 'friction x', 'friction y'))

    plt.savefig(os.path.join('img', 'forces.png'))
    plt.clf()  # clear the figure
    plt.close()  # close the window and release the memory


def plot_normal_force(times, forces):
    plt.plot(times, forces)
    plt.title('normal (z) force - log scale')
    plt.ylabel('Newton')
    plt.yscale('log')
    plt.xlabel('second')

    plt.savefig(os.path.join('img', 'normal_force.png'))
    plt.clf()  # clear the figure
    plt.close()  # close the window and release the memory


def plot_penetration(times, penetration):
    plt.plot(times, penetration)
    plt.title('penetration depth')
    plt.ylabel('millimeter')
    plt.xlabel('second')

    plt.savefig(os.path.join('img', 'penetration.png'))
    plt.clf()  # clear the figure
    plt.close()  # close the window and release the memory


def plot_acceleration(times, acceleration):
    plt.plot(times, acceleration)
    plt.title('acceleration')
    plt.ylabel('(meter,radian)/s/s')

    plt.savefig(os.path.join('img', 'acceleration.png'))
    plt.clf()  # clear the figure
    plt.close()  # close the window and release the memory


def plot_velocity(times, velocity):
    plt.plot(times, velocity)
    plt.title('velocity')
    plt.ylabel('(meter,radian)/s')
    plt.xlabel('second')

    plt.savefig(os.path.join('img', 'velocity.png'))
    plt.clf()  # clear the figure
    plt.close()  # close the window and release the memory

def plot_ncon(times, ncon):
    plt.plot(times, ncon)
    plt.title('number of contacts')
    plt.ylabel('count')
    plt.yticks(range(6))

    plt.savefig(os.path.join('img', 'ncon.png'))
    plt.clf()  # clear the figure
    plt.close()  # close the window and release the memory


class ArmSim:

    def __init__(self, xml) -> None:
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # enable joint visualization option:
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        self.model.opt.timestep = 0.002

    def run_with_viewer(self):
            
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # todo this is linear speed, add acceleration
        shoulder_angle = np.linspace(10, 86, 100)
        elbow_angle = np.linspace(110, 0, 100)
        motion_idx = 0
        direction = 1

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:

            # set camera
            viewer.cam.lookat[0] = 0  # x position
            viewer.cam.lookat[1] = 0  # y position
            viewer.cam.lookat[2] = 0  # z position
            viewer.cam.distance = 5  # distance from the target
            viewer.cam.elevation = -30  # elevation angle
            viewer.cam.azimuth = 0  # azimuth angle

            # random initial rotational velocity:
            mujoco.mj_resetData(self.model, self.data)

            # print(self.data.qpos)
            # print(self.data.qvel)

            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()

            while viewer.is_running() and time.time() - start < 5:

                step_start = time.time()

                mujoco.mj_step(self.model, self.data)

                # joint manipulation start

                q = quaternions.axangle2quat(
                    [0, 1, 0], math.radians(shoulder_angle[motion_idx]), is_normalized=True)

                self.data.qpos[0] = q[0]  # w
                self.data.qpos[1] = q[1]  # x
                self.data.qpos[2] = q[2]  # y
                self.data.qpos[3] = q[3]  # z

                self.data.qpos[4] = math.radians(elbow_angle[motion_idx])  # w
                self.data.qvel[3] = 0
                # print(data.qvel)

                if direction == 1:
                    motion_idx += 1
                else:
                    motion_idx -= 1

                if motion_idx >= len(elbow_angle) - 1:
                    direction *= -1
                elif motion_idx <= 0:
                    direction *= -1

                # joint manipulation end

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.model.opt.timestep - \
                    (time.time() - step_start)

                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def run(self, n_steps=100):

        # random initial rotational velocity:
        mujoco.mj_resetData(self.model, self.data)

        # todo this is linear speed, add acceleration
        shoulder_angle = np.linspace(10, 86, 100)
        elbow_angle = np.linspace(110, 0, 100)
        motion_idx = 0
        direction = 1

        # print(self.data.qpos)
        # print(self.data.qvel)

        sim_time = np.zeros(n_steps)
        forcetorque = np.zeros(6)

        forces = np.zeros((n_steps, 3))
        penetration = np.zeros(n_steps)
        

        ncon = np.zeros(n_steps)
        velocity = np.zeros((n_steps, self.model.nv))
        acceleration = np.zeros((n_steps, self.model.nv))

        # print(self.model.geom('upper_arm').id)
        # print(dir(self.data))

        # Close the viewer automatically after 30 wall-seconds.
        for i in range(n_steps):

            sim_time[i] = self.data.time

            mujoco.mj_step(self.model, self.data)

            # joint manipulation start
            q = quaternions.axangle2quat(
                [0, 1, 0], math.radians(shoulder_angle[motion_idx]), is_normalized=True)

            self.data.qpos[0] = q[0]  # w
            self.data.qpos[1] = q[1]  # x
            self.data.qpos[2] = q[2]  # y
            self.data.qpos[3] = q[3]  # z

            self.data.qpos[4] = math.radians(elbow_angle[motion_idx])  # w
            self.data.qvel[3] = 0
            # print(data.qvel)

            if direction == 1:
                motion_idx += 1
            else:
                motion_idx -= 1

            if motion_idx >= len(elbow_angle) - 1:
                direction *= -1
            elif motion_idx <= 0:
                direction *= -1
            # joint manipulation end

            ncon[i] = self.data.ncon
            velocity[i] = self.data.qvel[:]
            acceleration[i] = self.data.qacc[:]

            # todo, geom views are not changing? use joint info for observations
            # the following are observation
            # cartesian position of upper arm, lower arm, hand and ball
            # quaternion of upper arm, lower arm
            # reward is 
            # 

            print('--------------- acc')
            print(self.data.qacc[4:])

            upper_arm_p = self.data.geom_xpos[self.model.geom('upper_arm').id]
            lower_arm_p = self.data.geom_xpos[self.model.geom('lower_arm').id]
            hand_p = self.data.geom_xpos[self.model.geom('hand').id]
            ball_p = self.data.geom_xpos[self.model.geom('ball').id]

            upper_arm_r = quaternions.mat2quat(self.data.geom_xmat[self.model.geom('upper_arm').id])
            lower_arm_r = quaternions.mat2quat(self.data.geom_xmat[self.model.geom('lower_arm').id])

            # concatenate numpy arrays
            observation = np.concatenate((upper_arm_p, lower_arm_p, hand_p, ball_p, upper_arm_r, lower_arm_r), axis=None)

            print('--------------- observation')
            print(observation)

            # print(observation)
            # print(dir(self.data))
            # print(self.data.qpos.shape) # 12
            # print(self.data.qvel.shape) # 10
            # print(self.data.qacc.shape) # 10
            # print(self.model.geom('upper_arm').pos)
            # print(self.model.geom('lower_arm').quat)
            # print(self.model.geom('hand'))
            # print(self.model.geom('ball'))

            
            """
            <MjContact
            H: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0.])
            dim: 3
            dist: -0.0020774606072078428
            efc_address: 1
            exclude: 0
            frame: array([ 0.92154504,  0.11084697,  0.37211248, -0.10278388,  0.99383749,
                -0.0415033 , -0.37441985,  0.        ,  0.92725928])
            friction: array([1.e+00, 1.e+00, 5.e-03, 1.e-04, 1.e-04])
            geom1: 6
            geom2: 5
            includemargin: 0.0
            mu: 1.0
            pos: array([-1.13065943,  0.52141824,  1.42430136])
            solimp: array([9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e+00])
            solref: array([0.02, 1.  ])
            solreffriction: array([0., 0.])
            >

            pos: the position of the contact point in global coordinates2.

            solimp: the solver parameters for the normal direction of the contact2. 
            These are three numbers that control the stiffness, damping and friction of the contact3.

            solref: the reference values for the normal direction of the contact2. 
            These are two numbers that specify the minimum and maximum normal force that can be applied at the contact3.

            solreffriction: the reference values for the tangential direction of the contact2. 
            These are two numbers that specify the minimum and maximum friction force that can be applied at the contact3
            """

            if len(self.data.contact):
                for j, c in enumerate(self.data.contact):

                    mujoco.mj_contactForce(
                        self.model, self.data, j, forcetorque)

                    forces[i] += forcetorque[0:3]
                    penetration[i] = min(penetration[i], c.dist)

                    # print('name of geom1: ', self.model.geom(c.geom1).name)
                    # print('name of geom2: ', self.model.geom(c.geom2).name)

        # print(forces)
        # print(penetration)

        plot_force(sim_time, forces)
        plot_normal_force(sim_time, forces[:, 0])
        plot_penetration(sim_time, penetration)
        plot_acceleration(sim_time, acceleration)
        plot_velocity(sim_time, velocity)
        plot_ncon(sim_time, ncon)


if __name__ == "__main__":

    if not os.path.exists('img'):
        # create dir img
        os.mkdir('img')

    rnder = ArmSim(arm_xml)

    rnder.run(10)
    # rnder.run_with_viewer()
