import time
import mujoco
import mujoco.viewer


class BaseRender:

    def __init__(self, xml) -> None:
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # enable joint visualization option:
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    def run(self, init_callback=None, step_callback=None):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:

            # set camera
            viewer.cam.lookat[0] = 0  # x position
            viewer.cam.lookat[1] = 0  # y position
            viewer.cam.lookat[2] = 0  # z position
            viewer.cam.distance = 4  # distance from the target
            viewer.cam.elevation = -30  # elevation angle
            viewer.cam.azimuth = 0  # azimuth angle

            # random initial rotational velocity:
            mujoco.mj_resetData(self.model, self.data)

            if callable(init_callback):
                init_callback(self.model, self.data)

            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()

            while viewer.is_running() and time.time() - start < 6:

                step_start = time.time()

                mujoco.mj_step(self.model, self.data)

                if callable(step_callback):
                    step_callback(self.model, self.data)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.model.opt.timestep - \
                    (time.time() - step_start)

                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
