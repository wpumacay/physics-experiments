import time
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import mujoco as mj
import mujoco.viewer as mjviewer
import numpy as np

CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = CURRENT_DIR.parent / "assets"
EMPTY_SCENE_PATH = ASSETS_DIR / "empty_scene.xml"

DIRTY_RESET = False
DIRTY_PAUSE = False
DIRTY_TRACK = False
DIRTY_SAVE_PLOT = False


def key_callback(keycode: int) -> None:
    global DIRTY_RESET, DIRTY_PAUSE, DIRTY_TRACK, DIRTY_SAVE_PLOT
    keychr = chr(keycode)
    if keychr == "P":
        DIRTY_RESET = True
    elif keycode == 32:
        DIRTY_PAUSE = not DIRTY_PAUSE
        msg = "paused" if DIRTY_PAUSE else "running"
        print(f"Simulation: {msg}")
    elif keychr == "O":
        DIRTY_TRACK = not DIRTY_TRACK
        msg = "yes" if DIRTY_TRACK else "no"
        print(f"Tracking: {msg}")
    elif keychr == "L":
        DIRTY_SAVE_PLOT = True


def main() -> int:
    global DIRTY_RESET, DIRTY_PAUSE, DIRTY_TRACK, DIRTY_SAVE_PLOT

    spec = mj.MjSpec.from_file(EMPTY_SCENE_PATH.as_posix())

    body_spec = spec.worldbody.add_body(
        name="primmitive",
        pos=[0.0, 0.0, 1.0],
    )
    body_spec.add_geom(
        name="primitive-collider",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.1, 0.1, 0.1],
        rgba=[0.8, 0.1, 0.1, 1.0],
    )
    body_spec.add_freejoint()

    model = spec.compile()
    data = mj.MjData(model)

    collider_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "primitive-collider")
    force_magnitude = 0.0
    forces_history = []

    plt.ion()

    with mjviewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        while viewer.is_running():
            if DIRTY_RESET:
                DIRTY_RESET = False
                print("Reset simulation")
                DIRTY_TRACK = False
                print("Disabled tracking")
                mj.mj_resetData(model, data)
                mj.mj_forward(model, data)

            if DIRTY_SAVE_PLOT:
                DIRTY_SAVE_PLOT = False
                plt.grid(True)
                plt.plot(forces_history)
                plt.xlabel("Step")
                plt.ylabel("Force (N)")
                plt.savefig("force_history.png")
                forces_history = []
                print("Saved plot")

            if not DIRTY_PAUSE:
                mj.mj_step(model, data)
                net_force = np.zeros(3, dtype=float)
                for contact_id in range(data.ncon):
                    geom_id_0 = data.contact[contact_id].geom[0]
                    geom_id_1 = data.contact[contact_id].geom[1]
                    if geom_id_0 == collider_geom_id or geom_id_1 == collider_geom_id:
                        force_torque_buff = np.zeros(6, dtype=float)
                        mj.mj_contactForce(model, data, contact_id, force_torque_buff)
                        R = data.contact[contact_id].frame.reshape(3, 3)
                        force_w = R.T @ force_torque_buff[:3]
                        net_force += force_w
                force_magnitude = np.linalg.norm(net_force)
                if DIRTY_TRACK:
                    forces_history.append(force_magnitude)

            viewer.sync()
            time.sleep(0.001)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
