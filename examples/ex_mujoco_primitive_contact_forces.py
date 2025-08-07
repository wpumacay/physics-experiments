import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer as mjviewer
import numpy as np

CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = CURRENT_DIR.parent / "assets"
EMPTY_SCENE_PATH = ASSETS_DIR / "empty_scene.xml"

DIRTY_RESET = False
DIRTY_PAUSE = False

def key_callback(keycode: int) -> None:
    global DIRTY_RESET
    breakpoint()
    keychr = chr(keycode)
    if keychr == "P":
        DIRTY_RESET = True


def main() -> int:
    global DIRTY_RESET

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
    max_force_magnitude = -np.inf

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
                mj.mj_resetData(model, data)

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
            max_force_magnitude = force_magnitude if force_magnitude > max_force_magnitude else max_force_magnitude

            viewer.sync()
            time.sleep(0.001)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
