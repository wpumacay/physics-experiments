import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import mujoco as mj
import mujoco.viewer as mjviewer
import numpy as np
import tyro

CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = CURRENT_DIR.parent / "assets"
EMPTY_SCENE_PATH = ASSETS_DIR / "empty_scene.xml"

DIRTY_RESET = False
DIRTY_PAUSE = False
DIRTY_TRACK = False
DIRTY_SAVE_PLOT = False
DIRTY_SHOW_INFO = False

PRIMITIVE_BODY_NAME = "primitive"
PRIMITIVE_GEOM_NAME = "primitive-collider"


@dataclass
class Args:
    timestep: float = 0.002
    primitive: Literal["sphere", "box"] = "box"


def key_callback(keycode: int) -> None:
    global \
        DIRTY_RESET, \
        DIRTY_PAUSE, \
        DIRTY_TRACK, \
        DIRTY_SAVE_PLOT, \
        DIRTY_SHOW_INFO

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
    elif keychr == "Y":
        DIRTY_SHOW_INFO = True


def print_simulation_info(model: mj.MjModel) -> None:
    print("Simulation -----------------------------")
    print(f"timestep: {model.opt.timestep}")
    print(f"gravity: {model.opt.gravity}")
    print("Primitive ------------------------------")
    body_mass = model.body(PRIMITIVE_BODY_NAME).mass
    body_weight = abs(model.opt.gravity[2]) * body_mass
    print(f"mass: {body_mass}")
    print(f"weight: {body_weight}")


def create_primitive(spec: mj.MjSpec, ptype: Literal["sphere", "box"]) -> None:
    body_spec = spec.worldbody.add_body(
        name=PRIMITIVE_BODY_NAME, pos=[0.0, 0.0, 1.0]
    )
    if ptype == "sphere":
        body_spec.add_geom(
            name=PRIMITIVE_GEOM_NAME,
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.1, 0.1, 0.1],
            rgba=[0.8, 0.1, 0.1, 1.0],
        )
    elif ptype == "box":
        body_spec.add_geom(
            name=PRIMITIVE_GEOM_NAME,
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.1, 0.1, 0.1],
            rgba=[0.8, 0.1, 0.1, 1.0],
        )
    body_spec.add_freejoint()


def main() -> int:
    global \
        DIRTY_RESET, \
        DIRTY_PAUSE, \
        DIRTY_TRACK, \
        DIRTY_SAVE_PLOT, \
        DIRTY_SHOW_INFO

    args = tyro.cli(Args)

    spec = mj.MjSpec.from_file(EMPTY_SCENE_PATH.as_posix())
    spec.option.timestep = args.timestep

    create_primitive(spec, args.primitive)

    model: mj.MjModel = spec.compile()
    data: mj.MjData = mj.MjData(model)

    gravity = abs(model.opt.gravity[2])
    body_mass = (model.body(PRIMITIVE_BODY_NAME).mass * gravity).item()

    print_simulation_info(model)

    collider_geom_id = mj.mj_name2id(
        model, mj.mjtObj.mjOBJ_GEOM, "primitive-collider"
    )
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
                plt.title(f"Force @ {args.primitive} of mass: {body_mass}")
                plt.plot(forces_history)
                plt.xlabel("Step")
                plt.ylabel("Force (N)")
                plt.savefig(f"force_history_{args.primitive}.png")
                forces_history = []
                print("Saved plot")

            if DIRTY_SHOW_INFO:
                DIRTY_SHOW_INFO = False
                print_simulation_info(model)

            if not DIRTY_PAUSE:
                mj.mj_step(model, data)
                net_force = np.zeros(3, dtype=float)
                for contact_id in range(data.ncon):
                    geom_id_0 = data.contact[contact_id].geom[0]
                    geom_id_1 = data.contact[contact_id].geom[1]
                    if (
                        geom_id_0 == collider_geom_id
                        or geom_id_1 == collider_geom_id
                    ):
                        force_torque_buff = np.zeros(6, dtype=float)
                        mj.mj_contactForce(
                            model, data, contact_id, force_torque_buff
                        )
                        R = data.contact[contact_id].frame.reshape(3, 3)
                        force_w = R.T @ force_torque_buff[:3]
                        net_force += force_w
                force_magnitude = np.linalg.norm(net_force)
                if DIRTY_TRACK:
                    forces_history.append(force_magnitude)

                print(f"total force at primitive: {force_magnitude}")

            viewer.sync()
            time.sleep(0.001)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
