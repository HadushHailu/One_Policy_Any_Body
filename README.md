# One Policy, Any Embodiment

### Zero-Shot Cross-Embodiment Transfer via Morphology-Conditioned Diffusion

> **OPAB:** A single diffusion policy, conditioned on a lightweight morphology descriptor extracted from URDF, transfers zero-shot to unseen robot embodiments — and DPO-based preference alignment makes it safe without reward engineering.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.1+-green.svg)](https://mujoco.org/)

---

## Simulation Environment

The simulation environment is built in MuJoCo and supports 5 robot embodiments across 7 manipulation tasks, with full domain randomization and multi-view camera observations.

### Robot Embodiments

| Robot | DOF | Gripper | Description |
|-------|:---:|---------|-------------|
| **Franka Panda** | 7 | Franka Hand | Research-grade cobot |
| **UR5** | 6 | Robotiq 2F-85 | Industrial manipulator |
| **WidowX** | 6 | WidowX Gripper | Low-cost desktop arm |
| **UFACTORY Lite6** | 6 | Lite6 Gripper | Lightweight cobot |
| **SO-101** | 5 | SO-101 Gripper | Ultra low-cost (~$300) |

### Manipulation Tasks

| Task | Description |
|------|-------------|
| **Pick & Place** | Grasp a cube and place it at a target location |
| **Push** | Slide a cube across the table to a target |
| **Stack** | Pick up one cube and stack it on top of another |
| **Peg Insertion** | Grasp a peg and insert it into a hole |
| **Drawer Open** | Grasp the drawer handle and pull it open |
| **Turn Faucet** | Grasp the faucet lever and rotate it |
| **Door Open** | Grasp the door handle and swing it open |

### Domain Randomization

Each task has a randomization budget that varies object poses, orientations, and initial conditions across episodes:

- **Object position randomization** — cubes, pegs, holes, handles placed within workspace-specific ranges
- **Object rotation** — random yaw for graspable objects (pick & place, stack)
- **Articulated object state** — initial joint angles for doors, faucets, drawers
- **Physics randomization** — cube mass, friction, joint damping, gravity perturbation
- **Visual randomization** — table/object colors, lighting conditions
- **Camera perturbation** — slight position and angle noise on camera views

### Camera Views

Each environment provides 3 camera observations:

| Camera | Description |
|--------|-------------|
| **Side View** | Angled perspective showing full workspace and arm motion |
| **Agent View** | Third-person view from behind the robot |
| **Top-Down** | Overhead view for spatial reasoning |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/one_policy_any_body.git
cd one_policy_any_body

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

# Verify
python -c "import opab; print('OPAB ready')"
```

---

## Training & Policy

🚧 **Coming soon** — morphology-conditioned diffusion policy training and zero-shot evaluation.

---

## Citation

```bibtex
@article{opab2026,
  title={One Policy, Any Embodiment: Zero-Shot Cross-Embodiment Transfer via Morphology-Conditioned Diffusion},
  author={TODO},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
