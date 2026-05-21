# Simulation Environment Research: Top Robotics Manipulation Papers

## Comprehensive Comparison of Setup Parameters

---

## 1. robosuite / robomimic (Stanford/UT Austin) — Manipulation Benchmark

### Environment Dimensions
| Parameter | Value |
|-----------|-------|
| **Table size (default)** | 0.8m × 0.8m × 0.05m (L×W×H) |
| **Table offset (height)** | z = 0.8m from ground |
| **Table friction** | (1.0, 0.005, 0.0001) — sliding, torsional, rolling |
| **Cube size** | 0.02m × 0.02m × 0.02m (edge = 2cm) |
| **Peg arena table** | 0.45m × 0.69m × 0.05m |
| **Bins arena** | 0.39m × 0.49m × 0.82m |
| **Wipe table** | 0.5m × 0.8m × 0.05m, offset (0.15, 0, 0.9) |

### Camera Setup
| Parameter | Value |
|-----------|-------|
| **Default resolution** | 256×256 (internal), **84×84** (for policy training/robomimic) |
| **Camera names** | `agentview`, `frontview`, `robot0_eye_in_hand` (wrist) |
| **agentview position** | ~(0.6, 0, 1.6) looking down at table |
| **frontview position** | (1.6, 0, 1.45) quat=(0.56, 0.43, 0.43, 0.56) |
| **Image crop (training)** | 76×76 from 84×84, or 216×216 from 240×240 (tool_hang) |

### Control & Episode
| Parameter | Value |
|-----------|-------|
| **Control frequency** | **20 Hz** |
| **Action space** | OSC_POSE: 7D (dx, dy, dz, droll, dpitch, dyaw, gripper) |
| **Episode horizon** | Lift/Can/Square: **400 steps** (20s); Transport/ToolHang: **700-1000 steps** |
| **Physics timestep** | 0.002s (500 Hz MuJoCo), with action repeat = 25 → 20 Hz control |

### Success Criteria
| Task | Criterion |
|------|-----------|
| **Lift** | Cube height > table_height + 0.04m |
| **Can** | Can placed in target bin |
| **Square** | Nut assembled on peg |
| **ToolHang** | Tool hanging on hook |
| **Door** | Door opened (handle turned + door angle) |

### Data Collection
| Parameter | Value |
|-----------|-------|
| **Proficient-Human (PH)** | ~200 demos per task (teleoperated) |
| **Multi-Human (MH)** | ~300 demos per task (multiple operators) |
| **Machine-Generated (MG)** | Scripted data collection via MimicGen |
| **Rollout horizon** | PH: 400 (Lift/Can/Square), 700 (Transport/ToolHang) |
| **Rollout horizon** | MH: 500 (Lift/Can/Square), 1100 (Transport) |

### Visual Style
- Clean, minimalist MuJoCo rendering
- Default table: gray/beige surface
- Robot: metallic silver (Panda), dark gray (Sawyer)
- Objects: colored (red cube, green can, etc.)
- Lighting: single overhead + ambient

---

## 2. ACT / ALOHA (Tony Zhao, Stanford) — Action Chunking with Transformers

### Environment Dimensions
| Parameter | Value |
|-----------|-------|
| **Workspace** | Bimanual ViperX setup, ~0.5m reach per arm |
| **Cube size** | Standard MuJoCo cube (~2cm edge) for transfer task |
| **Object randomization** | 15cm white reference line for position variation |

### Camera Setup
| Parameter | Value |
|-----------|-------|
| **Raw image resolution** | **480×640** (H×W) per camera |
| **Number of cameras** | 4 total: 2 stationary + 2 wrist-mounted |
| **Training camera (sim)** | `top` camera (overhead) |
| **Real robot cameras** | 4 RGB cameras at 480×640 |
| **Policy input** | Images normalized to [0,1], ImageNet normalization (mean=[0.485,0.456,0.406]) |

### Control & Episode
| Parameter | Value |
|-----------|-------|
| **Control frequency** | **50 Hz** (DT = 0.02s) |
| **Action space** | **14D joint positions** (2× [6 joint + 1 gripper]) |
| **Action chunk size** | **100** (for sim transfer_cube), configurable |
| **Episode length** | **400 steps** (transfer_cube), **500 steps** (insertion) |
| **Time limit** | 20 seconds (400 steps at 50Hz = 8s effective, extended by replay) |

### Success Criteria
| Task | Max Reward |
|------|-----------|
| **Transfer Cube** | 4 (graded: approach, grasp, transfer, release) |
| **Insertion** | 4 (graded: approach, grasp, align, insert) |
| **Real tasks** | 80-96% success with 50 demos |

### Data Collection
| Parameter | Value |
|-----------|-------|
| **Number of demonstrations** | **50** per task (both scripted and human) |
| **Collection method** | Scripted policy (sim) or teleoperation (real) |
| **Scripted policies** | `PickAndTransferPolicy`, `InsertionPolicy` in EE space |
| **Data format** | HDF5 with qpos(14), qvel(14), action(14), images(480×640×3) |
| **Train/val split** | 80/20 random |

### Scripted Policy Design
- Operates in **end-effector space** (ee_sim_env)
- Replays as **joint trajectories** in joint-space sim for recording
- Steps: move-to-pre-grasp → grasp → lift → transfer → release
- Gripper positions normalized to [-1, 1]

---

## 3. Diffusion Policy (Chi et al., Columbia/RSS 2023)

### Environment Dimensions
| Parameter | Value |
|-----------|-------|
| **Push-T workspace** | 2D: 512×512 pixel space, physical ~0.3m×0.3m |
| **Robomimic tasks** | Same as robosuite (see above) |
| **Block Pushing** | Tabletop with 2 blocks, ~0.5m×0.5m workspace |

### Camera Setup
| Parameter | Value |
|-----------|-------|
| **Robomimic tasks** | **84×84** (agentview + wrist cam) |
| **Push-T (sim)** | Top-down 2D view, 96×96 |
| **Real Push-T** | Single overhead camera, 240×320 → cropped |
| **Kitchen** | 3rd person view |

### Control & Episode
| Parameter | Value |
|-----------|-------|
| **Control frequency** | **10 Hz** (Push-T real), **20 Hz** (robomimic tasks) |
| **Action horizon (prediction)** | **16 steps** (predict 16, execute 8 — receding horizon) |
| **Action space** | 2D (Push-T: dx, dy), 7D (robomimic: OSC) |
| **Observation horizon** | 2 steps (current + 1 history) |
| **Episode length** | 200-300 (Push-T), 400 (robomimic Lift/Can) |
| **Diffusion steps** | 100 (DDPM), 10-16 (DDIM for inference) |

### Success Criteria
- **Push-T**: Coverage metric (% of T-block overlapping target region)
- **Robomimic tasks**: Same as robosuite definitions
- **Kitchen**: Number of subtasks completed (out of 4)

### Data Collection
| Parameter | Value |
|-----------|-------|
| **Push-T** | 206 demonstrations (teleoperated) |
| **Robomimic** | Uses existing PH/MH datasets (200-300 demos) |
| **Block Pushing** | 1000 demos (IBC dataset) |

---

## 4. Octo (UC Berkeley, RSS 2024) — Generalist Robot Policy

### Training Data
| Parameter | Value |
|-----------|-------|
| **Dataset** | Open X-Embodiment (OXE), 25 dataset mixture |
| **Total trajectories** | **800,000** episodes |
| **Data directory** | `gs://rail-orca-central2/resize_256_256` |
| **Embodiments** | WidowX, Franka, UR5, Google Robot, etc. |

### Image & Observation
| Parameter | Value |
|-----------|-------|
| **Image resolution** | **256×256** (resized from various native resolutions) |
| **Camera views** | `primary` (3rd person) + `wrist` (eye-in-hand) |
| **Augmentation** | Random resized crop (scale [0.8,1.0], ratio [0.9,1.1]) |
| **Additional augmentations** | Brightness [0.2], contrast [0.9,1.1], saturation [0.9,1.1], hue [0.05] |
| **Tokenizer** | SmallStem16 (16×16 patches) |

### Control & Action
| Parameter | Value |
|-----------|-------|
| **Action dimension** | **7** (default: EEF delta XYZ + RPY + gripper) |
| **Action horizon** | **4** (predict 4 future actions) |
| **Window size (obs history)** | **2** timesteps |
| **Action encoding types** | EEF_POS, JOINT_POS, JOINT_POS_BIMANUAL, NAV_2D |
| **Step duration (Bridge)** | 0.2s (5 Hz effective control) |
| **Action head** | Diffusion (DDPM, 100 timesteps) or L1 (for finetuning) |

### Model Sizes
| Model | Parameters |
|-------|-----------|
| **Octo-Small** | 27M |
| **Octo-Base** | 93M |

### Finetuning
| Parameter | Value |
|-----------|-------|
| **Target demos** | ~100 demonstrations per task |
| **Batch size** | 128 (finetuning), 512 (pretraining) |
| **Action horizon (ALOHA finetune)** | 50 |
| **Action dim (ALOHA finetune)** | 14 (bimanual) |

---

## 5. CrossFormer (Doshi et al., CoRL 2024) — Cross-Embodiment

### Training Data
| Parameter | Value |
|-----------|-------|
| **Total trajectories** | **900,000** across 30 robot embodiments |
| **Embodiments** | Single arm, dual arm, wheeled robots, quadcopters, quadrupeds |
| **Based on** | Open X-Embodiment dataset (extended) |

### Architecture
| Parameter | Value |
|-----------|-------|
| **Architecture** | Decoder-only transformer |
| **Tokenization** | Modality-specific tokenizers (vision, proprioception, task) |
| **Action heads** | Separate per embodiment class |
| **No action alignment** | Different action dimensions natively supported |

### Control Specifics by Embodiment
| Embodiment | Control Freq | Action Dim | Notes |
|------------|-------------|-----------|-------|
| **Single-arm (Franka)** | 5-10 Hz | 7 | EEF delta + gripper |
| **Single-arm (WidowX)** | 5 Hz | 7 | EEF delta + gripper |
| **Bimanual** | **50 Hz** | 14 | Joint positions, longer chunks |
| **Navigation** | 4 Hz | 2 | (delta_x, delta_y) waypoints |
| **Quadruped** | 50 Hz | 12 | Low-level joint control |

### Key Design Choices
- No manual action-space alignment needed
- Flexible token sequence lengths per embodiment
- Separate action heads per embodiment class
- Shared transformer backbone across all embodiments

---

## 6. RT-2 / RT-X (Google DeepMind) — Cross-Embodiment

### Training Data
| Parameter | Value |
|-----------|-------|
| **RT-2** | PaLM-E 12B or PaLI-X 55B as backbone |
| **RT-X (Open X-Embodiment)** | 22 robot types, 160,000+ tasks, 1M+ episodes |
| **RT-1-X** | Trained on OXE data mix |
| **RT-2-X** | 55 billion parameters |

### Image & Observation
| Parameter | Value |
|-----------|-------|
| **RT-2 resolution** | **320×320** (from 640×480 cameras) |
| **RT-1 resolution** | **300×300** |
| **Camera** | Typically single 3rd-person overhead or shoulder-mounted |
| **History** | 6 frames (RT-1), variable (RT-2) |

### Control & Action
| Parameter | Value |
|-----------|-------|
| **Control frequency** | **3 Hz** (RT-1/RT-2 on Google robots) |
| **Action space** | 7D: (x, y, z, roll, pitch, yaw, gripper_extension) + terminate |
| **Action discretization** | 256 bins per dimension (RT-2 tokenizes actions as text) |
| **Episode length** | ~30-60 seconds typical (90-180 steps at 3Hz) |

### Success Criteria
- Binary success/failure per task
- Evaluated by human raters in real-world
- Language-conditioned: "pick up the red cup" etc.

---

## 7. RoboCasa (UT Austin, RSS 2024) — Large-Scale Kitchen Simulation

### Environment Dimensions
| Parameter | Value |
|-----------|-------|
| **Scenes** | 120 kitchens (original), **2,500** (RoboCasa365) |
| **Kitchen layouts** | Based on real-world architecture magazines |
| **Standard specifications** | Modeled to real-world kitchen dimensions |
| **Object assets** | **3,200+** objects across 150+ categories |

### Camera & Visual
| Parameter | Value |
|-----------|-------|
| **Textures** | AI-generated via MidJourney (100 wall, 100 floor, 100 counter, 100 cabinet) |
| **Object sources** | Objaverse 1.0, LightWheel AI, Luma AI (text-to-3D) |
| **Domain randomization** | Texture swapping for diversity |
| **Styles** | 50+ visual styles per layout |

### Tasks
| Parameter | Value |
|-----------|-------|
| **Atomic tasks (RoboCasa365)** | 65 tasks across 10 foundational skills |
| **Composite tasks** | 365 everyday activities (LLM-guided) |
| **Skills** | Pick/place, open/close doors, drawers, twist knobs, turn levers, press buttons, insertion, navigation, sliding racks, open/close lids |

### Supported Embodiments
- Single-arm mobile manipulators
- Humanoid robots
- Quadruped robots with arms

### Data
| Parameter | Value |
|-----------|-------|
| **Human demos** | 600+ hours (RoboCasa365) |
| **Synthetic demos** | 1,600+ hours (automated generation) |
| **Supported policies** | Diffusion Policy, π₀, GR00T |

---

## 8. MimicGen (NVIDIA/Stanford, CoRL 2023) — Data Generation

### Core Concept
- Takes **small** source datasets (10-20 human demos) and generates **large** datasets (1000+ demos)
- Automated trajectory generation via subtask decomposition + spatial transforms

### Environment Setup
| Parameter | Value |
|-----------|-------|
| **Base environment** | robosuite (same table/object specs as above) |
| **Control frequency** | **20 Hz** |
| **Camera** | `agentview`, `robot0_eye_in_hand` |
| **Image resolution** | 84×84 (same as robomimic) |

### Task Horizons
| Task | Horizon (steps) |
|------|----------------|
| **Square** | 400 |
| **Coffee** | 400 |
| **Coffee Preparation** | 800 |
| **Nut Assembly** | 500 |
| **Pick Place** | 1000 |
| **Mug Cleanup** | 500 |

### Data Generation
| Parameter | Value |
|-----------|-------|
| **Source demos** | 10 human demonstrations (typical) |
| **Generated demos** | **1,000** trajectories per task |
| **Success guarantee** | Keep generating until N successes |
| **Max failures** | 25 before stopping |
| **Generation process** | Subtask decomposition → spatial transform → interpolation → execution |
| **Video FPS** | 20 fps for rendered playback |

### Scripted Approach
- Decompose task into subtasks via termination signals (e.g., `grasp_1`, `insert_1`)
- Transform end-effector poses between source and new configurations
- Interpolation segments bridge subtask transitions
- Action noise can be injected for diversity

---

## 9. HPT (Wang et al., NeurIPS 2024) — Heterogeneous Pre-Trained Transformers

### Training Data
| Parameter | Value |
|-----------|-------|
| **Datasets** | **52 datasets**, 200k+ trajectories |
| **Sources** | Real robot teleops, human videos, simulation, deployed robots |
| **Embodiments** | Multiple (proprioception varies per robot) |

### Architecture
| Parameter | Value |
|-----------|-------|
| **Structure** | Stems → Trunk → Heads |
| **Stems** | Embodiment-specific tokenizers (vision + proprioception → tokens) |
| **Trunk** | Large shared transformer (pre-trained) |
| **Heads** | Task/embodiment-specific action decoders |
| **Key insight** | Align heterogeneous inputs to shared latent token space |

### Key Results
- Outperforms baselines by **20%+** on unseen tasks
- Shows objective scaling behaviors across:
  - Model size
  - Dataset size  
  - Compute

### Supported Benchmarks
- Multiple simulator benchmarks (MetaWorld, robosuite, etc.)
- Real-world manipulation tasks
- Contact-rich precision tasks

---

## 10. Summary Comparison Table

| Project | Resolution | Control Hz | Action Dim | Episode Len | Demos | Collection |
|---------|-----------|-----------|-----------|-------------|-------|------------|
| **robosuite/robomimic** | 84×84 | 20 | 7 (OSC) | 400-700 | 200-300 | Teleoperated |
| **ACT/ALOHA** | 480×640 | 50 | 14 (joint) | 400-500 | 50 | Scripted + Teleop |
| **Diffusion Policy** | 84×84 / 96×96 | 10-20 | 2-7 | 200-400 | 200-1000 | Teleoperated |
| **Octo** | 256×256 | 5 (bridge) | 7 (EEF) | varies | 800K total | Mixed (OXE) |
| **CrossFormer** | 256×256 | 5-50 | 2-14 | varies | 900K total | Mixed (OXE+) |
| **RT-2/RT-X** | 320×320 | 3 | 7+1 | 90-180 | 1M+ total | Teleoperated |
| **RoboCasa** | varies | 20 | varies | varies | 2200+ hrs | Human + Synthetic |
| **MimicGen** | 84×84 | 20 | 7 (OSC) | 400-1000 | 10→1000 | Auto-generated |
| **HPT** | varies | varies | varies | varies | 200K traj | Mixed heterogeneous |

---

## Key Takeaways for Your Setup

### Recommended Defaults (based on consensus):
1. **Table**: 0.8m × 0.8m × 0.05m at height 0.8m (robosuite standard)
2. **Cube**: 2cm edge length (0.02m), red color
3. **Camera resolution**: 
   - **84×84** for fast sim training (robomimic standard)
   - **128×128** as good middle ground
   - **256×256** for generalist policies (Octo/CrossFormer)
4. **Control frequency**: **20 Hz** (most common for manipulation)
5. **Action space**: 7D OSC (delta EEF + gripper) for single arm
6. **Episode length**: 400 steps (20s at 20Hz) for basic tasks
7. **Demonstrations**: 50 for simple tasks (ACT), 200 for robust policies
8. **Camera placement**: 
   - `agentview`: pos ~(0.6, 0, 1.6), angled 45° down
   - `wrist`: mounted on gripper
   - `topdown`: directly overhead
9. **Success threshold**: Lift task = cube > table + 4cm
10. **Scripted policy**: EE-space waypoints → replay as joint trajectories

### For Cross-Embodiment (your project):
- Use **256×256** images (Octo/CrossFormer standard)
- Support **variable action dimensions** per robot
- **Action chunking** (predict 4-50 future steps)
- Separate **action heads per embodiment** (CrossFormer approach)
- Train with **mixed control frequencies** (normalize via subsampling)
