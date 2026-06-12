# Endless Racer Control Simulator

A small Tkinter desktop app built around a custom Gymnasium-style
environment (`racer_env.py`): a **top-down, vertically scrolling endless
racer** with random wheel slippage and random drift, so the car never drives
in a straight line without external control. Three control modes are
available:

1. **Manual control** -- steer the car yourself with the arrow keys (or
   on-screen buttons).
2. **PID control** -- a PID controller steers the car; tune Kp/Ki/Kd live
   with sliders.
3. **Reinforcement learning control** -- train (or load) a Stable-Baselines3
   PPO agent and watch it drive.

Each mode can be played with **obstacles off** (just stay in the lane) or
**obstacles on** (dodge slower traffic cars, drawn in red so they are
visually distinct from your blue car), toggled on the main menu.

This repository is the companion to
[inverted-pendulum-control](https://github.com/VikramRadhakrishnan/inverted-pendulum-control)
and follows the same structure.

## How it works

The environment (`EndlessRacerEnv` in `racer_env.py`) follows the Gymnasium
API (`reset`, `step`, `action_space`, `observation_space`):

- **State (obstacles off):** `[theta]` -- the car's heading angle (rad,
  0 = straight up the track, positive = pointing right).
- **State (obstacles on):** `[theta, angle_1, ..., angle_5]` -- the heading
  angle followed by the bearings to each obstacle currently visible in
  frame, nearest first (positive = obstacle to the right). Unused slots are
  padded with the sentinel `NO_OBSTACLE_ANGLE` (= pi).
- **Action:** a single steering command in `Box(-1, 1, (1,))`
  (negative = left, positive = right).
- **Reward:** the forward distance travelled each step. Colliding with the
  lane edges -- or with an obstacle, in obstacles-on mode -- ends the episode
  with a **-100 penalty**.
- **Disturbances:** every step the heading and lateral position receive
  random drift, and with some probability the wheels start *slipping* for a
  while (steering effectiveness drops to 15-50%), so active control is always
  needed.

The car's lateral offset from the lane centre and other useful quantities
are reported in the `info` dict, including (in obstacles-on mode)
`info["obstacles"]`: the relative `(dx, dy)` positions of the visible
traffic cars, in the same nearest-first order as the bearings in the
observation.

- **`controllers.py`** contains the control algorithms, each in its own
  function/class so they can be studied or re-implemented independently of
  the GUI:
  * `get_manual_action(action_space, pressed_keys)` -- maps held arrow
    keys/buttons to a steering command.
  * `compute_desired_heading(observation, info, obstacles_on)` -- the PID
    setpoint. With obstacles off it steers the car back to the middle of the
    lane; with obstacles on, it plans a safe lateral target from the
    visible traffic (free-gap selection with a reachability rule and a
    squeeze fallback) and converts the lateral error into a heading.
  * `PIDController` -- a PID controller (`compute_action`) with live-tunable
    `kp`/`ki`/`kd` gains that drives the heading to the setpoint.
  * `train_rl_agent`, `load_rl_agent`, `get_rl_action` -- train/load a
    Stable-Baselines3 PPO agent and query it for actions.
- **`racer_env.py`** is the Gymnasium-style environment, including the
  simple top-down Tkinter rendering (scrolling lane markings, blue player
  car, red traffic cars).
- **`simulation.py`** is the Tkinter application: the main menu (with the
  obstacles on/off toggle), the three mode screens, the simulation step loop
  (driven by `root.after`), and RL training/checkpoint management.
- **`notebooks/`** contains a Jupyter exercise notebook (the same functions
  left unimplemented, with explanations) and a solution notebook, each with
  an export cell that regenerates `controllers.py`.

## Project layout

```
racer_env.py                  # the Gymnasium-style endless racer environment
controllers.py                # the control algorithms (used by the simulator)
simulation.py                 # the Tkinter app / simulation loop
requirements.txt              # Python dependencies
checkpoints/                  # trained RL models are saved here
notebooks/
  controllers_exercise.ipynb  # same functions, left blank, with explanations
  controllers_solution.ipynb  # fully worked solutions + export script
```

## Setup

Requires Python 3.10+ with Tkinter (included in most Python installs).

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the simulator

```
python simulation.py
```

This opens a window with the **Obstacles (traffic cars)** toggle and three
buttons: **Manual Control**, **PID Control**, and **Reinforcement Learning
Control**.

The track renders in the left half of the window; a control panel with
status readouts (distance, heading, lateral offset, reward) sits on the
right. When an episode ends (a collision or the step limit), the simulation
pauses and the panel shows the final episode reward with a **Start New
Episode** button. Click it, or press `R`, to reset and continue -- or use
**Back to Menu** / `Esc` to exit.

### Manual control

- `←` / `→` -- steer left / right
- `R` -- reset the episode immediately
- `Esc` -- return to the main menu

The control panel also has **◀ Left** / **Right ▶** buttons that work the
same as the arrow keys, for trackpad/touch use.

### PID control

Three sliders (Kp, Ki, Kd) let you tune the controller while it runs. Every
step, `compute_desired_heading` produces the setpoint:

- **Obstacles off:** the desired heading gently points the car back towards
  the middle of the lane, so the controller keeps it driving forward and
  centred despite the drift and slippage.
- **Obstacles on:** the controller plans a safe *lateral target* from the
  visible traffic and turns the lateral error into a desired heading. Each
  visible car blocks a band of lateral positions; the lane line of a car
  that is already close can no longer be crossed (we stay on our side of
  it); the controller aims for the nearest free gap, and if no gap is free
  it squeezes towards whichever reachable extreme has the most room. The
  observation provides the obstacle bearings, and the `info` dict provides
  the same cars as relative `(dx, dy)` positions for the planner -- the
  classical controller gets richer telemetry than the RL agent, just as it
  already uses `lateral_offset` in obstacles-off mode.

The PID controller then steers the heading to that setpoint. Try
`Kp=3, Ki=0, Kd=0.4` as a starting point, then experiment. `R` resets the
episode immediately, `Esc` returns to the menu.

### Reinforcement learning control

- If no checkpoint exists yet for the selected mode
  (`checkpoints/ppo_racer_obstacles_off.zip` /
  `checkpoints/ppo_racer_obstacles_on.zip`), you configure PPO
  hyperparameters (learning rate, n_steps, batch_size, gamma, gae_lambda,
  ent_coef, total_timesteps) and train a new agent. A progress bar tracks
  training, which runs in the background so the UI stays responsive.
- Once a checkpoint exists, you can choose to **use the existing trained
  model** or **train a new one** (overwriting the checkpoint).
- After training/loading, the trained agent drives the car in the renderer.

Note that the obstacles-on observation only contains *angles* (no
distances), which makes it a genuinely interesting -- and partially
observable -- RL problem.

## Notebooks: implement the algorithms yourself

`controllers.py` ships with working implementations so the simulator runs
out of the box. To learn how each algorithm works (or re-implement them
yourself):

1. Open `notebooks/controllers_exercise.ipynb`. Each control algorithm has a
   markdown cell explaining how it works and step-by-step implementation
   instructions, followed by a code cell with the function/class signature
   but the body left as `# YOUR CODE HERE` / `raise NotImplementedError`.
2. Implement `get_manual_action`, `compute_desired_heading`,
   `PIDController`, `train_rl_agent`, `load_rl_agent`, and `get_rl_action`.
3. Run the **export cell** at the end of the notebook -- it writes your
   implementations to `../controllers.py`.
4. Run `python simulation.py` from the project root to test your
   implementations.

If you get stuck, `notebooks/controllers_solution.ipynb` contains the same
explanations with fully worked implementations, plus the same export cell.
