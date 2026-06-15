"""
Control algorithms for the ``EndlessRacerEnv`` environment (``racer_env.py``).

This file is consumed by ``simulation.py``. It is also the export target of
``notebooks/controllers_solution.ipynb`` (and, once you've completed the
exercises, ``notebooks/controllers_exercise.ipynb``) -- running the export
cell in either notebook regenerates this file from the notebook's code
cells.

Environment recap
------------------
- Observation (obstacles off): ``[offset]``
    - ``offset``: distance from the centre line (m, 0 = lane centre,
      positive = right of centre)
- Observation (obstacles on): ``[offset, angle_1, ..., angle_N]``
    - ``angle_i``: bearing to the i-th visible obstacle, nearest first
      (positive = obstacle is to the right). Unused slots are padded with
      the sentinel ``NO_OBSTACLE_ANGLE`` (= pi).
- Action: a single steering command, ``Box(-1.0, 1.0, (1,))``
  (negative = steer left, positive = steer right).
- Reward: +1 per step the car stays on the track; a large negative penalty
  on collision with the lane edges or an obstacle (which ends the episode).
- Disturbances: random drift each step, plus occasional wheel slippage during
  which steering effectiveness drops and the car drifts left or right.
- ``info`` dict: includes (in obstacles-on mode) ``"obstacles"``, the
  relative ``(dx, dy)`` positions of the visible traffic cars, used by the
  classical planner.
"""

import math
import os

import numpy as np

from racer_env import NO_OBSTACLE_ANGLE


# ---------------------------------------------------------------------------
# 1. Manual control
# ---------------------------------------------------------------------------
def get_manual_action(action_space, pressed_keys, magnitude=1.0):
    """Convert currently-held keyboard keys into a steering action.

    Parameters
    ----------
    action_space : gymnasium.spaces.Box
        The environment's action space.
    pressed_keys : dict
        Maps key names (``"left"``, ``"right"``) to booleans indicating
        whether the corresponding arrow key is currently held down.
    magnitude : float
        Magnitude of the steering command to apply while a key is held.

    Returns
    -------
    numpy.ndarray
        The action to send to ``env.step``, clipped to ``action_space``.
    """
    steer = 0.0
    if pressed_keys.get("left"):
        steer -= magnitude
    if pressed_keys.get("right"):
        steer += magnitude

    action = np.array([steer], dtype=action_space.dtype)
    return np.clip(action, action_space.low, action_space.high)


# ---------------------------------------------------------------------------
# 2. Desired lateral offset (the PID controller's setpoint)
# ---------------------------------------------------------------------------
def compute_desired_offset(
    observation,
    info,
    obstacles_on,
    plan_horizon=25.0,
    cross_lead=7.0,
    cross_per_m=2.0,
    clearance=2.2,
    corridor=4.5,
):
    """Compute the lateral position the car *should* be at right now.

    Obstacles off
        The goal is simply to drive in the middle of the lane, so the desired
        offset is the centre line itself: ``0.0``.

    Obstacles on
        The controller plans a safe *lateral target position* from the
        visible traffic. The observation's obstacle bearings tell us where
        traffic is, and the ``info`` dict provides the same cars as relative
        ``(dx, dy)`` positions (the classical controller is allowed richer
        telemetry than the RL agent). The plan has three ingredients:

        1. *Reachability* -- we can only swerve so fast, so the lane line of a
           car that is already close cannot be crossed any more: we must stay
           on whichever side of it we currently are. A line is considered
           crossable only while ``dy > cross_lead + cross_per_m * |dx|``.
        2. *Free gaps* -- every car within ``plan_horizon`` blocks the band of
           lateral positions within ``clearance`` of its centre. Whatever is
           left of the drivable corridor (``+/- corridor``) and still
           reachable forms the free gaps. We aim for the gap that requires
           the smallest move, drifting slightly towards its middle for
           margin.
        3. *Squeeze fallback* -- if no gap is free, we put as much room as
           possible between us and nearby traffic by driving to whichever
           reachable extreme is farthest from the closest cars.

    Parameters
    ----------
    observation : array-like
        The environment observation; ``observation[0]`` is the car's current
        distance from the centre line (m).
    info : dict
        The ``info`` dict from ``env.step``/``env.reset``; in obstacles-on
        mode it must contain ``"obstacles"``, a list of relative ``(dx, dy)``
        positions of the visible traffic cars.
    obstacles_on : bool
        Which mode the environment is in.
    plan_horizon : float
        Cars further ahead than this (m) are ignored by the planner.
    cross_lead, cross_per_m : float
        Crossability rule: a car's lane line can only be crossed while
        ``dy > cross_lead + cross_per_m * |dx|``.
    clearance : float
        Half-width (m) of the lateral band each car blocks.
    corridor : float
        Half-width (m) of the drivable corridor the planner uses.

    Returns
    -------
    float
        The desired lateral offset (m from the centre line, positive =
        right of centre).
    """
    if not obstacles_on:
        return 0.0

    offset = float(observation[0])

    # Traffic cars as absolute lateral position + distance ahead.
    cars = [
        (offset + dx, dy)
        for dx, dy in info.get("obstacles", [])
        if dy < plan_horizon
    ]

    # 1. Reachability: stay on our side of any line we can no longer cross.
    reach_lo, reach_hi = -corridor, corridor
    for car_x, dy in cars:
        crossable = dy > cross_lead + cross_per_m * abs(car_x - offset)
        if not crossable:
            if offset >= car_x:
                reach_lo = max(reach_lo, car_x)
            else:
                reach_hi = min(reach_hi, car_x)

    # 2. Free gaps between the blocked bands, inside the reachable range.
    free, lo = [], reach_lo
    for b_lo, b_hi in sorted((x - clearance, x + clearance) for x, _ in cars):
        if b_lo > lo:
            free.append((lo, min(b_lo, reach_hi)))
        lo = max(lo, b_hi)
    if lo < reach_hi:
        free.append((lo, reach_hi))
    free = [(a, b) for a, b in free if b - a > 0.5 and b > reach_lo]

    if free:
        # Aim for the gap that needs the smallest lateral move.
        def move_needed(gap):
            g_lo, g_hi = gap
            if g_lo <= offset <= g_hi:
                return 0.0
            return min(abs(offset - g_lo), abs(offset - g_hi))

        g_lo, g_hi = min(free, key=move_needed)
        target = float(np.clip(offset, g_lo + 0.3, g_hi - 0.3))
        # Drift gently towards the middle of the gap for extra margin.
        target = 0.8 * target + 0.2 * (g_lo + g_hi) / 2.0
    else:
        # 3. Squeeze: maximise room to the closest cars.
        close = [c for c in cars if c[1] < 15.0]
        if close:
            def room(p):
                return min(abs(p - x) for x, _ in close)

            target = max((reach_lo, reach_hi), key=room)
        else:
            target = float(np.clip(offset, reach_lo, reach_hi))

    return float(target)


# ---------------------------------------------------------------------------
# 3. PID control
# ---------------------------------------------------------------------------
class PIDController:
    """A PID controller that steers the car's lateral offset to ``setpoint``.

    The controller drives the distance from the centre line
    (``observation[0]``) to the desired offset by issuing a steering command:

        u(t) = Kp * e(t) + Ki * integral(e) + Kd * d(e)/dt

    where ``e(t) = setpoint(t) - offset(t)``. The setpoint is updated every
    step by ``compute_desired_offset`` (lane centring / obstacle avoidance).
    Because steering controls the *heading* (and the heading in turn moves
    the car sideways), the derivative term is essential for damping -- a
    purely proportional controller will oscillate around the centre line.
    """

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.reset()

    def reset(self):
        """Clear the integral and derivative memory (call between episodes)."""
        self._integral = 0.0
        self._prev_error = 0.0

    def set_gains(self, kp, ki, kd):
        """Update the P, I and D gains (e.g. from GUI sliders)."""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_setpoint(self, setpoint):
        """Update the desired lateral offset (m from the centre line)."""
        self.setpoint = setpoint

    def compute_action(self, observation, action_space, dt):
        """Compute the steering action for the current observation.

        Parameters
        ----------
        observation : array-like
            The environment observation; ``observation[0]`` is the car's
            distance from the centre line (m).
        action_space : gymnasium.spaces.Box
            The environment's action space, used to clip the output.
        dt : float
            Time step between calls (``env.dt``), used for the integral and
            derivative terms.

        Returns
        -------
        numpy.ndarray
            The action to send to ``env.step``, clipped to ``action_space``.
        """
        offset = observation[0]
        error = self.setpoint - offset

        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        action = np.array([output], dtype=action_space.dtype)
        return np.clip(action, action_space.low, action_space.high)


# ---------------------------------------------------------------------------
# 4. Reinforcement learning control (Stable-Baselines3 PPO)
# ---------------------------------------------------------------------------
def train_rl_agent(env, hyperparams, total_timesteps, save_path, callback=None):
    """Train a PPO agent on ``env`` and save it to ``save_path``.

    Parameters
    ----------
    env : gymnasium.Env
        The (non-rendered) training environment.
    hyperparams : dict
        PPO hyperparameters. Recognised keys: ``learning_rate``, ``n_steps``,
        ``batch_size``, ``gamma``, ``gae_lambda``, ``ent_coef``. Missing keys
        fall back to Stable-Baselines3 defaults.
    total_timesteps : int
        Number of environment steps to train for.
    save_path : str
        Path (without/with ``.zip``) to save the trained model checkpoint.
    callback : stable_baselines3.common.callbacks.BaseCallback, optional
        Callback passed through to ``model.learn`` (e.g. for progress
        reporting in the GUI).

    Returns
    -------
    stable_baselines3.PPO
        The trained model.
    """
    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparams.get("learning_rate", 3e-4),
        n_steps=hyperparams.get("n_steps", 1024),
        batch_size=hyperparams.get("batch_size", 64),
        gamma=hyperparams.get("gamma", 0.99),
        gae_lambda=hyperparams.get("gae_lambda", 0.95),
        ent_coef=hyperparams.get("ent_coef", 0.0),
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_path)

    return model


def load_rl_agent(path, env=None):
    """Load a previously-trained PPO model from ``path``.

    Parameters
    ----------
    path : str
        Path to the saved model (``.zip`` checkpoint).
    env : gymnasium.Env, optional
        Environment to attach to the loaded model.

    Returns
    -------
    stable_baselines3.PPO
        The loaded model.
    """
    from stable_baselines3 import PPO

    return PPO.load(path, env=env)


def get_rl_action(model, observation, deterministic=True):
    """Get the action chosen by a trained RL model for ``observation``.

    Parameters
    ----------
    model : stable_baselines3.PPO
        A trained (or loaded) model.
    observation : array-like
        The environment observation.
    deterministic : bool
        Whether to use the deterministic policy (recommended for evaluation).

    Returns
    -------
    numpy.ndarray
        The action to send to ``env.step``.
    """
    action, _ = model.predict(observation, deterministic=deterministic)
    return action
