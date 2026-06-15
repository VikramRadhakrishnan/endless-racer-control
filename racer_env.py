"""
A simple top-down "endless racer" environment in the style of Gymnasium.

The car drives forward at a constant speed on an endless, vertically
scrolling track. Small random drift perturbs the car every step, and every
so often the wheels *slip*: steering effectiveness drops and the car drifts
to the left or right of the centre line until grip returns. Without external
steering control the car will not stay on the track. The agent's job is to
steer.

Two modes:

- **Obstacles off** -- the observation is the car's distance from the centre
  line ``[offset]`` (m, 0 = lane centre, positive = right of centre).
- **Obstacles on** -- slower "traffic" cars appear ahead and must be dodged.
  The observation is ``[offset, angle_1, ..., angle_N]`` where ``angle_i`` is
  the bearing from the car to the i-th visible obstacle (sorted nearest
  first, positive = obstacle is to the right). Unused slots are padded with
  the sentinel ``NO_OBSTACLE_ANGLE`` (= pi, i.e. "directly behind", which a
  visible obstacle can never be).

The goal is to stay on the track as long as possible: the reward is +1 for
every step the car survives. Colliding with the lane edges (or an obstacle,
in obstacles-on mode) ends the episode with a negative reward as a penalty.

Extra quantities useful for control (e.g. the car's lateral offset from the
lane centre) are reported in the ``info`` dict returned by ``reset``/``step``.
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Physical / gameplay constants (metres, seconds, radians)
# ---------------------------------------------------------------------------
DT = 0.05                     # simulation time step (s)
CAR_SPEED = 20.0              # constant forward speed of our car (m/s)
LANE_HALF_WIDTH = 6.0         # distance from lane centre to each wall (m)
CAR_WIDTH = 1.8               # car width (m)
CAR_LENGTH = 4.0              # car length (m)

MAX_STEER_RATE = 1.6          # max heading change rate at full steering (rad/s)
MAX_HEADING = math.pi / 3.0   # heading is clipped to +/- 60 degrees

# Random disturbances -- these are what make the problem interesting.
HEADING_NOISE_STD = 0.02      # random drift in heading per step (rad)
LATERAL_NOISE_STD = 0.03      # random lateral drift per step (m)
SLIP_PROB = 0.02              # chance per step that the wheels start slipping
SLIP_DURATION_RANGE = (5, 15)   # slip lasts this many steps (uniform)
SLIP_GRIP_RANGE = (0.25, 0.6)    # steering effectiveness while slipping
SLIP_DRIFT_RATE_RANGE = (0.2, 0.6)  # heading drift rate while slipping (rad/s),
                                    # applied towards a random side (left/right)

# Obstacles (only used in obstacles-on mode)
VIEW_DISTANCE = 60.0          # how far ahead obstacles are visible (m)
MAX_VISIBLE_OBSTACLES = 5     # observation slots for obstacle angles
NO_OBSTACLE_ANGLE = math.pi   # sentinel for unused obstacle slots
OBSTACLE_SPEED_RANGE = (9.0, 11.0)   # traffic cars are slower than us (m/s)
OBSTACLE_GAP_RANGE = (40.0, 70.0)    # forward gap between spawned obstacles (m)

# Episode
COLLISION_PENALTY = -100.0
MAX_EPISODE_STEPS = 2000

# Rendering (pixels)
RENDER_WIDTH = 360
RENDER_HEIGHT = 540
PPM = 24.0                    # pixels per metre
CAR_SCREEN_Y = RENDER_HEIGHT - 110   # screen y of our car (px from top)


class EndlessRacerEnv(gym.Env):
    """Top-down endless racer with random slippage and drift.

    Parameters
    ----------
    obstacles : bool
        ``False``: observation is ``[offset]`` (distance from the centre
        line, positive = right of centre).
        ``True``: observation is ``[offset, angle_1, ..., angle_N]`` with
        ``N = MAX_VISIBLE_OBSTACLES`` slots padded by ``NO_OBSTACLE_ANGLE``.
    render_mode : str or None
        ``"human"`` opens a standalone Tkinter window; ``None`` disables
        rendering (used for RL training). The simulator GUI instead embeds
        a canvas and calls :meth:`draw` directly.
    """

    metadata = {"render_modes": ["human"], "render_fps": int(1 / DT)}

    def __init__(self, obstacles=False, render_mode=None):
        super().__init__()
        self.obstacles_on = bool(obstacles)
        self.render_mode = render_mode

        # Action: steering command in [-1, 1] (negative = left, positive = right)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation: [offset] or [offset, angle_1, ..., angle_N].
        # The first entry is the distance from the centre line (m), the
        # remaining entries (obstacles-on only) are obstacle bearings (rad).
        obs_dim = 1 + (MAX_VISIBLE_OBSTACLES if self.obstacles_on else 0)
        low = np.full(obs_dim, -math.pi, dtype=np.float32)
        high = np.full(obs_dim, math.pi, dtype=np.float32)
        low[0], high[0] = -LANE_HALF_WIDTH, LANE_HALF_WIDTH
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.dt = DT

        # State
        self.x = 0.0            # lateral position (0 = lane centre)
        self.theta = 0.0        # heading (0 = straight ahead)
        self.distance = 0.0     # total forward distance travelled
        self.steps = 0
        self._slip_steps_left = 0
        self._grip = 1.0
        self._slip_drift = 0.0  # heading drift rate (rad/s) while slipping
        self._obstacles = []    # list of dicts: {"x", "y", "speed"} (y = metres ahead)
        self._next_spawn_y = 0.0

        self._window = None     # standalone Tk window for render_mode="human"
        self._canvas = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 0.0
        self.theta = 0.0
        self.distance = 0.0
        self.steps = 0
        self._slip_steps_left = 0
        self._grip = 1.0
        self._slip_drift = 0.0
        self._obstacles = []
        self._next_spawn_y = VIEW_DISTANCE * 0.75
        if self.obstacles_on:
            # Pre-populate the road ahead.
            while self._next_spawn_y < VIEW_DISTANCE:
                self._spawn_obstacle(self._next_spawn_y)
                self._next_spawn_y += self.np_random.uniform(*OBSTACLE_GAP_RANGE)
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(-1),
            self.action_space.low,
            self.action_space.high,
        )
        steer = float(action[0])

        # --- random wheel slippage -------------------------------------
        # Every so often the wheels lose grip: steering effectiveness drops
        # and the car drifts towards a random side (left or right) of the
        # centre line until grip returns.
        if self._slip_steps_left > 0:
            self._slip_steps_left -= 1
            if self._slip_steps_left == 0:
                self._grip = 1.0
                self._slip_drift = 0.0
        elif self.np_random.random() < SLIP_PROB:
            self._slip_steps_left = int(self.np_random.integers(*SLIP_DURATION_RANGE))
            self._grip = float(self.np_random.uniform(*SLIP_GRIP_RANGE))
            direction = 1.0 if self.np_random.random() < 0.5 else -1.0
            self._slip_drift = direction * float(
                self.np_random.uniform(*SLIP_DRIFT_RATE_RANGE)
            )

        # --- car dynamics (+ random drift) ------------------------------
        self.theta += steer * MAX_STEER_RATE * self._grip * self.dt
        self.theta += self._slip_drift * self.dt
        self.theta += self.np_random.normal(0.0, HEADING_NOISE_STD)
        self.theta = float(np.clip(self.theta, -MAX_HEADING, MAX_HEADING))

        forward = CAR_SPEED * math.cos(self.theta) * self.dt
        self.x += CAR_SPEED * math.sin(self.theta) * self.dt
        self.x += self.np_random.normal(0.0, LATERAL_NOISE_STD)
        self.distance += forward
        self.steps += 1

        # --- obstacles scroll towards us --------------------------------
        if self.obstacles_on:
            self._advance_obstacles(forward)

        # --- reward & termination ---------------------------------------
        reward = 1.0  # +1 for every step the car stays on the track
        terminated = False

        if abs(self.x) + CAR_WIDTH / 2.0 >= LANE_HALF_WIDTH:
            reward = COLLISION_PENALTY        # hit the lane edge
            terminated = True
        elif self.obstacles_on and self._check_obstacle_collision():
            reward = COLLISION_PENALTY        # hit a traffic car
            terminated = True

        truncated = self.steps >= MAX_EPISODE_STEPS

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------------------
    def _spawn_obstacle(self, y):
        margin = CAR_WIDTH  # keep a drivable gap near the walls
        self._obstacles.append(
            {
                "x": float(
                    self.np_random.uniform(
                        -(LANE_HALF_WIDTH - margin), LANE_HALF_WIDTH - margin
                    )
                ),
                "y": float(y),
                "speed": float(self.np_random.uniform(*OBSTACLE_SPEED_RANGE)),
            }
        )

    def _advance_obstacles(self, forward):
        for ob in self._obstacles:
            # Obstacles drive forward too, but slower, so they approach us.
            ob["y"] -= forward - ob["speed"] * self.dt
        # Drop obstacles once they are well behind us.
        self._obstacles = [ob for ob in self._obstacles if ob["y"] > -3 * CAR_LENGTH]
        # Spawn new traffic at the horizon.
        self._next_spawn_y -= forward
        while self._next_spawn_y < VIEW_DISTANCE:
            spawn_at = max(self._next_spawn_y, VIEW_DISTANCE)
            self._spawn_obstacle(spawn_at)
            self._next_spawn_y += self.np_random.uniform(*OBSTACLE_GAP_RANGE)

    def _check_obstacle_collision(self):
        for ob in self._obstacles:
            if (
                abs(ob["x"] - self.x) < CAR_WIDTH
                and abs(ob["y"]) < CAR_LENGTH
            ):
                return True
        return False

    def visible_obstacles(self):
        """Obstacles ahead of the car and within ``VIEW_DISTANCE``."""
        return sorted(
            (ob for ob in self._obstacles if -CAR_LENGTH < ob["y"] <= VIEW_DISTANCE),
            key=lambda ob: ob["y"],
        )

    def obstacle_angles(self):
        """Bearings (rad) to visible obstacles, nearest first.

        Positive = obstacle is to the right of the car.
        """
        return [
            math.atan2(ob["x"] - self.x, ob["y"]) for ob in self.visible_obstacles()
        ]

    # ------------------------------------------------------------------
    # Observation / info
    # ------------------------------------------------------------------
    def _get_obs(self):
        offset = float(np.clip(self.x, -LANE_HALF_WIDTH, LANE_HALF_WIDTH))
        if not self.obstacles_on:
            return np.array([offset], dtype=np.float32)
        angles = self.obstacle_angles()[:MAX_VISIBLE_OBSTACLES]
        padded = angles + [NO_OBSTACLE_ANGLE] * (MAX_VISIBLE_OBSTACLES - len(angles))
        return np.array([offset] + padded, dtype=np.float32)

    def _get_info(self):
        visible = self.visible_obstacles() if self.obstacles_on else []
        return {
            "lateral_offset": self.x,
            "heading": self.theta,
            "distance": self.distance,
            "grip": self._grip,
            "slip_drift": self._slip_drift,
            "num_visible_obstacles": len(visible),
            # Relative (dx, dy) of each visible traffic car, nearest first
            # (same order as the angles in the observation). dx > 0 means the
            # car is to our right, dy is metres ahead (can be slightly
            # negative for a car right beside us).
            "obstacles": [(ob["x"] - self.x, ob["y"]) for ob in visible],
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        """Standalone Tkinter window (``render_mode="human"``)."""
        if self.render_mode != "human":
            return
        import tkinter as tk

        if self._window is None:
            self._window = tk.Tk()
            self._window.title("Endless Racer")
            self._canvas = tk.Canvas(
                self._window, width=RENDER_WIDTH, height=RENDER_HEIGHT
            )
            self._canvas.pack()
        self.draw(self._canvas)
        self._window.update()

    def draw(self, canvas):
        """Draw the current state onto any Tkinter canvas (used by the GUI)."""
        w, h = RENDER_WIDTH, RENDER_HEIGHT
        cx = w / 2.0

        canvas.delete("all")
        canvas.configure(bg="#3a7d44")  # grass

        # Road
        road_half_px = LANE_HALF_WIDTH * PPM
        canvas.create_rectangle(
            cx - road_half_px, 0, cx + road_half_px, h, fill="#4a4a4a", outline=""
        )
        # Lane edges
        for side in (-1, 1):
            edge = cx + side * road_half_px
            canvas.create_rectangle(
                edge - 4, 0, edge + 4, h, fill="#e8e8e8", outline=""
            )

        # Scrolling centre-line dashes (scroll with distance travelled)
        dash_len, dash_gap = 2.5 * PPM, 2.5 * PPM
        period = dash_len + dash_gap
        offset = (self.distance * PPM) % period
        y = -period + offset
        while y < h:
            canvas.create_rectangle(
                cx - 3, y, cx + 3, y + dash_len, fill="#f4d35e", outline=""
            )
            y += period

        # Obstacle cars (red, with a roof patch to look distinct)
        if self.obstacles_on:
            for ob in self._obstacles:
                ox = cx + (ob["x"] - self.x) * PPM
                oy = CAR_SCREEN_Y - ob["y"] * PPM
                self._draw_car(canvas, ox, oy, 0.0, body="#c1121f", roof="#780000")

        # Our car (blue), rotated by heading
        self._draw_car(canvas, cx, CAR_SCREEN_Y, self.theta, body="#1d6fd1", roof="#0b3d91")

        # HUD
        canvas.create_text(
            8, 8, anchor="nw", fill="white", font=("Helvetica", 10, "bold"),
            text=f"distance: {self.distance:7.1f} m",
        )
        if self._grip < 1.0:
            arrow = "→" if self._slip_drift > 0 else "←"
            canvas.create_text(
                8, 26, anchor="nw", fill="#ffd60a", font=("Helvetica", 10, "bold"),
                text=f"!! wheel slip {arrow} !!",
            )

    @staticmethod
    def _draw_car(canvas, px, py, theta, body, roof):
        """Draw a car as a rotated polygon centred at (px, py)."""
        hw = CAR_WIDTH / 2.0 * PPM
        hl = CAR_LENGTH / 2.0 * PPM
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def rot(dx, dy):
            # dy negative = towards top of screen (forward)
            return (px + dx * cos_t - dy * sin_t, py + dx * sin_t + dy * cos_t)

        canvas.create_polygon(
            *rot(-hw, -hl), *rot(hw, -hl), *rot(hw, hl), *rot(-hw, hl),
            fill=body, outline="black",
        )
        canvas.create_polygon(  # roof / windshield patch
            *rot(-hw * 0.7, -hl * 0.1), *rot(hw * 0.7, -hl * 0.1),
            *rot(hw * 0.7, hl * 0.7), *rot(-hw * 0.7, hl * 0.7),
            fill=roof, outline="",
        )

    def close(self):
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None
            self._canvas = None
