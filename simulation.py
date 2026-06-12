"""
Endless Racer control simulation.

Provides a small Tkinter front-end for the ``EndlessRacerEnv`` environment
(``racer_env.py``) -- a top-down, vertically scrolling endless racer with
random wheel slippage and drift -- with three control modes:

1. Manual control  -- steer the car with the arrow keys.
2. PID control     -- a PID controller steers the car; tune Kp/Ki/Kd live
                       with sliders. With obstacles on, the PID setpoint
                       (desired heading) is computed from the bearings of
                       the visible obstacles.
3. RL control      -- train (or load) a Stable-Baselines3 PPO agent and
                       watch it drive.

Each mode can be run with **obstacles on** (dodge slower traffic cars) or
**obstacles off** (just stay in the lane), toggled on the main menu.

The actual control algorithms live in ``controllers.py`` so they can be
implemented/edited independently (see ``notebooks/``).

Run with::

    python simulation.py
"""

import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import racer_env
from racer_env import EndlessRacerEnv
from controllers import (
    PIDController,
    compute_desired_heading,
    get_manual_action,
    get_rl_action,
    load_rl_agent,
    train_rl_agent,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


def checkpoint_path(obstacles_on):
    name = "ppo_racer_obstacles_on" if obstacles_on else "ppo_racer_obstacles_off"
    return os.path.join(CHECKPOINT_DIR, name + ".zip")


# Delay (ms) between simulation steps, driven by Tkinter's event loop.
STEP_DELAY_MS = int(racer_env.DT * 1000)


class RacerSimulatorApp:
    """Tkinter application that ties the GUI to the racer environment."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Endless Racer Control Simulator")
        self.root.resizable(False, False)

        self.env = None
        self.obs = None
        self.info = None
        self.episode_reward = 0.0
        self.pressed_keys = {}
        self.after_id = None
        self.episode_end_frame = None
        self.canvas = None

        self.obstacles_var = tk.BooleanVar(value=False)
        self.pid = PIDController()
        self.rl_model = None
        self._training_thread = None

        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)

        self.show_main_menu()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def run(self):
        self.root.mainloop()

    def _clear(self):
        self._stop_loop()
        for widget in self.root.winfo_children():
            widget.destroy()
        self.episode_end_frame = None
        self.canvas = None

    def _stop_loop(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _close_env(self):
        self._stop_loop()
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    def _make_env(self):
        return EndlessRacerEnv(obstacles=self.obstacles_var.get())

    def _on_key_press(self, event):
        if event.keysym == "Left":
            self.pressed_keys["left"] = True
        elif event.keysym == "Right":
            self.pressed_keys["right"] = True
        elif event.keysym in ("r", "R"):
            self.pressed_keys["reset"] = True
        elif event.keysym == "Escape":
            self.pressed_keys["quit"] = True

    def _on_key_release(self, event):
        if event.keysym == "Left":
            self.pressed_keys["left"] = False
        elif event.keysym == "Right":
            self.pressed_keys["right"] = False

    # ------------------------------------------------------------------
    # Main menu
    # ------------------------------------------------------------------
    def show_main_menu(self):
        self._clear()
        self._close_env()

        frame = tk.Frame(self.root, padx=30, pady=24)
        frame.pack()

        tk.Label(
            frame, text="Endless Racer Control Simulator",
            font=("Helvetica", 16, "bold"),
        ).pack(pady=(0, 4))
        tk.Label(
            frame,
            text=(
                "A top-down endless racer with random wheel slippage and\n"
                "drift. Keep the car in the lane -- and out of the traffic."
            ),
            justify="center",
        ).pack(pady=(0, 14))

        tk.Checkbutton(
            frame, text="Obstacles (traffic cars)", variable=self.obstacles_var,
            font=("Helvetica", 11),
        ).pack(pady=(0, 14))

        for text, cmd in (
            ("Manual Control", self.show_manual_mode),
            ("PID Control", self.show_pid_mode),
            ("Reinforcement Learning Control", self.show_rl_mode),
        ):
            tk.Button(frame, text=text, width=32, command=cmd).pack(pady=4)

        tk.Label(
            frame,
            text="Keys during simulation:  \u2190/\u2192 steer (manual)   R reset   Esc menu",
            fg="grey",
        ).pack(pady=(14, 0))

    # ------------------------------------------------------------------
    # Common simulation scaffolding
    # ------------------------------------------------------------------
    def _build_sim_screen(self, title, controls_builder=None):
        """Canvas on the left, a control/status panel on the right."""
        self._clear()

        outer = tk.Frame(self.root)
        outer.pack()

        self.canvas = tk.Canvas(
            outer, width=racer_env.RENDER_WIDTH, height=racer_env.RENDER_HEIGHT,
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0)

        panel = tk.Frame(outer, padx=14, pady=12, width=260)
        panel.grid(row=0, column=1, sticky="ns")
        panel.grid_propagate(False)

        tk.Label(panel, text=title, font=("Helvetica", 13, "bold")).pack(pady=(0, 2))
        mode = "obstacles ON" if self.obstacles_var.get() else "obstacles OFF"
        tk.Label(panel, text=f"({mode})", fg="grey").pack(pady=(0, 8))

        self.status_var = tk.StringVar(value="")
        tk.Label(panel, textvariable=self.status_var, justify="left").pack(pady=(0, 8))

        if controls_builder is not None:
            controls_builder(panel)

        tk.Button(panel, text="Back to Menu", command=self.show_main_menu).pack(
            side="bottom", pady=(8, 0), fill="x"
        )
        self.panel = panel
        return panel

    def _start_episode(self):
        self.episode_reward = 0.0
        self.pressed_keys = {}
        self.obs, self.info = self.env.reset()
        self.pid.reset()
        if self.episode_end_frame is not None:
            self.episode_end_frame.destroy()
            self.episode_end_frame = None

    def _update_status(self):
        self.status_var.set(
            f"distance: {self.info['distance']:8.1f} m\n"
            f"heading:  {self.info['heading']:+8.3f} rad\n"
            f"offset:   {self.info['lateral_offset']:+8.2f} m\n"
            f"reward:   {self.episode_reward:8.1f}"
        )

    def _handle_episode_end(self, restart_cmd):
        self._stop_loop()
        self.episode_end_frame = tk.Frame(self.panel, pady=8)
        self.episode_end_frame.pack()
        tk.Label(
            self.episode_end_frame,
            text=f"Episode over!\nTotal reward: {self.episode_reward:.1f}",
            font=("Helvetica", 11, "bold"),
        ).pack(pady=(0, 6))
        tk.Button(
            self.episode_end_frame, text="Start New Episode", command=restart_cmd
        ).pack()

    def _common_step(self, action, restart_cmd):
        """Step the env, draw, and handle reset/quit/episode-end. Returns
        True if the simulation loop should continue."""
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.episode_reward += reward
        self.env.draw(self.canvas)
        self._update_status()

        if self.pressed_keys.pop("quit", False):
            self.show_main_menu()
            return False
        if self.pressed_keys.pop("reset", False):
            restart_cmd()
            return False
        if terminated or truncated:
            self._handle_episode_end(restart_cmd)
            return False
        return True

    # ------------------------------------------------------------------
    # Manual control
    # ------------------------------------------------------------------
    def show_manual_mode(self):
        def controls(panel):
            btns = tk.Frame(panel)
            btns.pack(pady=4)
            left = tk.Button(btns, text="\u25c0 Left", width=8)
            right = tk.Button(btns, text="Right \u25b6", width=8)
            left.grid(row=0, column=0, padx=3)
            right.grid(row=0, column=1, padx=3)
            for btn, key in ((left, "left"), (right, "right")):
                btn.bind("<ButtonPress>", lambda _e, k=key: self.pressed_keys.__setitem__(k, True))
                btn.bind("<ButtonRelease>", lambda _e, k=key: self.pressed_keys.__setitem__(k, False))

        self._build_sim_screen("Manual Control", controls)
        self.env = self._make_env()
        self._restart_manual()

    def _restart_manual(self):
        self._stop_loop()
        self._start_episode()
        self._manual_loop()

    def _manual_loop(self):
        action = get_manual_action(self.env.action_space, self.pressed_keys)
        if self._common_step(action, self._restart_manual):
            self.after_id = self.root.after(STEP_DELAY_MS, self._manual_loop)

    # ------------------------------------------------------------------
    # PID control
    # ------------------------------------------------------------------
    def show_pid_mode(self):
        def controls(panel):
            self.kp_var = tk.DoubleVar(value=3.0)
            self.ki_var = tk.DoubleVar(value=0.0)
            self.kd_var = tk.DoubleVar(value=0.4)
            for label, var, hi in (
                ("Kp", self.kp_var, 10.0),
                ("Ki", self.ki_var, 5.0),
                ("Kd", self.kd_var, 2.0),
            ):
                tk.Label(panel, text=label).pack()
                tk.Scale(
                    panel, variable=var, from_=0.0, to=hi, resolution=0.01,
                    orient="horizontal", length=200,
                ).pack()
            self.setpoint_var = tk.StringVar(value="desired heading: +0.000")
            tk.Label(panel, textvariable=self.setpoint_var, fg="grey").pack(pady=(6, 0))

        self._build_sim_screen("PID Control", controls)
        self.env = self._make_env()
        self._restart_pid()

    def _restart_pid(self):
        self._stop_loop()
        self._start_episode()
        self._pid_loop()

    def _pid_loop(self):
        self.pid.set_gains(self.kp_var.get(), self.ki_var.get(), self.kd_var.get())
        desired = compute_desired_heading(
            self.obs, self.info, self.obstacles_var.get()
        )
        self.pid.set_setpoint(desired)
        self.setpoint_var.set(f"desired heading: {desired:+.3f}")
        action = self.pid.compute_action(self.obs, self.env.action_space, self.env.dt)
        if self._common_step(action, self._restart_pid):
            self.after_id = self.root.after(STEP_DELAY_MS, self._pid_loop)

    # ------------------------------------------------------------------
    # RL control
    # ------------------------------------------------------------------
    def show_rl_mode(self):
        path = checkpoint_path(self.obstacles_var.get())
        if os.path.exists(path):
            self._clear()
            frame = tk.Frame(self.root, padx=30, pady=24)
            frame.pack()
            tk.Label(
                frame, text="A trained model already exists for this mode.",
                font=("Helvetica", 12),
            ).pack(pady=(0, 12))
            tk.Button(
                frame, text="Use existing trained model", width=30,
                command=lambda: self._run_rl_agent(path),
            ).pack(pady=4)
            tk.Button(
                frame, text="Train a new model (overwrite)", width=30,
                command=self._show_rl_training_config,
            ).pack(pady=4)
            tk.Button(frame, text="Back to Menu", command=self.show_main_menu).pack(
                pady=(12, 0)
            )
        else:
            self._show_rl_training_config()

    def _show_rl_training_config(self):
        self._clear()
        frame = tk.Frame(self.root, padx=30, pady=24)
        frame.pack()
        tk.Label(
            frame, text="PPO Training Hyperparameters",
            font=("Helvetica", 13, "bold"),
        ).pack(pady=(0, 10))

        defaults = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "total_timesteps": 150_000,
        }
        self.hp_vars = {}
        grid = tk.Frame(frame)
        grid.pack()
        for row, (key, val) in enumerate(defaults.items()):
            tk.Label(grid, text=key).grid(row=row, column=0, sticky="e", padx=4, pady=2)
            var = tk.StringVar(value=str(val))
            tk.Entry(grid, textvariable=var, width=12).grid(row=row, column=1, pady=2)
            self.hp_vars[key] = var

        self.progress = ttk.Progressbar(frame, length=260, maximum=1.0)
        self.progress.pack(pady=(12, 4))
        self.train_status = tk.StringVar(value="")
        tk.Label(frame, textvariable=self.train_status).pack()

        self.train_button = tk.Button(frame, text="Start Training", command=self._start_training)
        self.train_button.pack(pady=8)
        tk.Button(frame, text="Back to Menu", command=self.show_main_menu).pack()

    def _start_training(self):
        try:
            hp = {
                "learning_rate": float(self.hp_vars["learning_rate"].get()),
                "n_steps": int(self.hp_vars["n_steps"].get()),
                "batch_size": int(self.hp_vars["batch_size"].get()),
                "gamma": float(self.hp_vars["gamma"].get()),
                "gae_lambda": float(self.hp_vars["gae_lambda"].get()),
                "ent_coef": float(self.hp_vars["ent_coef"].get()),
            }
            total_timesteps = int(self.hp_vars["total_timesteps"].get())
        except ValueError:
            messagebox.showerror("Invalid input", "Hyperparameters must be numeric.")
            return

        self.train_button.configure(state="disabled")
        self.train_status.set("Training... (runs in the background)")
        obstacles_on = self.obstacles_var.get()
        path = checkpoint_path(obstacles_on)

        from stable_baselines3.common.callbacks import BaseCallback

        app = self

        class ProgressCallback(BaseCallback):
            def _on_step(self):
                frac = self.num_timesteps / total_timesteps
                app.root.after(0, lambda: app.progress.configure(value=frac))
                return True

        def work():
            env = EndlessRacerEnv(obstacles=obstacles_on)
            try:
                train_rl_agent(env, hp, total_timesteps, path, callback=ProgressCallback())
            finally:
                env.close()
            app.root.after(0, lambda: app._on_training_done(path))

        self._training_thread = threading.Thread(target=work, daemon=True)
        self._training_thread.start()

    def _on_training_done(self, path):
        self.train_status.set("Training complete!")
        self._run_rl_agent(path)

    def _run_rl_agent(self, path):
        self._build_sim_screen("RL Control (PPO)")
        self.env = self._make_env()
        self.rl_model = load_rl_agent(path, env=None)
        self._restart_rl()

    def _restart_rl(self):
        self._stop_loop()
        self._start_episode()
        self._rl_loop()

    def _rl_loop(self):
        action = get_rl_action(self.rl_model, self.obs)
        if self._common_step(action, self._restart_rl):
            self.after_id = self.root.after(STEP_DELAY_MS, self._rl_loop)


if __name__ == "__main__":
    app = RacerSimulatorApp()
    app.run()
