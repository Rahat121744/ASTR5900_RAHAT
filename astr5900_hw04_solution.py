
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Orbital dynamics: Euler vs Leapfrog + Voyager 2 gravity assist
#   distance = 1 AU
#   mass     = 1 Msun
#   time     = 1 yr
# so the gravitational constant is
#   G = 4*pi^2
# Velocity unit conversion:
#   1 AU/yr = 4.74047 km/s

G = 4.0 * math.pi**2
M_JUPITER = 0.0009543              # Jupiter mass in solar masses
R_JUPITER = 5.2                    # Jupiter orbital radius in AU
AU_KM = 149597870.7
YEAR_SEC = 365.25 * 24.0 * 3600.0
AUYR_TO_KMS = AU_KM / YEAR_SEC

OUTDIR = Path("astr5900_hw04_outputs")
FIGDIR = OUTDIR / "figures"
OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)

# Basic two-body utilities

def acc_sun(r: np.ndarray) -> np.ndarray:
    """Acceleration from a fixed Sun at the origin."""
    dist = np.linalg.norm(r)
    return -G * r / dist**3


def specific_energy(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Specific orbital energy epsilon = 1/2 v^2 - GM/r."""
    return 0.5 * np.sum(v * v, axis=1) - G / np.linalg.norm(r, axis=1)


def integrate_euler(r0: np.ndarray, v0: np.ndarray, dt: float, tmax: float):
    """Forward Euler integrator for a test particle around the Sun."""
    n = int(np.ceil(tmax / dt)) + 1
    t = np.arange(n) * dt
    r = np.zeros((n, 2))
    v = np.zeros((n, 2))
    r[0] = r0
    v[0] = v0

    for i in range(n - 1):
        a = acc_sun(r[i])
        r[i + 1] = r[i] + v[i] * dt
        v[i + 1] = v[i] + a * dt

    return t, r, v


def integrate_leapfrog(r0: np.ndarray, v0: np.ndarray, dt: float, tmax: float):
    """Kick-drift-kick leapfrog integrator."""
    n = int(np.ceil(tmax / dt)) + 1
    t = np.arange(n) * dt
    r = np.zeros((n, 2))
    v = np.zeros((n, 2))
    r[0] = r0
    v[0] = v0

    a0 = acc_sun(r0)
    v_half = v0 + 0.5 * a0 * dt

    for i in range(n - 1):
        r[i + 1] = r[i] + v_half * dt
        a_new = acc_sun(r[i + 1])
        v_half = v_half + a_new * dt
        v[i + 1] = v_half - 0.5 * a_new * dt

    return t, r, v



# Jupiter + spacecraft (restricted 3-body)

def jupiter_state(t: float, phi0: float):
    """Circular Jupiter orbit around a fixed Sun."""
    omega = math.sqrt(G / R_JUPITER**3)
    phi = phi0 + omega * t
    r = np.array([R_JUPITER * math.cos(phi), R_JUPITER * math.sin(phi)])
    v = np.array([-R_JUPITER * omega * math.sin(phi), R_JUPITER * omega * math.cos(phi)])
    return r, v


def spacecraft_acceleration(t: float, r_sc: np.ndarray, phi0: float) -> np.ndarray:
    """Acceleration on Voyager 2 from the Sun and Jupiter."""
    r_j, _ = jupiter_state(t, phi0)
    d_sun = np.linalg.norm(r_sc)
    d_jup_vec = r_sc - r_j
    d_jup = np.linalg.norm(d_jup_vec)

    a_sun = -G * r_sc / d_sun**3
    a_jup = -G * M_JUPITER * d_jup_vec / d_jup**3
    return a_sun + a_jup


def integrate_spacecraft_leapfrog(
    r0: np.ndarray, v0: np.ndarray, dt: float, tmax: float, phi0: float
):
    """Leapfrog integration for Voyager 2, while Jupiter follows a circular orbit."""
    n = int(np.ceil(tmax / dt)) + 1
    t = np.arange(n) * dt

    r_sc = np.zeros((n, 2))
    v_sc = np.zeros((n, 2))
    r_j = np.zeros((n, 2))
    v_j = np.zeros((n, 2))

    r_sc[0] = r0
    v_sc[0] = v0
    r_j[0], v_j[0] = jupiter_state(0.0, phi0)

    a0 = spacecraft_acceleration(0.0, r0, phi0)
    v_half = v0 + 0.5 * a0 * dt

    for i in range(n - 1):
        r_sc[i + 1] = r_sc[i] + v_half * dt
        ti = t[i + 1]
        r_j[i + 1], v_j[i + 1] = jupiter_state(ti, phi0)
        a_new = spacecraft_acceleration(ti, r_sc[i + 1], phi0)
        v_half = v_half + a_new * dt
        v_sc[i + 1] = v_half - 0.5 * a_new * dt

    return t, r_sc, v_sc, r_j, v_j


def save_figure(filename: str):
    plt.tight_layout()
    plt.savefig(FIGDIR / filename, bbox_inches="tight")
    plt.close()


def make_earth_animation(t, r_euler, r_leapfrog, gif_path: Path):
    """Animation for part 1."""
    step = 10
    idx = np.arange(0, len(t), step)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=120)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "--", lw=1, alpha=0.5, label="Exact circle")
    ax.plot(0, 0, "o", ms=7, label="Sun")

    line_e, = ax.plot([], [], label="Euler")
    line_l, = ax.plot([], [], label="Leapfrog")
    pt_e, = ax.plot([], [], "o", ms=4)
    pt_l, = ax.plot([], [], "o", ms=4)

    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    title = ax.set_title("Earth orbit comparison")
    ax.legend(loc="upper right")

    def update(k):
        j = idx[k]
        line_e.set_data(r_euler[: j + 1, 0], r_euler[: j + 1, 1])
        line_l.set_data(r_leapfrog[: j + 1, 0], r_leapfrog[: j + 1, 1])
        pt_e.set_data([r_euler[j, 0]], [r_euler[j, 1]])
        pt_l.set_data([r_leapfrog[j, 0]], [r_leapfrog[j, 1]])
        title.set_text(f"Earth orbit comparison, t = {t[j]:.2f} yr")
        return line_e, line_l, pt_e, pt_l, title

    ani = FuncAnimation(fig, update, frames=len(idx), interval=40, blit=True)
    ani.save(gif_path, writer=PillowWriter(fps=20))
    plt.close(fig)


def make_voyager_animation(t, r_sc, v_sc, r_j, gif_path: Path):
    """Animation for part 3."""
    speed = np.linalg.norm(v_sc, axis=1) * AUYR_TO_KMS
    step = 40
    idx = np.arange(0, len(t), step)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=120)
    ax.plot(0, 0, "o", ms=7, label="Sun")

    line_j, = ax.plot([], [], label="Jupiter")
    line_v, = ax.plot([], [], label="Voyager 2")
    pt_j, = ax.plot([], [], "o", ms=4)
    pt_v, = ax.plot([], [], "o", ms=4)
    txt = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")

    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.legend(loc="upper left")

    def update(k):
        j = idx[k]
        line_j.set_data(r_j[: j + 1, 0], r_j[: j + 1, 1])
        line_v.set_data(r_sc[: j + 1, 0], r_sc[: j + 1, 1])
        pt_j.set_data([r_j[j, 0]], [r_j[j, 1]])
        pt_v.set_data([r_sc[j, 0]], [r_sc[j, 1]])
        txt.set_text(f"t = {12.0 * t[j]:.1f} months\nv = {speed[j]:.2f} km/s")
        return line_j, line_v, pt_j, pt_v, txt

    ani = FuncAnimation(fig, update, frames=len(idx), interval=50, blit=True)
    ani.save(gif_path, writer=PillowWriter(fps=15))
    plt.close(fig)


def main():
    
    # Part 1: Earth in circular orbit
    
    r0 = np.array([1.0, 0.0])
    v_circ = math.sqrt(G / 1.0)
    v0 = np.array([0.0, v_circ])

    dt1 = 0.001
    t1, r1_e, v1_e = integrate_euler(r0, v0, dt1, 3.0)
    _, r1_l, v1_l = integrate_leapfrog(r0, v0, dt1, 3.0)

    speed1_e = np.linalg.norm(v1_e, axis=1) * AUYR_TO_KMS
    speed1_l = np.linalg.norm(v1_l, axis=1) * AUYR_TO_KMS
    energy1_e = specific_energy(r1_e, v1_e)
    energy1_l = specific_energy(r1_l, v1_l)

    plt.figure(figsize=(6, 6))
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    plt.plot(0, 0, "o", label="Sun")
    plt.plot(r1_e[:, 0], r1_e[:, 1], label="Euler")
    plt.plot(r1_l[:, 0], r1_l[:, 1], label="Leapfrog")
    plt.plot(np.cos(theta), np.sin(theta), "--", lw=1, label="Exact circle")
    plt.axis("equal")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.title("Part 1: Circular orbit for 3 yr")
    plt.legend()
    save_figure("part1_trajectory.png")

    plt.figure(figsize=(7, 4.5))
    plt.plot(t1, speed1_e, label="Euler")
    plt.plot(t1, speed1_l, label="Leapfrog")
    plt.xlabel("Time [yr]")
    plt.ylabel("Speed [km/s]")
    plt.title("Part 1(b): Speed vs time")
    plt.legend()
    save_figure("part1_speed.png")

    plt.figure(figsize=(7, 4.5))
    plt.plot(t1, energy1_e, label="Euler")
    plt.plot(t1, energy1_l, label="Leapfrog")
    plt.xlabel("Time [yr]")
    plt.ylabel(r"Specific energy $\epsilon$ [AU$^2$/yr$^2$]")
    plt.title("Part 1(c): Specific energy vs time")
    plt.legend()
    save_figure("part1_energy.png")

    make_earth_animation(t1, r1_e, r1_l, OUTDIR / "earth_orbit_comparison.gif")

    # Part 2: Sub-circular launch
    v0_sub = np.array([0.0, 0.8 * v_circ])

    dt2 = 0.0005
    t2, r2_e, v2_e = integrate_euler(r0, v0_sub, dt2, 5.0)
    _, r2_l, v2_l = integrate_leapfrog(r0, v0_sub, dt2, 5.0)

    energy2_e = specific_energy(r2_e, v2_e)
    energy2_l = specific_energy(r2_l, v2_l)

    plt.figure(figsize=(6, 6))
    plt.plot(0, 0, "o", label="Sun")
    plt.plot(r2_e[:, 0], r2_e[:, 1], label="Euler")
    plt.plot(r2_l[:, 0], r2_l[:, 1], label="Leapfrog")
    plt.axis("equal")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.title(r"Part 2(a): $v_0 = 0.8\,v_{\rm circ}$")
    plt.legend()
    save_figure("part2_trajectory.png")

    plt.figure(figsize=(7, 4.5))
    plt.plot(t2, energy2_e, label="Euler")
    plt.xlabel("Time [yr]")
    plt.ylabel(r"Specific energy $\epsilon$ [AU$^2$/yr$^2$]")
    plt.title("Part 2(b): Euler energy")
    plt.legend()
    save_figure("part2_energy_euler.png")

    plt.figure(figsize=(7, 4.5))
    plt.plot(t2, energy2_l, label="Leapfrog")
    plt.xlabel("Time [yr]")
    plt.ylabel(r"Specific energy $\epsilon$ [AU$^2$/yr$^2$]")
    plt.title("Part 2(b): Leapfrog energy")
    plt.legend()
    save_figure("part2_energy_leapfrog.png")

    # Part 3: Simplified Voyager 2 gravity assist
    # Using a Hohmann-like transfer speed from 1 AU to 5.2 AU.

    a_transfer = 0.5 * (1.0 + R_JUPITER)
    v_transfer = math.sqrt(G * (2.0 / 1.0 - 1.0 / a_transfer))
    t_transfer = math.pi * math.sqrt(a_transfer**3 / G)

    # Choosen Jupiter's initial phase so that Jupiter is near the
    # spacecraft's aphelion arrival point when the transfer reaches ~5.2 AU.
    omega_j = math.sqrt(G / R_JUPITER**3)
    phi_j0 = (math.pi - omega_j * t_transfer) % (2.0 * math.pi)

    r_sc0 = np.array([1.0, 0.0])
    v_sc0 = np.array([0.0, v_transfer])

    t3, r_sc, v_sc, r_j, v_j = integrate_spacecraft_leapfrog(
        r_sc0, v_sc0, 0.0005, 6.0, phi_j0
    )

    d_j = np.linalg.norm(r_sc - r_j, axis=1)
    speed3 = np.linalg.norm(v_sc, axis=1) * AUYR_TO_KMS
    i_close = np.argmin(d_j)

    plt.figure(figsize=(6.4, 6.2))
    plt.plot(0, 0, "o", label="Sun")
    plt.plot(r_j[:, 0], r_j[:, 1], label="Jupiter")
    plt.plot(r_sc[:, 0], r_sc[:, 1], label="Voyager 2")
    plt.scatter([r_sc[i_close, 0]], [r_sc[i_close, 1]], s=20, label="Closest approach")
    plt.axis("equal")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.title("Part 3(a): Simplified Voyager 2 gravity assist")
    plt.legend()
    save_figure("part3_trajectory.png")

    plt.figure(figsize=(7, 4.5))
    plt.plot(12.0 * t3, speed3)
    plt.axvline(12.0 * t3[i_close], ls="--", label=f"closest approach = {12.0 * t3[i_close]:.1f} mo")
    plt.xlabel("Time [months]")
    plt.ylabel("Voyager 2 speed [km/s]")
    plt.title("Part 3(b): Voyager 2 speed during flyby")
    plt.legend()
    save_figure("part3_speed.png")

    make_voyager_animation(t3, r_sc, v_sc, r_j, OUTDIR / "voyager2_gravity_assist.gif")

    # Printed a short numerical summary for the terminal
    print("========== PART 1 ==========")
    print(f"Euler speed range      = {speed1_e.min():.4f} to {speed1_e.max():.4f} km/s")
    print(f"Leapfrog speed range   = {speed1_l.min():.4f} to {speed1_l.max():.4f} km/s")
    print(f"Euler energy drift     = {energy1_e[-1] - energy1_e[0]:.6e}")
    print(f"Leapfrog energy drift  = {energy1_l[-1] - energy1_l[0]:.6e}")

    print("\n========== PART 2 ==========")
    print(f"Euler energy drift     = {energy2_e[-1] - energy2_e[0]:.6e}")
    print(f"Leapfrog energy drift  = {energy2_l[-1] - energy2_l[0]:.6e}")
    print(f"Leapfrog r_min         = {np.linalg.norm(r2_l, axis=1).min():.6f} AU")
    print(f"Leapfrog r_max         = {np.linalg.norm(r2_l, axis=1).max():.6f} AU")

    print("\n========== PART 3 ==========")
    print(f"Closest approach       = {d_j[i_close]:.6f} AU")
    print(f"Closest approach       = {d_j[i_close] * AU_KM:.1f} km")
    print(f"Time of closest app.   = {12.0 * t3[i_close]:.3f} months")
    print(f"Launch speed           = {speed3[0]:.3f} km/s")
    print(f"Speed at closest app.  = {speed3[i_close]:.3f} km/s")
    print(f"Final speed (6 yr)     = {speed3[-1]:.3f} km/s")

    print(f"\nAll output files saved in: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
