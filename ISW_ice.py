# -*- coding: utf-8 -*-
"""
Pedagogical Internal Solitary Wave – Ice Floe Interaction
========================================================

This script:
1) Animates a two-layer ISW experiment with tunable ice floes
2) Sweeps floe length for N = 1
3) Computes dimensionless quantities and compares with figure 3(b) of
   Carr et al. (2022) – Laboratory Experiments on Internal Solitary Waves in
Ice-Covered Waters (https://hal.science/hal-04202464v1)



@author: Matilde Bureau, Azeline Effertz, ChatGPT
"""

#%% Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import os

#%% ============================================================
# ----------------------- USER OPTIONS ------------------------
# ============================================================

save_results = False # set to True to save outputs
output_dir = "results"
animation_title = "ISW–Ice Floe Interaction"
figure_title = "Dimensionless floe speed vs floe thickness"

# Artificial speed-up of the animation (purely visual)
animation_speedup = 10

#%% ============================================================
# ------------------ PHYSICAL PARAMETERS ---------------------
# ============================================================

# Tank geometry
Lx = 10.0                 # Horizontal length of the tank (m)
Nx = 800
x = np.linspace(0, Lx, Nx) # discrete x axis

# Two-layer stratification
H1 = 0.05                 # Upper layer thickness (m)
H3 = 0.32                 # Lower layer thickness (m)

# Internal solitary wave
eta0 = 0.9*H1             # ISW amplitude (m), at most it reaches upper layer height
c0 = 0.14                 # ISW speed (m/s), from Carr et al.
dt = 0.02                 # Time step (s)
tmax = 18.0               # Maximum simulation time (s)

#%% Parameters

# ============================================================
# -------------------- ICE PROPERTIES ------------------------
# ============================================================

rho_ice = 800.0           # Ice density (kg/m^3), from Carr et al.
rho_1 = 1.025             # Salted water density (kg/m^3) - upper layer, from Carr et al.
rho_3 = 1.045             # Salted water density (kg/m^3) - lower layer, from Carr et al.
rho_w = rho_3  # Reference water density used in force scaling

delta_rho = np.abs(rho_1 - rho_3)  # Density difference between layers
g = 9.81                  # Gravity (m/s^2)

# Reduced gravity
g_prime = (delta_rho * g) / rho_3

h_f = 0.013               # Ice thickness (m), reference value, from Carr et al.


# ============================================================
# -------------------- FLOE PARAMETERS -----------------------
# ============================================================

N_init = 3                # Initial number of floes
lf_init = 0.8             # Initial floe length (m)

#%% Model and functions for ISW
# ============================================================
# ------------------- INTERNAL WAVE MODEL --------------------
# ============================================================

def solitary_wave(x, t):
    """
    Analytical internal solitary wave profile (sech^2).
    """
    return eta0 / np.cosh((x - c0*t - 1.0) / 0.4)**2

# ============================================================
# ------------------- DIAGNOSTIC TOOLS -----------------------
# ============================================================

def compute_isw_wavelength():
    """
    Compute ISW wavelength as full width at half maximum.
    """
    eta = solitary_wave(x, 4.0)
    half_max = np.max(eta) / 2
    idx = np.where(eta >= half_max)[0]
    return x[idx[-1]] - x[idx[0]]

def floe_speed_from_length(lf, lambda_isw):
    """
    Experimental linear law from Carr et al. (2022):

    cf / c = -0.61 * (lf / (2*lambda)) + 0.79
    """
    cf_over_c = -0.61 * (lf / (2 * lambda_isw)) + 0.79
    return max(cf_over_c, 0.0) * c0


#%% Animation ISW/ice
# ============================================================
# -------------------- ANIMATION SETUP -----------------------
# ============================================================

fig, ax = plt.subplots(figsize=(10, 4))
plt.subplots_adjust(bottom=0.30)

ax.set_xlim(0, Lx)
ax.set_ylim(-H3, H1)
ax.set_title(animation_title)
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")

# Two-layer fluid
ax.fill_between(x, -H3, 0, color="royalblue", alpha=0.9)
ax.fill_between(x, 0, H1, color="firebrick", alpha=0.9)

interface_line, = ax.plot(x, np.zeros_like(x), 'w', lw=2) #pycnocline
floe_patches = []

lambda_isw = compute_isw_wavelength() #compute wavelength of ISW


text_law = fig.text(0.02, 0.96, "", fontsize=10)
text_value = fig.text(0.02, 0.92, "", fontsize=10)

N_current = None
reflection_time = None
stop_after = 2.0  # seconds after first reflection, animation stops there


def init_floes(N, lf):
    """
    Initialize floe positions and velocities using experimental law.
    """
    global floe_x, floe_u, floe_patches, N_current

    # Fully remove previous floes
    for p in floe_patches:
        p.remove()
    floe_patches = []

    N_current = N

    if N == 0:
        return

    # --- Ensure non-overlapping floes ---
    gap = 0.05 * lf  # small visual gap between floes
    total_length = N * (lf + gap)

    # Distribute floes over the domain (periodic if needed)
    start_x = 0.5
    floe_x = (start_x + np.arange(N) * (lf + gap)) % Lx

    # Floe velocity prescribed by experimental linear law
    cf = floe_speed_from_length(lf, lambda_isw)
    floe_u = cf * np.ones(N)

    for xi in floe_x:
        patch = plt.Rectangle(
            (xi - lf/2, H1 - 0.01),
            lf, 0.02, color='white'
        )
        ax.add_patch(patch)
        floe_patches.append(patch)

    text_law.set_text(
        r"$\frac{c_f}{c} = -0.61\,\frac{l_f}{2\lambda_{ISW}} + 0.79$"
    )
    text_value.set_text(
        rf"Evaluated value: $c_f / c = {cf/c0:.2f}$   |   Animation speed ×{animation_speedup}"
    )


# ============================================================
# ------------------------ SLIDERS ---------------------------
# ============================================================

ax_N = plt.axes([0.15, 0.15, 0.65, 0.03])
ax_lf = plt.axes([0.15, 0.10, 0.65, 0.03])

slider_N = Slider(ax_N, "Number of floes N", 0, 50, valinit=N_init, valstep=1)
slider_lf = Slider(ax_lf, "Floe length $l_f$ (m)", 0.0, Lx, valinit=lf_init)

def slider_update(val):
    init_floes(int(slider_N.val), slider_lf.val)

slider_N.on_changed(slider_update)
slider_lf.on_changed(slider_update)

init_floes(N_init, lf_init)

# ============================================================
# -------------------- TIME EVOLUTION ------------------------
# ============================================================

def update(frame):
    """
    Advance ISW and advect floes at prescribed velocity.
    """
    global N_current, reflection_time

    # Detect change in number of floes during animation
    if int(slider_N.val) != N_current:
        init_floes(int(slider_N.val), slider_lf.val)

    # Advance internal wave (with reflection)
    t_raw = frame * dt * animation_speedup
    T_reflect = Lx / c0
    t_mod = t_raw % (2 * T_reflect)

    if t_mod <= T_reflect:
        t = t_mod
    else:
        t = 2 * T_reflect - t_mod
        if reflection_time is None:
            reflection_time = t_raw

    eta = solitary_wave(x, t)
    interface_line.set_ydata(eta)

    # Advect floes with solid-wall reflection
    for i, patch in enumerate(floe_patches):
        floe_x[i] += floe_u[i] * dt * animation_speedup

        if floe_x[i] > Lx:
            floe_x[i] = 2*Lx - floe_x[i]
            floe_u[i] *= -1
            if reflection_time is None:
                reflection_time = t_raw

        if floe_x[i] < 0:
            floe_x[i] = -floe_x[i]
            floe_u[i] *= -1
            if reflection_time is None:
                reflection_time = t_raw

        patch.set_x(floe_x[i] - slider_lf.val/2)

    # Stop animation 2 s after first reflection
    if reflection_time is not None:
        if t_raw >= reflection_time + stop_after:
            if anim.event_source.running:
                anim.event_source.stop()

    return interface_line, *floe_patches


# call animation function and run it
anim = FuncAnimation(
    fig,
    update,
    frames=int(tmax / dt),
    interval=30,
    repeat=True
)

if save_results:
    os.makedirs(output_dir, exist_ok=True)
    anim.save(f"{output_dir}/isw_ice_animation.mp4", fps=30)

plt.show()

#%% ============================================================
# -------- CRITICAL FLOE LENGTH FOR ADVECTION ARREST ----------
# ============================================================

# Solve cf = 0 from the imposed linear law
lf_crit_star = 0.79 / 0.61              # dimensionless lf / (2 lambda)
lf_crit = lf_crit_star * 2 * lambda_isw # dimensional critical floe length (m)

print("Critical floe length for advection arrest:")
print(f"  lf / (2 lambda_ISW) = {lf_crit_star:.2f}")
print(f"  lf_crit = {lf_crit:.2f} m")




#%% ============================================================
# -------- RESIDENCE TIME OF FLOES OVER THE ISW ---------------
# ============================================================

# Dimensionless floe length sweep
lf_star = np.linspace(0.05, lf_crit_star +0.05, 50) #sweep until critical floe length is reached
lf_values = lf_star * 2 * lambda_isw

T_res = np.zeros_like(lf_values)

for i, lf in enumerate(lf_values):

    cf = floe_speed_from_length(lf, lambda_isw)

    # Relative speed between ISW crest and floe
    c_rel = c0 - cf

    if c_rel <= 0:
        T_res[i] = np.inf
    else:
        # Residence time ~ ISW width / relative speed
        T_res[i] = lambda_isw / c_rel

# Dimensionless residence time
T_star = T_res / (lambda_isw / c0)

#%% ============================================================
# --------- DIMENSIONLESS RESIDENCE TIME PLOT -----------------
# ============================================================

# Increase global font size for all text elements
plt.rcParams.update({'font.size': 14}) 

plt.figure(figsize=(8, 6)) # Slightly larger figure helps with readability

# Use 'linewidth' for the plot line and 'markersize' for the points
plt.plot(lf_star, T_star, 'o-', linewidth=2.5, markersize=8)

# Use 'linewidth' for the vertical line
plt.axvline(0.79/0.61, color='r', ls='--', linewidth=2, label="Advection arrest")

# Increase labels specifically if global size isn't enough
plt.xlabel(r" $l_f / (2\lambda_{ISW})$", fontsize=16, labelpad=10)
plt.ylabel(r" $T_{res} \cdot c_0 / (\lambda_{ISW})$", fontsize=16, labelpad=10)
plt.title("Residency Time vs floe length (dimensionless)", fontsize=18, fontweight='bold')

# Increase the size of the numbers on the axes (ticks)
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
plt.tick_params(axis='both', which='minor', width=1, length=4) # For log scale readability

plt.yscale("log")
plt.grid(axis='both', which='both', alpha=0.3) # 'both' grids help log scales
plt.legend(fontsize=12, frameon=True)

plt.tight_layout() # Prevents labels from being cut off


if save_results:
    plt.savefig(f"{output_dir}/residency_time.pdf", dpi=300)
    plt.savefig(f"{output_dir}/residency_time.png", dpi=300)
    
plt.show()



#%% ============================================================
# ---- CUMULATIVE DISPLACEMENT PER ISW PASSAGE ----------------
# ============================================================

# Dimensionless floe length sweep
lf_star_disp = np.linspace(0.05, lf_crit_star +0.1, 60) #sweep until critical floe length is reached
lf_values_disp = lf_star_disp * 2 * lambda_isw

DeltaX = np.zeros_like(lf_values_disp)

for i, lf in enumerate(lf_values_disp):

    cf = floe_speed_from_length(lf, lambda_isw)
    c_rel = c0 - cf

    if c_rel <= 0:
        DeltaX[i] = np.inf
    else:
        # Displacement accumulated over one interaction
        T_res = lambda_isw / c_rel
        DeltaX[i] = cf * T_res

# Dimensionless displacement
DeltaX_star = DeltaX / lambda_isw

#%% ============================================================
# --------- DIMENSIONLESS DISPLACEMENT PLOT -------------------
# ============================================================


plt.rcParams.update({'font.size': 14}) 

plt.figure(figsize=(8, 6))


plt.plot(lf_star_disp, DeltaX_star, 'o-', linewidth=2.5, markersize=8, label="Displacement")

plt.axvline(0.79/0.61, color='r', ls='--', linewidth=2, label="Advection arrest")


plt.xlabel(r" $l_f / (2\lambda_{ISW})$", fontsize=16, labelpad=10)
plt.ylabel(r"Cumulative displacement $\Delta X / \lambda_{ISW}$", fontsize=16, labelpad=10)
plt.title("Ice transport efficiency per ISW (dimensionless) ", fontsize=18, fontweight='bold', pad=15)


plt.yscale("log")

plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=7)
plt.tick_params(axis='both', which='minor', width=1, length=4)

plt.grid(True, which="both", ls="-", alpha=0.3)

# 7. Clean legend
plt.legend(fontsize=12, frameon=True, loc='best')

plt.tight_layout()

if save_results:
    plt.savefig(f"{output_dir}/cumulative_displacement.png", dpi=300)
    plt.savefig(f"{output_dir}/cumulative_displacement.pdf", dpi=300)

plt.show()


#%% ============================================================
# -------- FLOE–WAVE LOCKING / REGIME DIAGRAM -----------------
# ============================================================

# Regime classification based on cf / c0
cf_over_c = np.array([
    floe_speed_from_length(lf, lambda_isw)/c0 for lf in lf_values_disp
])

regime = np.zeros_like(cf_over_c)

# 0: transit, 1: quasi-locked, 2: arrested
regime[cf_over_c < 0.4] = 0
regime[(cf_over_c >= 0.4) & (cf_over_c < 0.9)] = 1
regime[cf_over_c <= 0.01] = 2

#%% ============================================================
# --------- REGIME DIAGRAM PLOT -------------------------------
# ============================================================

plt.rcParams.update({'font.size': 14}) 

plt.figure(figsize=(8, 4)) 


plt.scatter(lf_star_disp, regime, c=regime, cmap="plasma", s=100, edgecolors='k', linewidth=0.5)


plt.yticks([0, 1, 2], ["Transit", "Quasi-locked", "Arrested"], fontsize=15, fontweight='medium')


plt.xlabel(r" $l_f / (2\lambda_{ISW})$", fontsize=16, labelpad=12)
plt.title("Floe–ISW interaction regimes", fontsize=18, fontweight='bold', pad=20)


plt.tick_params(axis='x', which='major', labelsize=14, width=2, length=7)
plt.gca().spines['top'].set_visible(False) # Optional: cleaner look
plt.gca().spines['right'].set_visible(False)


plt.grid(axis="x", linestyle='--', alpha=0.6)

plt.tight_layout()

if save_results:
    plt.savefig(f"{output_dir}/locking_regime_diagram.png", dpi=300)
    plt.savefig(f"{output_dir}/locking_regime_diagram.pdf", dpi=300)

plt.show()


