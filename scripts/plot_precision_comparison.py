import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def find_csv(fname):
    for prefix in ['', 'results/', 'results/trajectories/']:
        p = os.path.join(prefix, fname)
        if os.path.exists(p):
            return p
    return fname

files = {
    'Baseline (N=0)': find_csv('trajectory_baseline_t0.csv'),
    'Lookahead (N=1, +0.1s)': find_csv('trajectory_points_n1.csv'),
    'Lookahead (N=2, +0.2s)': find_csv('trajectory_points_n2.csv'),
    'Lookahead (N=3, +0.3s)': find_csv('trajectory_interval_0.1s.csv'),
}

data = {}
for name, fpath in files.items():
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        data[name] = df
    else:
        print(f"Warning: file {fpath} not found!")

if not data:
    print("No data found!")
    exit(1)

fig = plt.figure(figsize=(16, 12))

colors = {
    'Baseline (N=0)': '#e74c3c',         # Red
    'Lookahead (N=1, +0.1s)': '#e67e22', # Orange
    'Lookahead (N=2, +0.2s)': '#2ecc71', # Green
    'Lookahead (N=3, +0.3s)': '#2980b9', # Blue
}

ref_df = next(iter(data.values()))

# 1. 3D Trajectory Plot
ax3d = fig.add_subplot(2, 2, 1, projection='3d')
ax3d.plot(ref_df['target_x'], ref_df['target_y'], ref_df['target_z'], 'k--', linewidth=2.5, label='Target Trajectory (5.0s Figure-8)', alpha=0.9)

for name, df in data.items():
    ax3d.plot(df['pos_x'], df['pos_y'], df['pos_z'], color=colors[name], label=name, alpha=0.85, linewidth=1.8)

ax3d.set_title('3D High-Precision Tracking (5cm Corridor & Exponential Reward)', fontsize=12, fontweight='bold')
ax3d.set_xlabel('X (m)')
ax3d.set_ylabel('Y (m)')
ax3d.set_zlabel('Z (m)')
ax3d.legend(loc='upper right', fontsize=9)

# 2. Top-down 2D (X-Y) View
ax2d = fig.add_subplot(2, 2, 2)
ax2d.plot(ref_df['target_x'], ref_df['target_y'], 'k--', linewidth=2.5, label='Target Figure-8', alpha=0.9)
for name, df in data.items():
    ax2d.plot(df['pos_x'], df['pos_y'], color=colors[name], label=name, alpha=0.85, linewidth=1.8)

# Corridor envelope: +/- 5cm
theta = np.linspace(0, 2*np.pi, 200)
ax2d.set_title('Top-Down View (X-Y Flight Path)', fontsize=12, fontweight='bold')
ax2d.set_xlabel('X Position (m)')
ax2d.set_ylabel('Y Position (m)')
ax2d.axis('equal')
ax2d.legend(loc='lower center', fontsize=9)
ax2d.grid(True, linestyle=':', alpha=0.6)

# 3. Position Error vs Time
ax_err = fig.add_subplot(2, 2, 3)
for name, df in data.items():
    mean_err = (df['pos_error'] * 100).mean()
    max_err = (df['pos_error'] * 100).max()
    ax_err.plot(df['t'], df['pos_error'] * 100.0, color=colors[name], 
                label=f"{name} (mean: {mean_err:.2f}cm, max: {max_err:.2f}cm)", alpha=0.85, linewidth=1.5)

ax_err.axhline(5.0, color='red', linestyle='--', linewidth=1.5, label='5.0cm Hard Limit (Termination)')
ax_err.axhline(2.0, color='orange', linestyle=':', linewidth=1.5, label='2.0cm Zone')
ax_err.axhline(1.0, color='green', linestyle=':', linewidth=1.5, label='1.0cm Target Goal')
ax_err.set_title('Tracking Error vs Flight Time (Strict 5cm Limit)', fontsize=12, fontweight='bold')
ax_err.set_xlabel('Time (s)')
ax_err.set_ylabel('Position Error (cm)')
ax_err.set_ylim(0, 6.0)
ax_err.legend(loc='upper right', fontsize=9)
ax_err.grid(True, linestyle=':', alpha=0.6)

# 4. Error Distribution & Comparison
ax_bar = fig.add_subplot(2, 2, 4)
categories = ['< 1cm', '< 2cm', '< 5cm (Survival)']
x = np.arange(len(categories))
n_models = len(data)
width = 0.8 / max(n_models, 1)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax_bar.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

for idx, (name, df) in enumerate(data.items()):
    r_1 = (df['pos_error'] < 0.01).mean() * 100
    r_2 = (df['pos_error'] < 0.02).mean() * 100
    r_5 = (df['pos_error'] < 0.05).mean() * 100
    offset = (idx - (n_models - 1) / 2.0) * width
    rects = ax_bar.bar(x + offset, [r_1, r_2, r_5], width, label=name, color=colors[name], alpha=0.85)
    autolabel(rects)

ax_bar.set_ylabel('Percentage of Flight Time (%)')
ax_bar.set_title('High-Precision Dwell Rates Across Thresholds', fontsize=12, fontweight='bold')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(categories, fontsize=10)
ax_bar.set_ylim(0, 115)
ax_bar.legend(loc='upper left', fontsize=8)
ax_bar.grid(True, axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
os.makedirs('results', exist_ok=True)
output_img = 'results/precision_trajectory_comparison.png'
plt.savefig(output_img, dpi=200)
print(f"Saved precision comparison plot to: {output_img}")

artifact_img = '/Users/sehwanhong/.gemini/antigravity-cli/brain/d613cd3e-d018-464c-ab1b-318e50af4aa5/precision_trajectory_comparison.png'
plt.savefig(artifact_img, dpi=200)
print(f"Saved to artifact dir: {artifact_img}")
