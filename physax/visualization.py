import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
from PIL import Image
try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from physax.config import LS, PERCENTILES
from physax.analysis import compute_snapshot_properties


# SS: use percentiles not avg -- include births
#def plot_metrics(timestamps, pop_sizes, avg_lens, filename="metrics.png"):
def plot_metrics(timestamps, pop_sizes, births, q_lens, filename="metrics.png"):
    """Plot population size and average genome length over time."""

    # SS: wider figure for longer runs
    #fig, ax1 = plt.subplots(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(20, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Cycle')
    # SS: plot births as well as pop size
    #ax1.set_ylabel('Population Size', color=color)
    ax1.set_ylabel('Population Size / ...Births', color=color)
    ax1.plot(timestamps, pop_sizes, color=color, label='Pop Size')
    # SS: plot births
    ax1.plot(timestamps, births, color=color, ls='dotted', lw=0.7, label='Births')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:red'

    # SS: plot percentiles, not avg
    # ax2.set_ylabel('Avg Genome Length', color=color)
    ax2.set_ylabel('Percentile Genome Lengths', color=color)
    #ax2.plot(timestamps, avg_lens, color=color, linestyle='--', label='Avg Len')
    for i, ls in enumerate(LS): # quartiles/percentiles
        qv = [t[i] for t in q_lens]
        ax2.plot(timestamps, qv, color=color, linestyle=ls, lw=0.8)

    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Simulation Metrics')
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved metrics plot to {filename}")


def save_grid_gif(snapshots, filename, cfg):
    """Generate a GIF of the 2D grid representation."""
    print("Generating GIF...")
    frames = []
    grid_side = int(np.ceil(np.sqrt(cfg.pop_size)))
    max_len = cfg.max_genome_len

    for i, snap in enumerate(snapshots):
        alive_mask = snap['alive']
        genome_lens = snap['genome_len']
        pad_size = grid_side * grid_side - cfg.pop_size

        alive_grid = np.pad(alive_mask, (0, pad_size), constant_values=False).reshape(grid_side, grid_side)

        if cfg.use_species_color and 'hash' in snap:
            hashes = snap['hash']
            if hashes.ndim == 2:
                hashes = hashes[:, 0]
            h = (hashes.astype(np.float32) * 0.618033988749895) % 1.0
            s = np.full_like(h, 0.8)
            v = np.full_like(h, 1.0)
            colors = np.stack([h, s, v], axis=-1)
            colors_padded = np.pad(colors, ((0, pad_size), (0, 0)), constant_values=0.0)
            hsv_grid = colors_padded.reshape(grid_side, grid_side, 3)
            rgb = mcolors.hsv_to_rgb(hsv_grid)
        else:
            len_grid = np.pad(genome_lens, (0, pad_size), constant_values=0).reshape(grid_side, grid_side)
            norm_len = np.clip(len_grid / max_len, 0, 1)
            cmap = plt.get_cmap('viridis')
            rgba = cmap(norm_len)
            rgb = rgba[..., :3]

        mask = alive_grid[..., None]
        final_img = np.where(mask, rgb, 0.0)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(final_img, interpolation='nearest')
        ax.axis('off')
        ax.set_title(f"Cycle {snap['cycle']}")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)

        frames.append(imageio.imread(buf) if imageio else np.array(Image.open(buf)))

    if imageio:
        imageio.mimsave(filename, frames, fps=10)
        print(f"Saved GIF to {filename}")
    else:
        print("imageio not installed, cannot save GIF.")


# Physis color spectrum (matching ColorRange.java)
# Anchor points: dark blue → blue → cyan → green → yellow → red → pink
PHYSIS_SPECTRUM = [
    (0, 0, 96/255),      # dark blue
    (0, 0, 1),            # blue
    (0, 1, 1),            # cyan
    (0, 1, 0),            # green
    (1, 1, 0),            # yellow
    (1, 0, 0),            # red
    (1, 128/255, 128/255) # pink
]
physis_cmap = mcolors.LinearSegmentedColormap.from_list('physis', PHYSIS_SPECTRUM, N=128)


def save_physis_view_gif(snapshots, filename, cfg, view_mode='all'):
    """Generate a GIF using Physis-style property-based coloring.

    view_mode: 'fitness', 'merit', 'age', 'all' (3-panel), or 'species' (HSV lineage).
    Snapshots must include 'alive', 'genome_len', 'executed', 'gestation_time', 'age'.
    """
    if not imageio:
        print("imageio not installed, cannot save GIF.")
        return

    print(f"Generating physis-view GIF (mode={view_mode})...")
    grid_side = int(np.ceil(np.sqrt(cfg.pop_size)))
    pad_size = grid_side * grid_side - cfg.pop_size

    # Backward-compatible: species mode uses existing HSV colors
    if view_mode == 'species':
        save_grid_gif(snapshots, filename, cfg)
        return

    # Compute properties for all snapshots
    props_list = []
    for snap in snapshots:
        eff_len, merit, fitness, fertile = compute_snapshot_properties(snap, cfg.max_genome_len)
        props_list.append({
            'effective_length': eff_len,
            'merit': merit,
            'fitness': fitness,
            'fertile': fertile,
            'age': snap['age'].astype(np.float64),
            'alive': snap['alive'],
            'cycle': snap['cycle'],
        })

    # Determine which views to render
    if view_mode == 'all':
        views = ['fitness', 'merit', 'age']
    else:
        views = [view_mode]

    # Compute running max for BY_MAX_EVER_REACHED normalization
    max_ever = {v: 0.0 for v in views}
    for p in props_list:
        for v in views:
            vals = p[v][p['alive']] if np.any(p['alive']) else np.array([0.0])
            if len(vals) > 0:
                max_ever[v] = max(max_ever[v], float(np.max(vals)))
    # Ensure non-zero
    for v in views:
        max_ever[v] = max(max_ever[v], 1.0)

    frames = []
    n_views = len(views)
    figw = 5 * n_views + 0.5
    figh = 5.0

    for pi, p in enumerate(props_list):
        fig, axes = plt.subplots(1, n_views, figsize=(figw, figh), squeeze=False)
        axes = axes[0]

        alive = p['alive']
        fertile = p['fertile']
        pop_count = int(np.sum(alive))

        for vi, view_name in enumerate(views):
            ax = axes[vi]
            vals = p[view_name]

            # Normalize to [0, 1] by max-ever
            normed = vals / max_ever[view_name]
            normed = np.clip(normed, 0, 1)

            # Build RGB grid
            # Map through physis colormap
            rgba = physis_cmap(normed)
            rgb = rgba[:, :3]  # (pop_size, 3)

            # Dead: black; alive but not fertile: dark gray
            dead_mask = ~alive
            newborn_mask = alive & ~fertile
            rgb[dead_mask] = [0.0, 0.0, 0.0]
            rgb[newborn_mask] = [64/255, 64/255, 64/255]

            # Reshape to grid
            rgb_padded = np.pad(rgb, ((0, pad_size), (0, 0)), constant_values=0.0)
            grid_img = rgb_padded.reshape(grid_side, grid_side, 3)

            ax.imshow(grid_img, interpolation='nearest', aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            # Gridlines
            for gx in range(grid_side + 1):
                ax.axvline(gx - 0.5, color='gray', linewidth=0.3, alpha=0.5)
            for gy in range(grid_side + 1):
                ax.axhline(gy - 0.5, color='gray', linewidth=0.3, alpha=0.5)
            ax.set_title(view_name.capitalize(), fontsize=11, fontweight='bold')

        fig.suptitle(f"Cycle {p['cycle']}  |  Pop: {pop_count}/{cfg.pop_size}",
                     fontsize=10, y=0.02)
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        frames.append(np.array(Image.open(buf).convert('RGB')))

        if (pi + 1) % 20 == 0:
            print(f"  Frame {pi + 1}/{len(props_list)}")

    imageio.mimsave(filename, frames, fps=10)
    print(f"Saved physis-view GIF to {filename}")


def draw_3panel_frame(snap, cfg, max_gestation):
    """Draws a single 3-panel frame and returns an RGB numpy array."""
    grid_side = int(np.ceil(np.sqrt(cfg.pop_size)))
    pad_size = grid_side * grid_side - cfg.pop_size
    
    alive = snap.get('alive', np.array([]))
    if len(alive) == 0:
        return None
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    dead_mask = ~alive
    
    # Panel 1: Unique Hash
    ax_hash = axes[0]
    hash_vals = snap.get('hash', np.zeros_like(alive, dtype=np.uint32))
    if hash_vals.ndim == 2:
        hash_vals = hash_vals[:, 0]
    
    # Map hash to RGB using 'hsv' colormap
    cmap_hash = plt.get_cmap('hsv')
    normed_hash = (hash_vals % 1000) / 1000.0
    rgba_hash = cmap_hash(normed_hash)
    rgb_hash = rgba_hash[..., :3]
    rgb_hash[dead_mask] = [0.0, 0.0, 0.0]
    
    grid_hash = np.pad(rgb_hash, ((0, pad_size), (0, 0)), constant_values=0.0).reshape(grid_side, grid_side, 3)
    ax_hash.imshow(grid_hash, interpolation='nearest', aspect='equal')
    ax_hash.set_title("Unique Genomes")
    ax_hash.axis('off')
    
    # Panel 2: Gestation Time
    ax_gest = axes[1]
    gest = snap.get('gestation_time', np.zeros_like(alive, dtype=float)).copy()
    
    status = snap.get('status', np.zeros_like(alive, dtype=int))
    
    cmap_gest = plt.get_cmap('plasma')
    
    sm_gest = plt.cm.ScalarMappable(cmap=cmap_gest, norm=plt.Normalize(vmin=0, vmax=max_gestation))
    
    # Clip gestation values between 0 and max_gestation
    clipped_gest = np.clip(gest, 0, max_gestation)
    rgba_gest = sm_gest.to_rgba(clipped_gest)
    rgb_gest = rgba_gest[..., :3]
    
    # Show gestation time only for SELF_REPLICATING (1), FERTILE (2), and NON_STANDARD (4)
    show_gest = ((status == 1) | (status == 2) | (status == 4)) & alive
    hide_gest = (~show_gest) & alive
    
    rgb_gest[hide_gest] = [0.25, 0.25, 0.25] # grey
    rgb_gest[dead_mask] = [0.0, 0.0, 0.0]
    
    grid_gest = np.pad(rgb_gest, ((0, pad_size), (0, 0)), constant_values=0.0).reshape(grid_side, grid_side, 3)
    im_gest = ax_gest.imshow(grid_gest, interpolation='nearest', aspect='equal')
    ax_gest.set_title("Gestation Time (cycles)\nSeed Ancestor GT=21")
    ax_gest.axis('off')
    
    # Add colorbar for gestation
    sm_gest.set_array([])
    cb_gest = fig.colorbar(sm_gest, ax=ax_gest, fraction=0.046, pad=0.04, format='%d')
    cb_gest.set_label("Cycles", size=16)
    cb_gest.ax.tick_params(labelsize=14)
    
    # Panel 3: Category (Status)
    ax_stat = axes[2]
    status = snap.get('status', np.zeros_like(alive, dtype=int))
    
    # Colors: 0=Grey(Unclassified), 1=Green(SelfReplicating), 2=Blue(Fertile), 3=Brown(NonFertile), 4=Orange(NonStandard)
    rgb_stat = np.zeros((*status.shape, 3), dtype=float)
    rgb_stat[status == 0] = [0.5, 0.5, 0.5]  # UNCLASSIFIED
    rgb_stat[status == 1] = [0.2, 0.8, 0.2]  # SELF_REPLICATING
    rgb_stat[status == 2] = [0.2, 0.2, 0.8]  # FERTILE
    rgb_stat[status == 3] = [0.6, 0.4, 0.2]  # NON_FERTILE
    rgb_stat[status == 4] = [1.0, 0.6, 0.0]  # NON_STANDARD
    rgb_stat[dead_mask] = [0.0, 0.0, 0.0]
    
    grid_stat = np.pad(rgb_stat, ((0, pad_size), (0, 0)), constant_values=0.0).reshape(grid_side, grid_side, 3)
    ax_stat.imshow(grid_stat, interpolation='nearest', aspect='equal')
    ax_stat.set_title("Agent Category")
    ax_stat.axis('off')
    
    # Custom legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.5, 0.5, 0.5], label='Unclassified'),
        Patch(facecolor=[0.2, 0.8, 0.2], label='Self Replicating'),
        Patch(facecolor=[0.2, 0.2, 0.8], label='Fertile'),
        Patch(facecolor=[0.6, 0.4, 0.2], label='Non Fertile'),
        Patch(facecolor=[1.0, 0.6, 0.0], label='Non Standard')
    ]
    ax_stat.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=12)
    
    fig.suptitle(f"Cycle {snap['cycle']} | Pop: {np.sum(alive)}/{cfg.pop_size}", fontsize=14)
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))

def save_custom_3panel_gif(snapshots, filename, cfg):
    """Generate a GIF with three panels:
    1) Unique Hash Color
    2) Time to reproduce (gestation_time) gradient
    3) Age gradient
    """
    if not imageio:
        print("imageio not installed, cannot save GIF.")
        return

    print("Generating custom 3-panel GIF...")
    
    # Use fixed max gestation
    max_gestation = 21.0 + 10

    frames = []
    
    for pi, snap in enumerate(snapshots[::35]):  # [::2]
        frame = draw_3panel_frame(snap, cfg, max_gestation)
        if frame is None:
            continue
            
        frames.append(frame)

        if (pi + 1) % 20 == 0:
            print(f"  Frame {pi + 1}/{len(snapshots[::35])}")

    imageio.mimsave(filename, frames, fps=10)
    print(f"Saved custom 3-panel GIF to {filename}")

def plot_gestation_and_diversity(stats, filename="gestation_diversity.png"):
    """
    Plot the average gestation period and diversity of self-replicating/fertile agents over time.
    """
    from physax.config import UNCLASSIFIED, NON_FERTILE, SELF_REPLICATING, FERTILE
    
    cycles = []
    
    # Self-Replicating stats
    avg_gest_self_rep = []
    shannon_div_self_rep = []
    total_self_rep = []
    unique_self_rep = []
    
    # Fertile stats
    avg_gest_fertile = []
    shannon_div_fertile = []
    total_fertile = []
    unique_fertile = []
    
    for chunk in stats:
        cycle = chunk['cycle']
        snap = chunk['snapshot']
        
        alive = snap['alive']
        status = snap['status']
        gest_times = snap['gestation_time']
        hashes = snap['hash']
        if hashes.ndim == 2:
            hashes = (hashes[:, 0].astype(np.int64) << 32) | (hashes[:, 1].astype(np.uint32).astype(np.int64))
        
        cycles.append(cycle)
        
        # Self-replicating and Fertile masks
        mask_sr = alive & (status == SELF_REPLICATING)
        mask_f = alive & (status == FERTILE)
        
        # --- Self-Replicating Calculations ---
        if np.any(mask_sr):
            gests_sr = gest_times[mask_sr]
            gests_sr = gests_sr[gests_sr < 2000000000]
            avg_gest_self_rep.append(np.mean(gests_sr) if len(gests_sr) > 0 else np.nan)
            
            hashes_sr = hashes[mask_sr]
            uniq_sr, counts_sr = np.unique(hashes_sr, return_counts=True)
            total_self_rep.append(np.sum(mask_sr))
            unique_self_rep.append(len(uniq_sr))
            
            probs_sr = counts_sr / np.sum(counts_sr)
            shannon_div_self_rep.append(-np.sum(probs_sr * np.log(probs_sr)))
        else:
            avg_gest_self_rep.append(np.nan)
            total_self_rep.append(0)
            unique_self_rep.append(0)
            shannon_div_self_rep.append(0.0)
            
        # --- Fertile Calculations ---
        if np.any(mask_f):
            gests_f = gest_times[mask_f]
            gests_f = gests_f[gests_f < 2000000000]
            avg_gest_fertile.append(np.mean(gests_f) if len(gests_f) > 0 else np.nan)
            
            hashes_f = hashes[mask_f]
            uniq_f, counts_f = np.unique(hashes_f, return_counts=True)
            total_fertile.append(np.sum(mask_f))
            unique_fertile.append(len(uniq_f))
            
            probs_f = counts_f / np.sum(counts_f)
            shannon_div_fertile.append(-np.sum(probs_f * np.log(probs_f)))
        else:
            avg_gest_fertile.append(np.nan)
            total_fertile.append(0)
            unique_fertile.append(0)
            shannon_div_fertile.append(0.0)
            
    if not cycles:
        print("No replicating agents found to plot gestation and diversity.")
        return
        
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Top Panel: Gestation & Shannon Diversity ---
    color_sr = '#2ca02c' # Green for Self-Replicating (aligns with GIF category color)
    color_f = '#1f77b4' # Blue for Fertile (aligns with GIF category color)
    
    # Left Y-axis: Gestation Time
    ax1.set_ylabel('Average Gestation Time (cycles)', fontsize=12)
    l1 = ax1.plot(cycles, avg_gest_self_rep, color=color_sr, lw=2, label='Self-Replicating')
    l2 = ax1.plot(cycles, avg_gest_fertile, color=color_f, lw=2, label='Fertile (but not self-replicating)')
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)
    
    # # Right Y-axis: Diversity
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Diversity (Shannon Entropy)', color='#555555', fontsize=12)
    # l3 = ax2.plot(cycles, shannon_div_self_rep, color=color_sr, lw=1.5, linestyle='--', label='Self-Replicating (Diversity)')
    # l4 = ax2.plot(cycles, shannon_div_fertile, color=color_f, lw=1.5, linestyle='--', label='Fertile (Diversity)')
    # ax2.tick_params(axis='y', labelcolor='#555555')
    
    # Combine legends from both axes
    lines = l1 + l2 #+ l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    ax1.set_title('Average Gestation Time and Genome Diversity Over Time', fontsize=14, pad=10)
    
    # --- Bottom Panel: Replicator Counts (Total vs. Unique) ---
    ax3.plot(cycles, total_self_rep, color=color_sr, lw=2, label='Total Self-Replicating')
    ax3.plot(cycles, unique_self_rep, color=color_sr, lw=1.5, linestyle='-.', label='Unique Self-Replicating (Genomes)')
    
    ax3.plot(cycles, total_fertile, color=color_f, lw=2, label='Total Fertile')
    ax3.plot(cycles, unique_fertile, color=color_f, lw=1.5, linestyle='-.', label='Unique Fertile (Genomes)')
    
    ax3.set_xlabel('Cycle', fontsize=12)
    ax3.set_ylabel('Agent Count', fontsize=12)
    ax3.tick_params(axis='y')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    ax3.set_title('Total vs. Unique Replicators by Category', fontsize=14, pad=10)
    
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved gestation, diversity and counts plot to {filename}")

def generate_all_visualizations(stats, output_dir, cfg=None):
    from pathlib import Path
    from physax.genome_analysis import analyze_and_plot_top_genomes
    from physax.config import make_config
    
    path = Path(output_dir)
    if not path.exists():
        path.mkdir(parents=True)
        
    timestamps = [s['cycle'] for s in stats]
    pop_sizes = [s['pop_size'] for s in stats]
    births = [s['births'] for s in stats]
    q_lens = [s['q_len'] for s in stats]

    plot_metrics(timestamps, pop_sizes, births, q_lens, str(path / "simulation_metrics.png"))

    snapshots = [s['snapshot'] for s in stats]
    
    if cfg is None:
        inferred_pop_size = len(snapshots[-1]['alive'])
        cfg = make_config(pop_size=inferred_pop_size)
        
    # # save_grid_gif(snapshots, str(path / "evolution.gif"), cfg)
    save_custom_3panel_gif(snapshots, str(path / "evolution_3panel.gif"), cfg)
    
    # Plot gestation and diversity over time
    plot_gestation_and_diversity(stats, str(path / "gestation_diversity.png"))
    
    # Analyze and plot top genomes
    top_hashes = analyze_and_plot_top_genomes(stats, str(path / "top_genomes_self_replicators.png"), only_self_replicators=True)
    _ = analyze_and_plot_top_genomes(stats, str(path / "top_genomes_all_fertile.png"), only_self_replicators=False)
    max_cycle = 10_000
    if stats[-1]['cycle'] > max_cycle:
        _ = analyze_and_plot_top_genomes(stats, str(path / f"top_genomes_after_{max_cycle}_cycles_self_replicators.png"), 
                                         start_cycle=max_cycle, only_self_replicators=True)
        _ = analyze_and_plot_top_genomes(stats, str(path / f"top_genomes_after_{max_cycle}_cycles_all_fertile.png"),
                                         start_cycle=max_cycle, only_self_replicators=False)
    return top_hashes

if __name__ == "__main__":
    import argparse
    import pickle
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Generate visualizations from a simulation run directory.")
    parser.add_argument("--folder", type=str, default=None, help="Folder name inside the base path")
    args = parser.parse_args()
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    base_path = Path(os.getenv("BASE_PATH", "output"))

    if args.folder:
        folder_path = base_path / args.folder
    else:
        runs = list(base_path.glob("run_*"))
        if not runs:
            print(f"Error: No run folders found in {base_path}")
            exit(1)
            
        folder_path = max(runs, key=os.path.getmtime)
        print(f"Auto-selected most recent run: {folder_path}")
    stats_file = folder_path / "simulation_stats.pkl"
    
    if not stats_file.exists():
        print(f"Error: Could not find {stats_file}")
        exit(1)
        
    print(f"Loading stats from {stats_file}...")
    with open(stats_file, "rb") as f:
        stats = pickle.load(f)
        
    if len(stats) == 0:
        print("Error: Stats file is empty.")
        exit(1)
        
    generate_all_visualizations(stats, folder_path)
    print("All visualizations generated successfully.")
