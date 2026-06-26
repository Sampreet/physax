import os
import numpy as np
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from physax.config import BLANK

def init_wandb(cfg, total_cycles):
    if not WANDB_AVAILABLE:
        print("WARNING: wandb not installed. Run: pip install wandb")
        return False
        
    cfg_dict = {}
    for k in dir(cfg):
        if not k.startswith('_') and not callable(getattr(cfg, k)):
            cfg_dict[k] = getattr(cfg, k)
    cfg_dict['total_cycles'] = total_cycles
    
    seed = getattr(cfg, 'seed', 'unknown')
    
    run = wandb.init(
        project="physis-jax",
        config=cfg_dict
    )
    
    if run and run.name:
        run.name = f"run_{total_cycles}_cycles_seed_{seed}_{run.name}"
    return True

def log_cycle_metrics(start, log_interval, stats):
    pop_sizes_arr = np.array(stats['pop_size'])
    births_arr = np.array(stats['births'])
    q_len_arr = np.array(stats['q_genome_len'])
    for i in range(log_interval):
        step = start + i + 1
        wandb.log({
            "cycle": step,
            "population/size": pop_sizes_arr[i],
            "population/births": births_arr[i],
            "genome/len_q5": q_len_arr[i, 0],
            "genome/len_q50": q_len_arr[i, 3],
        }, step=step)

def log_snapshot_and_diversity(cycle_num, snapshot, cfg):
    from physax.genome_analysis import compute_diversity_stats
    from physax.visualization import draw_3panel_frame
    from physax.config import BLANK, UNCLASSIFIED, SELF_REPLICATING, FERTILE, NON_FERTILE, NON_STANDARD
    
    div_stats = compute_diversity_stats(snapshot)
    wandb_dict = {f"diversity/{k}": v for k, v in div_stats.items()}
    
    alive_mask = snapshot['alive']
    if np.any(alive_mask):
        # 1. Genome Length Distribution (Histogram)
        lengths = snapshot['genome_len'][alive_mask]
        wandb_dict["population/genome_length_hist"] = wandb.Histogram(lengths)
        
        # 2. Gestation Time Distribution (Histogram)
        gestations = snapshot['gestation_time'][alive_mask]
        valid_gestations = gestations[gestations < 2147483647]
        if len(valid_gestations) > 0:
            wandb_dict["population/gestation_time_hist"] = wandb.Histogram(valid_gestations)
            
        # 3. Agents in population split by classes
        statuses = snapshot['status'][alive_mask]
        status_names = {
            UNCLASSIFIED: "unclassified",
            SELF_REPLICATING: "self_replicating",
            FERTILE: "fertile",
            NON_FERTILE: "non_fertile",
            NON_STANDARD: "non_standard"
        }
        
        unique_status, counts = np.unique(statuses, return_counts=True)
        for s, count in zip(unique_status, counts):
            name = status_names.get(s, f"class_{s}")
            wandb_dict[f"classes/{name}"] = count
            
        # Ensure all classes are logged
        for s, name in status_names.items():
            if s not in unique_status:
                wandb_dict[f"classes/{name}"] = 0
    
    max_gestation = 21.0 + 10
    img_arr = draw_3panel_frame(snapshot, cfg, max_gestation)
    if img_arr is not None:
        wandb_dict["population/screenshot"] = wandb.Image(img_arr, caption=f"Cycle {cycle_num}")
    
    wandb.log(wandb_dict, step=cycle_num)

def log_frequency_reports(cycle_num, snapshot, global_self_replicating, global_fertile, output_dir):
    from physax.genome_analysis import format_genome
    
    def format_decoded_genome(gen_arr, length):
        from physax.config import OP_NAMES, N_OPERANDS, BLANK
        i = 0
        decoded = []
        genome = [int(x) for x in gen_arr[:length]]
        
        in_instruction = False
        past_sep = False
        while i < len(genome):
            val = genome[i]
            
            if val == BLANK:
                decoded.append("BLANK")
                i += 1
                continue
                
            if val == 36: # SEP
                decoded.append(OP_NAMES.get(36))
                in_instruction = False
                past_sep = True
                i += 1
                continue
                
            if past_sep:
                decoded.append(str(val))
                i += 1
                continue
                
            if val == 34: # I
                decoded.append(OP_NAMES.get(34))
                in_instruction = True
                i += 1
                continue
                
            if val in [31, 32, 33, 35] and not in_instruction:
                decoded.append(OP_NAMES.get(val))
                i += 1
                continue
                
            opcode = abs(val) % 44
            n_args = int(N_OPERANDS[opcode])
            
            args = []
            for j in range(n_args):
                if i + 1 + j < len(genome):
                    args.append(str(genome[i + 1 + j]))
                else:
                    break
                    
            op_str = OP_NAMES.get(opcode, str(val))
            if args:
                op_str += f" {' '.join(args)}"
                
            decoded.append(op_str)
            i += 1 + len(args)
            
        # Format with line breaks
        lines = []
        line = []
        for item in decoded:
            if item == "I" or item == "SEP":
                if line:
                    lines.append("  " + ", ".join(line))
                line = [item]
            elif item == "BLANK":
                if line and "BLANK... (padded to end)" not in line:
                    if line:
                        lines.append("  " + ", ".join(line))
                    line = ["BLANK... (padded to end)"]
            else:
                line.append(str(item))
                if len(line) > 15:
                    lines.append("  " + ", ".join(line))
                    line = []
        if line:
            lines.append("  " + ", ".join(line))
        return "\n".join(lines)

    def generate_report(genomes_dict, title):
        alive_hashes = snapshot['hash'][snapshot['alive']]
        if alive_hashes.ndim == 2:
            alive_hashes_64 = (alive_hashes[:, 0].astype(np.int64) << 32) | (alive_hashes[:, 1].astype(np.uint32).astype(np.int64))
        else:
            alive_hashes_64 = alive_hashes
        
        unique_h, counts = np.unique(alive_hashes_64, return_counts=True)
        freq_dict = dict(zip(unique_h, counts))
        
        def get_freq(h):
            h_64 = (np.int64(h[0]) << 32) | np.int64(np.uint32(h[1]))
            return freq_dict.get(h_64, 0)
            
        sorted_hashes = sorted(genomes_dict.keys(), key=get_freq, reverse=True)
        top_40 = sorted_hashes[:40]
        report = f"# {title} (Cycle {cycle_num})\n\n"
        
        table_data = []
        idx = 1
        for h in top_40:
            freq = get_freq(h)
            if freq == 0: continue
            g = genomes_dict[h]
            g_len = int(np.sum(g != BLANK))
            
            mask = snapshot['alive'] & (snapshot['hash'][:, 0] == h[0]) & (snapshot['hash'][:, 1] == h[1])
            gest = np.min(snapshot['gestation_time'][mask]) if np.any(mask) else "N/A"
            
            decoded_str = format_decoded_genome(g, g_len)
            
            # Markdown File Format
            report += f"### Genome {idx}: {h[0]}_{h[1]}\n"
            report += f"- **DB Gestation Time**: {gest} cycles\n\n"
            report += "#### Parent Genome\n```\n" + decoded_str + "\n```\n\n"
            report += "#### Child Genome\n```\n" + decoded_str + "\n```\n\n"
            report += "---\n\n"
            
            # Table format for wandb
            table_data.append([str(h), freq, gest, decoded_str.replace('\n', '  ')])
            idx += 1
        
        table = wandb.Table(columns=["Hash", "Frequency", "Gestation Time", "Genome"], data=table_data)
        return report, table
    
    sr_report, sr_table = generate_report(global_self_replicating, "Top 40 Self-Replicating Genomes")
    f_report, f_table = generate_report(global_fertile, "Top 40 Fertile Genomes")
    
    sr_path = os.path.join(output_dir, f"report_cycle_{cycle_num}_self_replicators.md")
    f_path = os.path.join(output_dir, f"report_cycle_{cycle_num}_fertile.md")
    with open(sr_path, "w") as f:
        f.write(sr_report)
    with open(f_path, "w") as f:
        f.write(f_report)
        
    # Log wandb tables for dashboard view
    wandb.log({
        "reports/self_replicators_table": sr_table,
        "reports/fertile_table": f_table
    }, step=cycle_num)
    
    # Log the files as an artifact
    artifact = wandb.Artifact(f"reports_cycle_{cycle_num}", type="report")
    artifact.add_file(sr_path)
    artifact.add_file(f_path)
    wandb.log_artifact(artifact)

def finish_wandb():
    if WANDB_AVAILABLE:
        wandb.finish()
