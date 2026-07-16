"""Weights & Biases logging and end-of-run reports.

Per-cycle metrics, genome-structure percentiles, language properties, mutational
robustness, snapshot+diversity images, and the Markdown top-genome frequency
reports (which re-simulate genomes and write them to disk). Called from
`Model.run_simulation`; degrades gracefully when `wandb` is not installed.
"""
import os
import numpy as np
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from physis.analysis._util import fold_hash


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

def log_genome_part_distribution(cycle_num, pcts, q_registers, q_instructions, q_ops_per_instr):
    # SS: percentile distribution over the living population of each genome structural part:
    # registers/state elements, instructions, and op-codes per instruction. Logging the spread
    # (p5..p95) rather than just the median exposes shifts in the tails as variants sweep, which
    # a flat median hides. Keys use the percentile value, e.g. genome/registers_p50.
    pcts = np.array(pcts)
    metrics = {}
    for name, q in (("registers", q_registers),
                    ("instructions", q_instructions),
                    ("ops_per_instruction", q_ops_per_instr)):
        q = np.array(q)
        for p, v in zip(pcts, q):
            metrics[f"genome/{name}_p{int(p)}"] = v
    wandb.log(metrics, step=cycle_num)

def log_language_properties(cycle_num, lang_stats, log_hist=False):
    # SS: properties of the emergent "language" -- both the genotype (raw token
    # sequence) and the genotype->phenotype mapping (each genome's header, which
    # defines its own registers and instruction set). Scalars are logged every
    # chunk so trends are visible; the opcode/symbol bar charts are heavier media
    # and only logged when log_hist is set (same cadence as the frequency reports).
    from physis.sim.config import OP_NAMES, UP_IS_SIZE

    metrics = {f"language/{k}": v for k, v in lang_stats['scalars'].items()}

    if log_hist:
        for key, counts in (("opcode_usage", lang_stats['opcode_counts']),
                            ("genome_symbols", lang_stats['symbol_counts'])):
            rows = [[OP_NAMES.get(i, str(i)), int(counts[i])] for i in range(UP_IS_SIZE)]
            table = wandb.Table(columns=["op", "count"], data=rows)
            metrics[f"language/{key}"] = wandb.plot.bar(
                table, "op", "count", title=f"{key} (cycle {cycle_num})"
            )

    wandb.log(metrics, step=cycle_num)

def log_mutational_robustness(cycle_num, rob_stats):
    # SS: point-mutational robustness of the living self-replicators -- the
    # fraction of single-substitution mutants whose replication phenotype
    # survives. baseline_* report how the unmutated genomes fare under the same
    # step budget; if baseline_self_replicates drops well below 1 the budget is
    # too short and the robustness numbers are underestimates.
    metrics = {
        "robustness/divide": rob_stats['robustness_divide'],
        "robustness/self_replicate": rob_stats['robustness_self_replicate'],
        "robustness/baseline_divides": rob_stats['baseline_divides'],
        "robustness/baseline_self_replicates": rob_stats['baseline_self_replicates'],
        "robustness/n_genomes": rob_stats['n_genomes'],
    }
    per_genome = rob_stats.get('per_genome_robustness_self_replicate')
    if per_genome is not None and len(per_genome) > 0:
        metrics["robustness/per_genome_hist"] = wandb.Histogram(np.asarray(per_genome))
    wandb.log(metrics, step=cycle_num)

def log_snapshot_and_diversity(cycle_num, snapshot, cfg):
    from physis.analysis.genome_stats import compute_diversity_stats
    from physis.analysis.visualization import draw_3panel_frame
    from physis.sim.config import BLANK, UNCLASSIFIED, SELF_REPLICATING, FERTILE, NON_FERTILE, NON_STANDARD
    
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
            wandb_dict["population/avg_gestation_time"] = np.mean(valid_gestations)
            
        # 2.5 Per-class histograms and average gestation times
        for label, status_val in [("self_rep", SELF_REPLICATING), ("fertile", FERTILE)]:
            class_mask = alive_mask & (snapshot['status'] == status_val)
            if np.any(class_mask):
                c_lengths = snapshot['genome_len'][class_mask]
                wandb_dict[f"population/genome_length_hist_{label}"] = wandb.Histogram(c_lengths)
                
                c_gestations = snapshot['gestation_time'][class_mask]
                c_valid_gestations = c_gestations[c_gestations < 2147483647]
                if len(c_valid_gestations) > 0:
                    wandb_dict[f"population/gestation_time_hist_{label}"] = wandb.Histogram(c_valid_gestations)
                    wandb_dict[f"population/avg_gestation_time_{label}"] = np.mean(c_valid_gestations)
            
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

def log_frequency_reports(cycle_num, snapshot, global_self_replicating, global_fertile, output_dir, cfg):
    from physis.analysis.genome_stats import format_genome
    
    def format_decoded_genome(gen_arr, length):
        from physis.sim.config import OP_NAMES, N_OPERANDS, BLANK
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
        import jax
        import jax.numpy as jnp
        import numpy as np
        from physis.sim.agent import Agent
        from physis.sim.config import UNCLASSIFIED, BLANK
        from physis.analysis.genome_evaluator import run_batch_until_division
        
        alive_hashes = snapshot['hash'][snapshot['alive']]
        if alive_hashes.ndim == 2:
            alive_hashes_64 = fold_hash(alive_hashes)
        else:
            alive_hashes_64 = alive_hashes

        unique_h, counts = np.unique(alive_hashes_64, return_counts=True)
        freq_dict = dict(zip(unique_h, counts))

        def get_freq(h):
            return freq_dict.get(fold_hash(h), 0)
            
        sorted_hashes = sorted(genomes_dict.keys(), key=get_freq, reverse=True)
        top_40 = sorted_hashes[:40]
        report = f"# {title} (Cycle {cycle_num})\n\n"
        
        table_data = []
        
        # Collect valid genomes to evaluate
        valid_genomes = []
        for h in top_40:
            freq = get_freq(h)
            if freq == 0: continue
            g = genomes_dict[h]
            g_len = int(np.sum(g != BLANK))
            valid_genomes.append((h, freq, g, g_len))
            
        if valid_genomes:
            agents_list = []
            for h, freq, g, g_len in valid_genomes:
                p_genome = np.full(cfg.max_genome_len, BLANK, dtype=np.int32)
                p_genome[:min(g_len, cfg.max_genome_len)] = g[:min(g_len, cfg.max_genome_len)]
                p_agent = Agent.init_organism(
                    jnp.array(p_genome, dtype=jnp.int32), 
                    jnp.int32(g_len), jnp.int32(-1), jnp.int32(UNCLASSIFIED), jnp.int32(-1), cfg
                )
                agents_list.append(p_agent)
                
            agents_batch = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *agents_list)
            
            # Find max gestation time among these to set max_steps
            max_gest = 100
            for h, freq, g, g_len in valid_genomes:
                mask = snapshot['alive'] & (snapshot['hash'][:, 0] == h[0]) & (snapshot['hash'][:, 1] == h[1])
                gest = np.min(snapshot['gestation_time'][mask]) if np.any(mask) else 100
                if gest != "N/A" and gest < 2000000000:
                    max_gest = max(max_gest, int(gest))
            
            global_max_steps = (max_gest * cfg.steps_per_update) + 1000
            keys_batch = jax.random.split(jax.random.PRNGKey(42), len(agents_list))
            
            # Run simulation
            final_agents, final_steps, _, final_finished_steps = run_batch_until_division(agents_batch, jnp.int32(global_max_steps), keys_batch, cfg)
            
            final_agents_list = [jax.tree_util.tree_map(lambda x: x[i], final_agents) for i in range(len(agents_list))]
            final_finished_steps_list = [int(final_finished_steps[i]) for i in range(len(agents_list))]
            
            idx = 1
            for (h, freq, g, g_len), final_agent, recalc_steps in zip(valid_genomes, final_agents_list, final_finished_steps_list):
                mask = snapshot['alive'] & (snapshot['hash'][:, 0] == h[0]) & (snapshot['hash'][:, 1] == h[1])
                gest = np.min(snapshot['gestation_time'][mask]) if np.any(mask) else "N/A"
                
                decoded_str = format_decoded_genome(g, g_len)
                
                recalc_cycles = (recalc_steps + cfg.steps_per_update - 1) // cfg.steps_per_update if recalc_steps > 0 else 0
                
                report += f"### Genome {idx}: {h[0]}_{h[1]}\n"
                report += f"- **DB Gestation Time**: {gest} cycles\n"
                
                if recalc_steps != -1:
                    report += f"- **Recalculated Gestation Time**: {recalc_steps} steps (~{recalc_cycles} cycles)\n\n"
                else:
                    report += f"- **Recalculated Gestation Time**: DNF (Did not finish within {global_max_steps} steps)\n\n"
                    
                report += "#### Parent Genome\n```\n" + decoded_str + "\n```\n\n"
                
                if bool(final_agent.has_child):
                    child_len = int(final_agent.child_len)
                    child_genome = np.array(final_agent.child)
                    decoded_child = format_decoded_genome(child_genome, child_len)
                    report += "#### Child Genome\n```\n" + decoded_child + "\n```\n\n"
                else:
                    report += "#### Child Genome\n*(No child produced)*\n\n"
                    
                report += "---\n\n"
                
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
