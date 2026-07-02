import glob
import numpy as np

from physax.config import BLANK, SEP, I, R, S, Q, OP_NAMES, N_OPERANDS


# ---------------------------------------------------------------------------
# Lineage reconstruction
#
# These helpers consume the per-chunk genealogy edge files written by
# Model.run_simulation(..., track_lineage=True) and the population snapshots it
# returns. Together they answer Susan's two questions:
#   1. "trace parents of self-repls"  -> trace_ancestry / self_replicators
#   2. "how the program is being changed without killing the individual"
#      -> conservation_map, over the genomes of a surviving lineage.
# ---------------------------------------------------------------------------

def load_lineage(lineage_dir):
    """Load every edges_*.npz file in lineage_dir into one flat edge list.

    Returns a dict with 1-D arrays birth_cycle / child_id / parent_id (sorted by
    birth_cycle), plus parent_of: {child_id -> (parent_id, birth_cycle)}.
    """
    files = sorted(glob.glob(f"{lineage_dir}/edges_*.npz"))
    bc, ci, pi = [], [], []
    for f in files:
        d = np.load(f)
        bc.append(d['birth_cycle']); ci.append(d['child_id']); pi.append(d['parent_id'])
    if bc:
        birth_cycle = np.concatenate(bc); child_id = np.concatenate(ci); parent_id = np.concatenate(pi)
        order = np.argsort(birth_cycle, kind='stable')
        birth_cycle, child_id, parent_id = birth_cycle[order], child_id[order], parent_id[order]
    else:
        birth_cycle = child_id = parent_id = np.empty(0, dtype=np.int64)

    parent_of = {int(c): (int(p), int(b)) for c, p, b in zip(child_id, parent_id, birth_cycle)}
    return {
        'birth_cycle': birth_cycle,
        'child_id': child_id,
        'parent_id': parent_id,
        'parent_of': parent_of,
    }


def trace_ancestry(organism_id, parent_of, max_depth=1_000_000):
    """Walk from organism_id back toward its seed ancestor.

    Returns the chain [organism_id, parent, grandparent, ..., seed], oldest last.
    Stops at a seed (an id with no recorded parent, i.e. an initial organism) or
    at a -1 sentinel. max_depth guards against cycles in corrupt data.
    """
    chain = [int(organism_id)]
    cur = int(organism_id)
    for _ in range(max_depth):
        entry = parent_of.get(cur)
        if entry is None:
            break  # reached a seed / an id not present in the edge list
        parent, _ = entry
        if parent < 0:
            break
        chain.append(parent)
        cur = parent
    return chain


def self_replicators(lineage, min_offspring=1):
    """Ids that produced at least min_offspring children (i.e. actually replicated).

    Returns dict {parent_id -> offspring_count}, descending by count.
    """
    ids, counts = np.unique(lineage['parent_id'], return_counts=True)
    keep = (ids >= 0) & (counts >= min_offspring)
    pairs = sorted(zip(ids[keep].tolist(), counts[keep].tolist()), key=lambda kv: -kv[1])
    return dict(pairs)


def index_snapshot_genomes(snapshots):
    """Build {organism_id -> genome_row} from population snapshots.

    Only organisms alive at a snapshot cycle are captured, so this is a sampled
    view of each lineage (organisms born and replaced between snapshots are
    missing). Later snapshots overwrite earlier ones, but ids are unique so it
    only matters if the same id somehow reappears.
    """
    genomes = {}
    for snap in snapshots:
        ids = snap['id']; genome = snap['genome']; alive = snap.get('alive')
        for i in range(len(ids)):
            if alive is not None and not alive[i]:
                continue
            genomes[int(ids[i])] = genome[i]
    return genomes


def conservation_map(chain, genomes_by_id):
    """Per-tape-position mutability across a lineage's captured genomes.

    chain: ancestry ids (e.g. from trace_ancestry); genomes_by_id: id -> genome row.
    Only ids present in genomes_by_id contribute (the snapshot-sampled members).

    Returns dict with, per position (up to max_genome_len):
      n_distinct  : number of distinct non-blank values seen at that position
      changed     : n_distinct > 1  (position mutated somewhere in the lineage)
      conserved   : position was present and never changed (n_distinct == 1)
    plus 'members' (ids actually used, oldest last) and 'lengths' (their genome
    lengths). NOTE: this aligns by raw position, so an insertion/deletion upstream
    shifts everything after it and shows up as a run of "changes". Use 'lengths'
    to spot indels; for exact per-mutation attribution, align genomes first.
    """
    members = [i for i in chain if i in genomes_by_id]
    if not members:
        return {'members': [], 'lengths': np.empty(0, dtype=int),
                'n_distinct': None, 'changed': None, 'conserved': None}

    mat = np.stack([np.asarray(genomes_by_id[i]) for i in members], axis=0)  # (n_members, L)
    lengths = np.array([int(np.sum(np.asarray(genomes_by_id[i]) != BLANK)) for i in members])

    n_pos = mat.shape[1]
    n_distinct = np.zeros(n_pos, dtype=int)
    for p in range(n_pos):
        col = mat[:, p]
        vals = np.unique(col[col != BLANK])
        n_distinct[p] = len(vals)

    changed = n_distinct > 1
    conserved = n_distinct == 1
    return {
        'members': members,
        'lengths': lengths,
        'n_distinct': n_distinct,
        'changed': changed,
        'conserved': conserved,
    }


def _decode_position(genome_row, pos):
    """Best-effort human label for a genome position: which section it's in and,
    for a code position, a rough opcode/marker name. Purely for readable reports.
    """
    g = np.asarray(genome_row)
    val = int(g[pos])
    if val == BLANK:
        return "blank"
    if val == SEP:
        return "SEP"
    if val in (R, S, Q, I):
        return {R: 'R', S: 'S', Q: 'Q', I: 'I'}[val] + " marker"
    name = OP_NAMES.get(abs(val) % 44, str(val))
    return f"~{name}({val})"


def lineage_report(lineage_dir, snapshots, top=3, n_positions=10):
    """End-to-end summary tying the pieces together.

    Loads the edge list, finds the deepest surviving self-replicator lineages,
    and prints, for the deepest one, its ancestry depth and the most-variable
    tape positions (where evolution changed the program without killing it).
    Returns the dict from conservation_map for the deepest lineage.
    """
    lineage = load_lineage(lineage_dir)
    genomes = index_snapshot_genomes(snapshots)
    parent_of = lineage['parent_of']

    # Candidate tips: ids alive in the final snapshot (the survivors we can see).
    if not snapshots:
        print("No snapshots provided."); return None
    final = snapshots[-1]
    tips = [int(i) for i, a in zip(final['id'], final['alive']) if a]

    # Rank tips by how deep their captured ancestry goes.
    scored = []
    for t in tips:
        chain = trace_ancestry(t, parent_of)
        captured = [i for i in chain if i in genomes]
        scored.append((len(captured), len(chain), t, chain))
    scored.sort(key=lambda s: (-s[0], -s[1]))

    reps = self_replicators(lineage)
    print(f"Total births recorded: {len(lineage['child_id'])}")
    print(f"Distinct self-replicators (produced >=1 child): {len(reps)}")
    print(f"Survivors in final snapshot: {len(tips)}\n")

    best = None
    for rank, (n_cap, depth, tip, chain) in enumerate(scored[:top]):
        cmap = conservation_map(chain, genomes)
        print(f"[lineage {rank}] tip id={tip}  ancestry depth={depth}  "
              f"snapshot-captured members={n_cap}")
        if n_cap >= 2 and cmap['changed'] is not None:
            n_changed = int(cmap['changed'].sum())
            print(f"    genome lengths along lineage: {cmap['lengths'].tolist()}")
            print(f"    positions that changed without killing the lineage: {n_changed}")
            ref = np.asarray(genomes[cmap['members'][-1]])  # oldest captured genome
            var_pos = np.argsort(-cmap['n_distinct'])[:n_positions]
            var_pos = [p for p in var_pos if cmap['n_distinct'][p] > 1]
            if var_pos:
                labels = ", ".join(f"pos {p} ({_decode_position(ref, p)}): "
                                   f"{cmap['n_distinct'][p]} variants" for p in var_pos)
                print(f"    most-variable positions: {labels}")
        print()
        if best is None:
            best = cmap
    return best


def compute_snapshot_properties(snap, max_genome_len):
    """Compute fitness, merit, and effective length from a snapshot dict.
    Pure NumPy, called at snapshot time (not JIT'd).
    Returns (effective_length, merit, fitness, fertile) arrays of shape (pop_size,).
    """
    mask = np.arange(max_genome_len)[None, :] < snap['genome_len'][:, None]
    effective_length = np.sum(snap['executed'] & mask, axis=1)
    merit = effective_length.astype(np.float64)  # bonus=1.0, no tasks
    gt = snap['gestation_time']
    INVALID = 2147483647
    fertile = gt < INVALID
    fitness = np.where(fertile, merit / np.maximum(gt, 1).astype(np.float64), 0.0)
    return effective_length, merit, fitness, fertile


# SS: use percentiles not avg -- include births
#
