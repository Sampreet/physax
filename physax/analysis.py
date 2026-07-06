import glob
import numpy as np

from physax.config import (BLANK, SEP, I, R, S, Q, OP_NAMES, N_OPERANDS,
                            UP_IS_SIZE, SELF_REPLICATING, FERTILE)


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


def load_snapshots(lineage_dir):
    """Load every snapshot_*.npz written by run_simulation(..., snapshot_interval>0).

    Returns a list of snapshot dicts (ordered by cycle) with keys id / genome /
    alive / status / executed / genome_len / ... — the format consumed by
    index_snapshot_genomes, conservation_map and lineage_report. Empty list if
    no snapshots were dumped (snapshot_interval was 0).
    """
    files = sorted(glob.glob(f"{lineage_dir}/snapshot_*.npz"),
                   key=lambda f: int(f.rsplit('_', 1)[1].split('.')[0]))
    snaps = []
    for f in files:
        d = np.load(f)
        snaps.append({k: d[k] for k in d.files})
    return snaps


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


def self_modification_rate(snap_a, snap_b):
    """Fraction of organisms alive in BOTH snapshots (matched by id) whose own
    genome changed between them -- direct evidence of self-modifying code, since
    ordinary mutation only ever hits an offspring at birth, never a living cell.

    Returns dict: survivors, modifiers, rate, mean_positions_changed (over the
    modifiers), and transitions {(status_a,status_b): count} for the modifiers.
    """
    def index(s):
        a = np.asarray(s['alive']).astype(bool)
        ids = np.asarray(s['id'])[a]
        return {int(i): j for j, i in enumerate(ids)}, a

    ia, aa = index(snap_a)
    ib, ab = index(snap_b)
    ga = np.asarray(snap_a['genome']); gb = np.asarray(snap_b['genome'])
    sa = np.asarray(snap_a['status']); sb = np.asarray(snap_b['status'])
    idx_a = {int(i): j for j, i in enumerate(np.asarray(snap_a['id']))}
    idx_b = {int(i): j for j, i in enumerate(np.asarray(snap_b['id']))}

    survivors = [i for i in ia if i in ib]
    modifiers = 0
    pos_changed = []
    trans = {}
    for i in survivors:
        ra, rb = idx_a[i], idx_b[i]
        g1, g2 = ga[ra], gb[rb]
        L = max(int(np.sum(g1 != BLANK)), int(np.sum(g2 != BLANK)))
        nd = int(np.sum(g1[:L] != g2[:L]))
        if nd > 0:
            modifiers += 1
            pos_changed.append(nd)
            key = (int(sa[ra]), int(sb[rb]))
            trans[key] = trans.get(key, 0) + 1
    n = len(survivors)
    return {
        'survivors': n,
        'modifiers': modifiers,
        'rate': (modifiers / n) if n else float('nan'),
        'mean_positions_changed': (float(np.mean(pos_changed)) if pos_changed else 0.0),
        'transitions': trans,
    }


def self_modification_over_time(lineage_dir):
    """Run self_modification_rate over every consecutive snapshot pair in
    lineage_dir. Returns a list of dicts (one per interval), oldest first, each
    tagged with cycle_from / cycle_to."""
    snaps = load_snapshots(lineage_dir)
    out = []
    for a, b in zip(snaps, snaps[1:]):
        rec = {'cycle_from': int(a.get('cycle', -1)), 'cycle_to': int(b.get('cycle', -1))}
        rec.update(self_modification_rate(a, b))
        out.append(rec)
    return out


def _pheno_key(snapshot, i, phenotype):
    """Phenotype key for organism index i: 'effective' executable program, its
    'gestation' replication speed, or its 'status' class."""
    if phenotype == "gestation":
        return int(np.asarray(snapshot["gestation_time"])[i])
    if phenotype == "status":
        return int(np.asarray(snapshot["status"])[i])
    gl = int(np.asarray(snapshot["genome_len"])[i])
    row = np.asarray(snapshot["genome"])[i, :gl]
    ex = row[np.asarray(snapshot["executed"])[i, :gl].astype(bool)]
    return (np.abs(ex) % UP_IS_SIZE).astype(np.int16).tobytes()


def _mask(snapshot, status_filter):
    m = np.asarray(snapshot["alive"]).astype(bool)
    if status_filter is not None:
        m = m & (np.asarray(snapshot["status"]) == status_filter)
    return m


def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def gp_map_bias(snapshot, phenotype="effective", status_filter=SELF_REPLICATING):
    """Structure of the genotype->phenotype map from ONE snapshot -- no lineage,
    no mutations, just group living genomes by the phenotype they express.

    Returns n_genotypes, n_phenotypes, redundancy (genotypes per phenotype),
    neutral_set_sizes (distinct genotypes mapping to each phenotype), and gini
    (bias: 0 = every phenotype equally redundant, ->1 = a few phenotypes soak up
    almost all genotypes).
    """
    m = _mask(snapshot, status_filter)
    idx = np.nonzero(m)[0]
    glen = np.asarray(snapshot["genome_len"]).astype(int)
    genome = np.asarray(snapshot["genome"])
    pheno_to_geno = {}
    genos = set()
    for i in idx:
        gk = genome[i, :int(glen[i])].tobytes()
        pk = _pheno_key(snapshot, i, phenotype)
        genos.add(gk)
        pheno_to_geno.setdefault(pk, set()).add(gk)
    sizes = np.array([len(v) for v in pheno_to_geno.values()])
    npheno = len(pheno_to_geno)
    return {
        "cycle": int(snapshot.get("cycle", -1)),
        "n_genotypes": len(genos), "n_phenotypes": npheno,
        "redundancy": (len(genos) / npheno) if npheno else float("nan"),
        "neutral_set_sizes": sizes,
        "gini": _gini(sizes), "max_set": int(sizes.max()) if sizes.size else 0,
    }


def _geno_dist(a, b):
    """Crude mutational distance: length difference + Hamming over the overlap.
    Equals a true single point-mutation / indel count for adjacent genomes."""
    la, lb = len(a), len(b)
    m = min(la, lb)
    return abs(la - lb) + int(np.sum(a[:m] != b[:m]))


def genotype_network(snapshot, phenotype="effective", status_filter=SELF_REPLICATING,
                     max_dist=1):
    """Build the genotype network from ONE snapshot: nodes are distinct genotypes,
    edges join genotypes within max_dist point-mutations/indels (so descendants,
    being mutationally close, cluster). Each node carries its expressed phenotype
    and how many organisms share it.

    Returns (G, node_info, components_per_phenotype) where node_info maps a genotype
    key -> {'pheno','count','arr'} and components_per_phenotype[p] is the number of
    connected components among genotypes expressing phenotype p (1 = one connected
    neutral network; >1 = fragmented / independent origins).
    """
    import networkx as nx
    m = _mask(snapshot, status_filter)
    idx = np.nonzero(m)[0]
    glen = np.asarray(snapshot["genome_len"]).astype(int)
    genome = np.asarray(snapshot["genome"])
    info = {}
    for i in idx:
        arr = genome[i, :int(glen[i])]
        gk = arr.tobytes()
        if gk not in info:
            info[gk] = {"pheno": _pheno_key(snapshot, i, phenotype), "count": 0, "arr": arr}
        info[gk]["count"] += 1
    nodes = list(info)
    G = nx.Graph()
    for k in nodes:
        G.add_node(k, pheno=info[k]["pheno"], count=info[k]["count"])
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            if _geno_dist(info[nodes[a]]["arr"], info[nodes[b]]["arr"]) <= max_dist:
                G.add_edge(nodes[a], nodes[b])
    pheno_nodes = {}
    for k in nodes:
        pheno_nodes.setdefault(info[k]["pheno"], []).append(k)
    comps = {p: nx.number_connected_components(G.subgraph(ns))
             for p, ns in pheno_nodes.items()}
    return G, info, comps


def gp_bipartite(snapshot, phenotype="effective", status_filter=SELF_REPLICATING):
    """The genotype<->phenotype map as a true bipartite graph: one node per distinct
    genotype and one per distinct phenotype, with an edge from each genotype to the
    phenotype it expresses. Because the map is a function every genotype has degree 1,
    so a PHENOTYPE node's degree is exactly its neutral-set size -- how many distinct
    genotypes map to it. Nodes carry kind in {'geno','pheno'}; phenotype nodes also
    carry 'nset' (their degree). Returns (G, geno_nodes, pheno_nodes)."""
    import networkx as nx
    m = _mask(snapshot, status_filter)
    idx = np.nonzero(m)[0]
    glen = np.asarray(snapshot["genome_len"]).astype(int)
    genome = np.asarray(snapshot["genome"])
    G = nx.Graph()
    geno_nodes, pheno_nodes = set(), set()
    for i in idx:
        gk = ("g", genome[i, :int(glen[i])].tobytes())
        pk = ("p", _pheno_key(snapshot, i, phenotype))
        if gk not in geno_nodes:
            G.add_node(gk, kind="geno"); geno_nodes.add(gk)
        if pk not in pheno_nodes:
            G.add_node(pk, kind="pheno"); pheno_nodes.add(pk)
        G.add_edge(gk, pk)
    for p in pheno_nodes:
        G.nodes[p]["nset"] = G.degree(p)
    return G, geno_nodes, pheno_nodes


def _seq_diversity(seqs):
    """Given a list of hashable sequences, return (count, n_unique, shannon_nats)."""
    from collections import Counter
    n = len(seqs)
    if n == 0:
        return {'count': 0, 'unique': 0, 'shannon': 0.0}
    c = Counter(seqs)
    total = sum(c.values())
    probs = np.array(list(c.values()), dtype=np.float64) / total
    shannon = float(-np.sum(probs * np.log(probs)))
    return {'count': n, 'unique': len(c), 'shannon': shannon}


def functional_diversity(snapshot, status_filter=SELF_REPLICATING):
    """Decompose genotypic diversity into three nested levels, to test whether
    genetic variation is functional or lives in junk / synonymous encodings.

    For the alive organisms of the given class in one snapshot, measures diversity of:
      1. raw       -- the full token sequence (genotype).
      2. executed  -- tokens only at positions the VM actually ran (drops junk /
                      non-coding tape; neutral-by-construction positions removed).
      3. effective -- canonical opcodes (abs(token) % UP_IS_SIZE) at executed
                      positions (further collapses modulo-synonymous encodings of
                      the same operation).

    A large drop unique(raw) >> unique(executed) >> unique(effective) means the
    high genotypic diversity is mostly non-functional: many tapes, few programs.

    Returns {'raw':..,'executed':..,'effective':..} each a dict with count / unique
    / shannon, plus 'collapse' = unique(effective)/unique(raw). status_filter=None
    uses all alive organisms.

    NOTE: 'effective' collapses opcode-value synonymy but not register-remapping
    synonymy (two programs using different operand registers to the same effect);
    for that, compare the decoded instruction_table / re-run the VM.
    """
    alive = np.asarray(snapshot['alive']).astype(bool)
    genome = np.asarray(snapshot['genome'])
    glen = np.asarray(snapshot['genome_len']).astype(int)
    executed = np.asarray(snapshot['executed']).astype(bool)
    mask = alive.copy()
    if status_filter is not None:
        mask &= (np.asarray(snapshot['status']) == status_filter)
    idx = np.nonzero(mask)[0]

    raw_seqs, exec_seqs, eff_seqs = [], [], []
    for i in idx:
        gl = int(glen[i])
        row = genome[i, :gl]
        raw_seqs.append(row.tobytes())
        ex = row[executed[i, :gl]]
        exec_seqs.append(ex.tobytes())
        eff_seqs.append((np.abs(ex) % UP_IS_SIZE).astype(np.int16).tobytes())

    raw = _seq_diversity(raw_seqs)
    ex = _seq_diversity(exec_seqs)
    eff = _seq_diversity(eff_seqs)
    collapse = (eff['unique'] / raw['unique']) if raw['unique'] else float('nan')
    return {'cycle': int(snapshot.get('cycle', -1)),
            'raw': raw, 'executed': ex, 'effective': eff, 'collapse': collapse}


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
