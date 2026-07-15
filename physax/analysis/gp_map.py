"""Offline genotype-phenotype (GP) map analysis over population snapshots.

Consumes the ``snapshot_*.npz`` files written by
``Model.run_simulation(..., snapshot_interval>0)`` and computes GP-map structure
(functional diversity, GP-map bias, genotype networks) plus per-snapshot
fitness/merit properties. Pure NumPy (+ optional networkx); not JIT'd. Used by
``docs/make_figures.py`` and ``physax.analysis.visualization``.
"""
import glob
import numpy as np

from physax.sim.config import UP_IS_SIZE, SELF_REPLICATING, FERTILE


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

