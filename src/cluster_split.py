"""
Section 11: Cluster-based data splitting (leakage prevention)

Wraps CD-HIT to cluster highly similar sequences so that near-duplicates
never end up split across train and test sets.

Install CD-HIT first:
    conda install -c bioconda cd-hit
"""
import subprocess
from pathlib import Path


def cluster_sequences(fasta_path: str, output_prefix: str, identity_threshold: float = 0.9) -> str:
    """
    Run CD-HIT to cluster sequences at the given identity threshold
    (default 90%). Returns path to the .clstr file mapping sequences
    to cluster IDs.
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # CD-HIT's recommended word size (-n) scales with identity threshold —
    # using the default n=5 (tuned for ~0.75-0.8 identity) at 0.90+ identity
    # is both less accurate AND much slower than the recommended n=8-10,
    # since a smaller word size explodes the candidate-pair search space.
    # -T 0 uses all available threads; -M 0 uses all available memory
    # (cap this via -M <MB> if running on a shared/memory-constrained
    # machine — see build_positive_dataset.py's run_cdhit for an example
    # with an explicit 4000MB cap).
    word_size = "8" if identity_threshold >= 0.90 else "5"

    cmd = [
        "cd-hit-est",
        "-i", str(fasta_path),
        "-o", str(output_prefix),
        "-c", str(identity_threshold),
        "-n", word_size,
        "-T", "0",
        "-M", "0",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "CD-HIT is not installed or not on PATH. "
            "Install it with: conda install -c bioconda cd-hit"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"CD-HIT failed:\n{e.stderr}")

    return f"{output_prefix}.clstr"


def parse_clusters(clstr_file: str) -> dict:
    """
    Parse a CD-HIT .clstr file into {sequence_id: cluster_id}.
    Use this to build the cluster_ids Series passed into model_training.py.
    """
    mapping = {}
    cluster_id = -1
    with open(clstr_file) as f:
        for line in f:
            if line.startswith(">Cluster"):
                cluster_id += 1
            else:
                # line format: 0   300aa, >seq_id... *
                seq_id = line.split(">")[1].split("...")[0]
                mapping[seq_id] = cluster_id
    return mapping


def cluster_aware_group_split(cluster_ids: dict, all_ids: list, labels: dict,
                               train_ratio: float, val_ratio: float, test_ratio: float,
                               seed: int = 42) -> dict:
    """
    Assigns each CD-HIT cluster entirely to train, val, or test — never
    split across sets — using a greedy balanced bin-packing heuristic:
    process clusters largest-first, and at each step assign the cluster
    to whichever split is furthest below its target size ratio. This
    keeps the realized split close to train_ratio/val_ratio/test_ratio
    even though clusters vary in size, without ever separating
    near-duplicate sequences across splits (the actual leakage-prevention
    requirement).

    Returns {sequence_id: "train" | "val" | "test"}.
    """
    import random
    from collections import defaultdict

    random.seed(seed)
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "train/val/test ratios must sum to 1.0"

    # Group sequence IDs by cluster
    clusters = defaultdict(list)
    for seq_id in all_ids:
        cluster_id = cluster_ids.get(seq_id, f"singleton_{seq_id}")  # unclustered = its own singleton
        clusters[cluster_id].append(seq_id)

    cluster_list = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    random.shuffle(cluster_list)  # break ties / avoid deterministic ordering artifacts
    cluster_list.sort(key=lambda kv: -len(kv[1]))  # then sort by size again (stable)

    target_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    split_counts = {"train": 0, "val": 0, "test": 0}
    total = len(all_ids)

    assignment = {}
    for cluster_id, members in cluster_list:
        # assign to the split currently furthest below its target share
        deficits = {
            split: target_ratios[split] - (split_counts[split] / total if total else 0)
            for split in split_counts
        }
        chosen = max(deficits, key=deficits.get)
        for seq_id in members:
            assignment[seq_id] = chosen
        split_counts[chosen] += len(members)

    return assignment


if __name__ == "__main__":
    import sys
    clstr = cluster_sequences(sys.argv[1], sys.argv[2])
    print(parse_clusters(clstr))
