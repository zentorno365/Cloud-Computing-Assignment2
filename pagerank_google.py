#!/usr/bin/env python3
"""
PageRank with teleportation.

Supports:
1) power iteration on sparse graphs (large datasets)
2) exact closed-form solve on smaller graphs

Expected input format for Google 2002 datasets:
- first 4 lines are metadata/comments
- remaining lines: "src dst"

Usage examples
--------------
python pagerank_google.py web-Google_10k.txt --p 0.15 --iters 10 --topk 10 --exact
python pagerank_google.py web-Google.txt --p 0.15 --iters 10 --topk 10

Notes
-----
If teleport probability is p, then the damping factor is beta = 1 - p.
The PageRank vector x satisfies:

    x = (1-p) M^T x + (p/n) 1

where M is the row-stochastic hyperlink matrix after handling dangling nodes.

Closed form:
    x = (p/n) * (I - (1-p) M^T)^(-1) 1

This is practical only for small/medium graphs.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def read_google_graph(path: str | Path, skip_header: int = 4):
    """Read edge list, remap arbitrary node ids to 0..n-1, and return sparse adjacency."""
    node_to_idx: Dict[int, int] = {}
    edges_src: List[int] = []
    edges_dst: List[int] = []

    def get_idx(node_id: int) -> int:
        if node_id not in node_to_idx:
            node_to_idx[node_id] = len(node_to_idx)
        return node_to_idx[node_id]

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < skip_header:
                continue
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split()
            u = get_idx(int(a))
            v = get_idx(int(b))
            edges_src.append(u)
            edges_dst.append(v)

    n = len(node_to_idx)
    data = np.ones(len(edges_src), dtype=np.float64)
    A = sparse.csr_matrix((data, (np.array(edges_src), np.array(edges_dst))), shape=(n, n))
    # collapse duplicate edges to 1 link count if duplicates exist
    A.data[:] = 1.0
    A.sum_duplicates()

    idx_to_node = np.empty(n, dtype=np.int64)
    for node_id, idx in node_to_idx.items():
        idx_to_node[idx] = node_id
    return A, idx_to_node


def build_row_stochastic(A: sparse.csr_matrix) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Return row-stochastic M and dangling mask."""
    outdeg = np.asarray(A.sum(axis=1)).ravel()
    dangling = (outdeg == 0)
    safe_outdeg = outdeg.copy()
    safe_outdeg[dangling] = 1.0
    inv_outdeg = 1.0 / safe_outdeg
    M = sparse.diags(inv_outdeg) @ A
    return M.tocsr(), dangling


def pagerank_power(
    A: sparse.csr_matrix,
    p: float = 0.15,
    iters: int = 10,
    tol: float | None = None,
):
    """
    Power iteration for:
        x_{t+1} = (1-p) M^T x_t + (1-p) * dangling_mass/n * 1 + p/n * 1
    """
    n = A.shape[0]
    M, dangling = build_row_stochastic(A)
    x = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = np.full(n, 1.0 / n, dtype=np.float64)

    history = []
    for t in range(iters):
        dangling_mass = x[dangling].sum()
        x_next = (1.0 - p) * (M.T @ x)
        x_next += ((1.0 - p) * dangling_mass + p) * teleport
        x_next /= x_next.sum()
        diff = np.abs(x_next - x).sum()
        history.append(diff)
        x = x_next
        if tol is not None and diff < tol:
            break
    return x, history


def pagerank_closed_form(A: sparse.csr_matrix, p: float = 0.15):
    """
    Exact solve for small graphs:
        x = (p/n) * (I - (1-p) P^T)^(-1) 1
    where P is the full stochastic matrix including dangling rows replaced by uniform rows.
    """
    n = A.shape[0]
    M, dangling = build_row_stochastic(A)
    ones = np.ones(n, dtype=np.float64)

    # Build P^T = M^T + uniform contribution for dangling rows
    PT = M.T.tocsr()
    if dangling.any():
        dang_idx = np.where(dangling)[0]
        rows = np.repeat(np.arange(n), len(dang_idx))
        cols = np.tile(dang_idx, n)
        vals = np.full(n * len(dang_idx), 1.0 / n, dtype=np.float64)
        PT = PT + sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))

    I = sparse.eye(n, format="csr", dtype=np.float64)
    rhs = (p / n) * ones
    x = spsolve(I - (1.0 - p) * PT, rhs)
    x = np.asarray(x, dtype=np.float64).ravel()
    x = np.maximum(x, 0)
    x /= x.sum()
    return x


def topk_report(x: np.ndarray, idx_to_node: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    order = np.argsort(-x)[:k]
    return [(int(idx_to_node[i]), float(x[i])) for i in order]


def compare_vectors(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "l1": float(np.abs(a - b).sum()),
        "l2": float(np.linalg.norm(a - b)),
        "linf": float(np.max(np.abs(a - b))),
        "sum_a": float(a.sum()),
        "sum_b": float(b.sum()),
    }


def write_tsv(path: str | Path, x: np.ndarray, idx_to_node: np.ndarray):
    order = np.argsort(-x)
    with open(path, "w", encoding="utf-8") as f:
        for i in order:
            f.write(f"{x[i]:.12e}\t{int(idx_to_node[i])}\n")


def demo_small_graph():
    """Built-in illustrative example."""
    # 0 -> 1,2,3 ; 1->0 ; 2->0 ; 3->0
    src = np.array([0, 0, 0, 1, 2, 3])
    dst = np.array([1, 2, 3, 0, 0, 0])
    A = sparse.csr_matrix((np.ones_like(src, dtype=float), (src, dst)), shape=(4, 4))
    idx_to_node = np.arange(4)
    print("Small 4-node example")
    for p in [0.05, 0.15, 0.5, 0.9]:
        exact = pagerank_closed_form(A, p=p)
        approx, _ = pagerank_power(A, p=p, iters=200, tol=1e-14)
        cmp = compare_vectors(exact, approx)
        print(f"p={p:.2f} top={topk_report(exact, idx_to_node, 4)} l1={cmp['l1']:.3e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="path to web-Google_10k.txt or web-Google.txt")
    parser.add_argument("--p", type=float, default=0.15, help="teleport probability p")
    parser.add_argument("--iters", type=int, default=10, help="number of power iterations")
    parser.add_argument("--tol", type=float, default=None, help="optional stopping tolerance")
    parser.add_argument("--topk", type=int, default=10, help="number of top nodes to print")
    parser.add_argument("--exact", action="store_true", help="also compute exact closed form (small graphs only)")
    parser.add_argument("--out", type=str, default=None, help="optional TSV output path")
    parser.add_argument("--demo", action="store_true", help="run built-in illustrative example")
    args = parser.parse_args()

    if args.demo:
        demo_small_graph()
        return

    if not args.path:
        raise SystemExit("Please provide a graph file path, or use --demo.")

    A, idx_to_node = read_google_graph(args.path)
    n, m = A.shape[0], A.nnz
    print(f"Loaded graph: n={n:,} nodes, m={m:,} directed edges")

    x_power, history = pagerank_power(A, p=args.p, iters=args.iters, tol=args.tol)
    print(f"Power iteration finished after {len(history)} steps")
    if history:
        print("Last L1 update:", history[-1])
    print(f"Top {args.topk} nodes by power iteration:")
    for node_id, score in topk_report(x_power, idx_to_node, args.topk):
        print(f"{node_id}\t{score:.12e}")

    if args.out:
        write_tsv(args.out, x_power, idx_to_node)
        print(f"Wrote ranked output to {args.out}")

    if args.exact:
        # Warn implicitly by running it only if requested.
        print("Computing exact closed-form solution...")
        x_exact = pagerank_closed_form(A, p=args.p)
        cmp = compare_vectors(x_exact, x_power)
        print("Comparison exact vs power:")
        for k, v in cmp.items():
            print(f"{k}: {v:.12e}")
        print(f"Top {args.topk} nodes by exact solve:")
        for node_id, score in topk_report(x_exact, idx_to_node, args.topk):
            print(f"{node_id}\t{score:.12e}")


if __name__ == "__main__":
    main()
