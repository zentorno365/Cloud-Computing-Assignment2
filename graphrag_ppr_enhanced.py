#!/usr/bin/env python3
"""
Enhanced GraphRAG-style Personalized PageRank demo.

Features:
- builds a small query-focused knowledge graph
- runs both standard and personalized PageRank
- returns top-k nodes for a multi-hop query
- generates multiple figures for a report:
    1. graphrag_pipeline_diagram.png
    2. graphrag_topk_scores_enhanced.png
    3. graphrag_personalized_vs_global.png
    4. graphrag_pathway_scores.png
    5. graphrag_query_subgraph_enhanced.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def build_demo_graph():
    G = nx.DiGraph()
    nodes = {
        "Marie Curie": "person",
        "Pierre Curie": "person",
        "polonium": "discovery",
        "radium": "discovery",
        "radioactivity": "concept",
        "radiation in medicine": "concept",
        "mobile X-ray units": "innovation",
        "battlefield radiography": "application",
        "radiotherapy": "application",
        "radioisotopes": "concept",
        "radiotracers": "concept",
        "nuclear medicine": "field",
        "PET": "imaging",
        "SPECT": "imaging",
        "medical imaging": "field",
        "cancer diagnosis": "application",
        "functional imaging": "application",
        "Curie Institute": "institution",
    }
    for n, kind in nodes.items():
        G.add_node(n, kind=kind)

    edges = [
        ("Marie Curie", "polonium", "discovered"),
        ("Marie Curie", "radium", "discovered"),
        ("Marie Curie", "radioactivity", "advanced science of"),
        ("Marie Curie", "radiation in medicine", "championed"),
        ("Marie Curie", "mobile X-ray units", "developed"),
        ("mobile X-ray units", "battlefield radiography", "enabled"),
        ("battlefield radiography", "medical imaging", "advanced"),
        ("radium", "radiotherapy", "enabled"),
        ("radioactivity", "radioisotopes", "underpins"),
        ("radioisotopes", "radiotracers", "used as"),
        ("radiotracers", "nuclear medicine", "used in"),
        ("nuclear medicine", "PET", "includes"),
        ("nuclear medicine", "SPECT", "includes"),
        ("PET", "medical imaging", "is a form of"),
        ("SPECT", "medical imaging", "is a form of"),
        ("PET", "functional imaging", "supports"),
        ("SPECT", "functional imaging", "supports"),
        ("functional imaging", "cancer diagnosis", "helps"),
        ("radiotherapy", "cancer diagnosis", "paired with"),
        ("radiation in medicine", "nuclear medicine", "influenced"),
        ("radiation in medicine", "radiotherapy", "influenced"),
        ("Curie Institute", "radiotherapy", "researches"),
        ("Curie Institute", "nuclear medicine", "researches"),
    ]
    for u, v, rel in edges:
        G.add_edge(u, v, relation=rel, weight=1.0)
        if not G.has_edge(v, u):
            G.add_edge(v, u, relation="reverse:" + rel, weight=0.35)
    return G


def run_rankings(graph, query_entities, alpha=0.85, k=10):
    personalization = {node: 0.0 for node in graph.nodes()}
    seed_weight = 1.0 / len(query_entities)
    for q in query_entities:
        personalization[q] = seed_weight

    ppr = nx.pagerank(graph, alpha=alpha, personalization=personalization, weight="weight")
    std = nx.pagerank(graph, alpha=alpha, weight="weight")
    topk = sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:k]
    return topk, ppr, std


def generate_figures(graph, topk, ppr, std, outdir="."):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Pipeline diagram
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.axis("off")
    boxes = [
        (0.05, 0.35, 0.18, 0.3, "Query\nMarie Curie +\nmedical imaging"),
        (0.29, 0.35, 0.18, 0.3, "Seed entities\nin knowledge graph"),
        (0.53, 0.35, 0.18, 0.3, "Personalized\nPageRank\npropagation"),
        (0.77, 0.35, 0.18, 0.3, "Top-k nodes\n+ answer paths"),
    ]
    for x, y, w, h, txt in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=11)
    for i in range(3):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i+1][0]
        ax.annotate("", xy=(x2-0.01, 0.5), xytext=(x1+0.01, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=2))
    plt.title("GraphRAG retrieval pipeline with Personalized PageRank")
    plt.tight_layout()
    plt.savefig(outdir / "graphrag_pipeline_diagram.png", dpi=180)
    plt.close()

    # Top-k scores
    plt.figure(figsize=(9, 5))
    plt.bar([n for n, _ in topk], [s for _, s in topk])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Personalized PageRank score")
    plt.title("Top-k relevant nodes for the Marie Curie query")
    plt.tight_layout()
    plt.savefig(outdir / "graphrag_topk_scores_enhanced.png", dpi=180)
    plt.close()

    # Standard vs personalized PageRank
    compare_nodes = [n for n, _ in topk[:8]]
    x = np.arange(len(compare_nodes))
    width = 0.38
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, [std[n] for n in compare_nodes], width=width, label="Standard PageRank")
    plt.bar(x + width/2, [ppr[n] for n in compare_nodes], width=width, label="Personalized PageRank")
    plt.xticks(x, compare_nodes, rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title("Why query personalization matters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "graphrag_personalized_vs_global.png", dpi=180)
    plt.close()

    # Pathway scores
    path1 = ["Marie Curie", "radioactivity", "radioisotopes", "radiotracers", "nuclear medicine", "PET", "medical imaging"]
    path2 = ["Marie Curie", "mobile X-ray units", "battlefield radiography", "medical imaging"]
    path_nodes = []
    for n in path1 + path2:
        if n not in path_nodes:
            path_nodes.append(n)
    plt.figure(figsize=(10, 5))
    plt.bar(path_nodes, [ppr[n] for n in path_nodes])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Personalized PageRank score")
    plt.title("Relevance propagation along the two main answer pathways")
    plt.tight_layout()
    plt.savefig(outdir / "graphrag_pathway_scores.png", dpi=180)
    plt.close()

    # Expanded subgraph
    top_nodes = {n for n, _ in topk}
    top_nodes.update(["Marie Curie", "medical imaging", "radioisotopes", "radiotracers", "mobile X-ray units"])
    H = graph.subgraph(top_nodes).copy()
    pos = nx.spring_layout(H, seed=5, k=1.05)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(H, pos, node_size=1700)
    nx.draw_networkx_labels(H, pos, font_size=8)
    nx.draw_networkx_edges(H, pos, arrows=True, arrowstyle="-|>", arrowsize=14)
    edge_labels = {(u, v): d["relation"] for u, v, d in H.edges(data=True) if not d["relation"].startswith("reverse")}
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=6)
    plt.title("Expanded query-focused GraphRAG subgraph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "graphrag_query_subgraph_enhanced.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    G = build_demo_graph()
    query_entities = ["Marie Curie", "medical imaging"]
    topk, ppr, std = run_rankings(G, query_entities, alpha=0.85, k=10)

    print("Top-k relevant nodes:")
    for node, score in topk:
        print(f"{node:25s} {score:.6f}")

    generate_figures(G, topk, ppr, std, ".")
