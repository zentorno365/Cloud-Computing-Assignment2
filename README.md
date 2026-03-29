# PageRank and GraphRAG Project

This project implements the PageRank algorithm and extends it to a GraphRAG-style retrieval system using Personalized PageRank.

## 📌 Project Structure

```
project/
│── pagerank_google.py
│── graphrag_ppr_enhanced.py
│
│── data/
│   └── web-Google_10k.txt
│
│── images/
│   ├── pagerank_top10_webgoogle10k.png
│   ├── pagerank_convergence_webgoogle10k.png
│   ├── graphrag_pipeline_diagram.png
│   ├── graphrag_topk_scores_enhanced.png
│   ├── graphrag_personalized_vs_global.png
│   ├── graphrag_pathway_scores.png
│   └── graphrag_query_subgraph_enhanced.png
│
│── report/
│   ├── final_report.tex
│   └── final_report.pdf
│
└── README.md
```

## ⚙️ Requirements

Install dependencies:

```
pip install numpy scipy matplotlib networkx
```

## 📊 Part 2 — PageRank

Run:

```
python pagerank_google.py data/web-Google_10k.txt --p 0.15 --iters 10 --topk 10
```

## 🧠 Part 3 — GraphRAG

Run:

```
python graphrag_ppr_enhanced.py
```

## 📈 Output

- Top-k nodes
- Graph visualizations
- Multi-hop reasoning paths

## 📄 Report

See:

```
report/final_report.pdf
```

## 🔗 Dataset

Download from:
https://hunglvosu.github.io/posts/2020/07/PA3/

## ✅ Notes

- Keep both `.tex` and `.pdf` in the repo
- Include all images used in the report
- Ensure code runs without modification
