# Dorby AI

RecSys course project (Prof. Ga Wu) for industry partner **Boardy AI**. The goal
is to explore multiple approaches to improve Boardy AI's recommendation
performance, starting with:

- **Two-tower model** trained on their dataset
- **Trained embeddings**, replacing their current approach of generic BERT embeddings

## Overview

Boardy AI currently relies on generic (non-fine-tuned) BERT embeddings for
recommendations. This project benchmarks that baseline against a two-tower
retrieval architecture and task-specific trained embeddings, evaluating
whichever combination yields the best offline/online recommendation metrics.
