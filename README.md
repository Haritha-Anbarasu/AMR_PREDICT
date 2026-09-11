# AMR-PREDICT

Explainable Python-based machine learning pipeline for antibiotic resistance
gene (ARG) prediction from bacterial genome sequences.

## Setup

```bash
# Python packages
pip install -r requirements.txt

# External bioinformatics tools (conda recommended)
conda install -c bioconda prodigal
conda install -c bioconda -c conda-forge abricate
conda install -c bioconda cd-hit
abricate --setupdb
```

## Project layout

```
AMR_PREDICT/
├── app.py                    # Streamlit web app
├── main.py                   # CLI pipeline controller
├── data/
│   ├── training/              # Labeled ARG + non-ARG sequences for model training
│   ├── testing/                # Held-out cluster-split test set
│   └── reference/              # Genomes to run the pipeline on
├── src/
│   ├── qc.py                   # Module 1: genome QC
│   ├── gene_prediction.py       # Module 2: Prodigal wrapper
│   ├── similarity_screening.py   # Modules 3-4: ABRicate/CARD wrapper
│   ├── cluster_split.py           # Leakage-safe train/test split (CD-HIT)
│   ├── sequence_features.py        # Module 5: DNA/k-mer features
│   ├── protein_features.py          # Module 5: protein features
│   ├── model_training.py             # Modules 6-7: binary + multiclass models
│   ├── prediction.py                  # Runs trained models on new genes
│   ├── decision_engine.py              # Hybrid similarity+ML decision logic
│   ├── explainability.py                # Module 8: SHAP
│   └── report_generation.py              # CSV/HTML report output
├── models/                    # Saved .pkl models (created after training)
├── results/                   # Pipeline outputs
└── requirements.txt
```

## Downloading training data

Scripts in `scripts/` automate dataset acquisition (run these on your own
machine with internet access — they reach out to card.mcmaster.ca, NCBI,
bitbucket.org, and megares.meglab.org, none of which are reachable from a
sandboxed environment):

```bash
# Positive class, multi-database (recommended): CARD + AMRFinderPlus +
# ResFinder + MEGARes in one merged, labeled dataset
python scripts/download_multi_database.py --outdir data/training

# Or pick a subset:
python scripts/download_multi_database.py --outdir data/training --databases card resfinder

# Negative class: downloads a few reference genomes, predicts genes,
# keeps only genes with no ARG similarity hit
python scripts/build_negative_set.py --outdir data/training
```

`download_multi_database.py` writes each database's labeled FASTA + CSV into
its own subfolder (`data/training/card/`, `data/training/amrfinderplus/`,
etc.) plus a combined `merged_arg_dataset.fasta` / `merged_arg_labels.csv`
with a `source_db` column — ready for `cluster_split.py` and the feature
extraction modules. `download_card_data.py` (CARD-only) still works if you
just want CARD alone.

Each database's format changes periodically — the script prints a clear
warning naming the exact file it expected and couldn't find/parse, so you
can quickly locate and fix the affected constant at the top of the script
rather than debugging blind.

## Recommended build order

1. Get Prodigal + ABRicate/CARD working end-to-end on one test genome (no ML yet).
2. Run the data-download scripts above, dedupe + CD-HIT cluster the combined set.
3. Extract features (`sequence_features.py` + `protein_features.py`) for the training set.
4. Train the binary model, then the multiclass model (`model_training.py`).
5. Add SHAP explainability on the trained binary model.
6. Wire everything into `main.py`, test on an independent genome.
7. Connect `app.py` to the working `main.py` pipeline.

## Running

```bash
# CLI pipeline (after models are trained and saved to models/)
python main.py --genome data/reference/genome.fasta --outdir results/run1

# Web app
streamlit run app.py
```

## Notes

- k-mer size defaults to k=3 (64 features) in `sequence_features.py` — increase only if needed.
- Confidence thresholds (0.90 / 0.70) in `decision_engine.py` and `prediction.py` are starting
  points from the project plan; validate and tune them during the project.
- Train/test split is cluster-based (via CD-HIT) to prevent near-duplicate sequence leakage.
