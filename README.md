# Chemical Evidence

Core code accompanying **Chemical Evidence for Accepting Infrared Polymer Assignments in Environmental Analysis**, by Xinkang Li and Xiaoxing Yin, Jiangsu Key Laboratory of New Drug Research and Clinical Pharmacy, Xuzhou Medical University, China. Manuscript in preparation; no publication or acceptance is claimed. Contact: yinxx@xzhmu.edu.cn.

The method separates a fixed library candidate from the evidence used to accept it. It records local features as detected, unresolved or unassessable, deduplicates shared measured peaks within a candidate, and keeps the reference-band denominator fixed. A score is measured support, not identity probability. Correctly detected shared bands do not establish a unique identity.

## Install and run

Python 3.9 or newer:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/reproduce_tables.py --output outputs
python scripts/make_demo.py
python -m chemical_evidence outputs/synthetic_pe.csv --candidate polyethylene --mode ATR --signal-type absorbance --output outputs/evidence.json
```
On Windows, activate using `.venv\Scripts\activate`. The demo is explicitly synthetic and is not part of performance evaluation. Input CSV columns must be `wavenumber,intensity`, with wavenumbers in cm^-1. Declare the acquisition mode and intensity convention; the program does not guess them. Output includes evidence states, measured positions, source links and a candidate support score. It does not provide a validated deployment threshold or automatic identity certification.

## Reproducibility scope

`chemical_evidence/bands.py` is copied byte-for-byte from the original server implementation. `retrieval.py` retains its original normalization and top-five class-similarity functions. `evaluation.py` exposes the retrospective fractional-tie calculation in a small standalone interface. `data/scores/` provides archived per-observation predictions and scores for all nine reported methods in MPX, FLOPP-e, C4 and C8. The table script recomputes 36 primary method/source rows and checks them against the archived metrics, then writes class-specific acceptance. This is numerical evaluation replay, not retraining the baseline models. The original project scripts are retained under `archive/`; their full historical inputs are not bundled.

## Interpretation

* The principal 90% known-class acceptance point is matched retrospectively using evaluation labels, not calibrated independently for deployment. Boundary ties receive identical fractional weights regardless of correctness.
* MPX comprises 3,789 spectra from ten product groups; all 447 unknown spectra are PLA. Correlated spectra are not independent material replicates.
* FLOPP-e has 195 spectra, including 46 unknowns and only 12 PS observations. Its fractional PS acceptance is not a stable population estimate. Source labels are not independent identity certification.
* C4 and C8 contain author-augmented rows, not 3,000 independent experimental specimens each. Benefits are smaller there and random forests can outperform the chemical procedure.
* Numerical scores use 702–1800 cm^-1, whereas chemical checks use the full available spectrum. The comparison is between complete scoring procedures, not an information-matched ablation.
* Full baseline and chemical candidates remain fixed in the paired comparison. Do not describe the decrease in unknown acceptance as recovery of unknown identities, an LLM-agent effect, or measured improvement in environmental abundance.
* Shared bands (e.g. EVA/PE) limit specificity. Incomplete baseline context (e.g. PS at 694 cm^-1) limits observability. The fixed denominator avoids over-weighting a few available bands but can penalize correct candidates.
* Additional external textile validation did not establish general transfer. Its archived summary and class-level results are retained in `data/additional_validation/`, separately from retrospective four-panel replay. Do not pool these protocols or describe the four development panels as independent confirmation.

Report false acceptance and class-specific acceptance together. There is no universal identification or error-control guarantee.

## Source data and attribution

Original spectra are not redistributed in this core release. Obtain them under their source terms:

* MicroPlastiX SpecDB: https://doi.org/10.22000/1820
* FLOPP/FLOPP-e: https://doi.org/10.1021/acs.analchem.1c02549 and https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/
* FTIR-Plastics C4/C8: https://zenodo.org/records/10736650 and https://doi.org/10.1016/j.dib.2024.110612
* Open Specy reference collection: https://doi.org/10.5281/zenodo.10126851
* Band positions (Table 1): https://doi.org/10.1002/jemt.23841
* External textile source: https://data.mendeley.com/datasets/rx3fjgz96x/1

FLOPP-e processing trims only contiguous nonpositive edge padding before the log transform; it does not interpolate internal invalid values. Full processing history is in the archived scripts. `data/provenance.json` identifies the server-derived score inputs and hashes; `MANIFEST.sha256` checks the release contents. No raw user data, credentials, model-service transcripts or unpublished manuscript files are included.

## Citation and reuse

Please cite the associated manuscript once published and the original data sources. No manuscript DOI is assigned yet. No software license has been selected by the authors for this initial release; public visibility is not a grant of an open-source license. Third-party source terms remain applicable.
