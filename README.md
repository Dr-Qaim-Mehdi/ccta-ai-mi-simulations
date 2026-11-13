# Simulation illustrations: AI-assisted coronary CT angiography and myocardial infarction prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24195555.svg)](https://doi.org/10.5281/zenodo.24195555)

This repository holds the companion code for the manuscript *From plaque measurement to myocardial infarction prediction: a critical review of artificial intelligence in coronary CT angiography with simulation illustrations* (Q. M. Rizvi; in preparation).

The simulations quantify five methodological problems that recur in studies of AI-assisted coronary CT angiography (CCTA) for predicting myocardial infarction (MI). The cohorts are simulated to resemble SCOT-HEART.

| Experiment | Question |
|---|---|
| A | How much does an outcome-selected ("optimal") plaque threshold overstate the hazard ratio when it is derived from 41–400 MI events? |
| B | What happens when a fixed volumetric (mm³) threshold is applied to a platform that measures plaque one-third lower, as reported for photon-counting CT? |
| C | How much does 1 − Kaplan–Meier overstate absolute MI risk when there are competing deaths? |
| D | How stable is the incremental C-statistic of plaque over a strong comparator (risk score, calcium, stenosis) at realistic event counts? |
| E | How much do composite endpoints that include scan-triggered revascularisation inflate apparent discrimination? |

**All data are simulated. No patient data are used or included.**

## Run it

**Google Colab (easiest):** open `ccta_ai_mi_simulations.ipynb` in Colab and choose *Runtime → Run all*. A CPU runtime is enough.

**Locally:**

```bash
pip install -r requirements.txt
python ccta_ai_mi_simulations.py
```

The full run takes about 3 minutes on a laptop CPU, or about 3–15 minutes on Colab. Set `QUICK = True` near the top of the script for a ~1-minute smoke test.

The script writes two files:
- `ccta_sim_results.json`: all results, also printed as one line between the `RESULTS` markers.
- `ccta_sim_figure.png`: a five-panel summary figure.

## Reproducibility

The fixed seed (`20261113`) gives identical numerical results across platforms. A full run on Google Colab (Linux, Python 3.13) and a full run on macOS (Python 3.13) agreed on every reported value. The reference output is `results/ccta_sim_results.json`. `results/simulation_figure.png` is the figure as used in the manuscript.

Each run performs two self-checks and prints the result:
- the Cox partial-likelihood estimator against `statsmodels.PHReg`;
- the Harrell C-index against brute-force pair counting.

## Model summary

For each simulated participant:
- A clinical risk score and a latent plaque burden are standard normal, with correlation 0.3.
- Observed log total plaque volume = log(150 mm³) + plaque + measurement error (SD 0.25).
- Log coronary calcium and obstructive stenosis are correlated with plaque.
- The cause-specific MI hazard has an HR of 2.0 per SD of plaque (about 1.6 per doubling of plaque volume) and an HR of 1.5 per SD of clinical risk.
- Baseline hazards are calibrated to 2.5% MI over 4.7 years, or to the stated competing-risk scenarios.

The full specification is in the script's docstrings and in Supplementary Methods S3 of the manuscript.

These are stylised illustrations of mechanism and magnitude. They are not estimates for any commercial product or published study.

## Citation

If you use this code, please cite the archived release:

> Rizvi QM. *Simulation illustrations: AI-assisted coronary CT angiography and myocardial infarction prediction* [software]. Version 1.0.0. Zenodo; 2026. https://doi.org/10.5281/zenodo.24195555

Machine-readable citation metadata is in `CITATION.cff`.

## License

MIT. See `LICENSE`.
