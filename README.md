# Uncertainty-Aware-SWA

Prediction examples and calibrated coefficients for uncertainty-aware land surface temperature retrieval using split-window algorithms.

## Quick start

Run from this folder:

```sh
python -m pip install -r requirements.txt
python -m notebook example.ipynb
```

Run all notebook cells to evaluate 10,000 simulated samples with clean and noisy
inputs. Bias/RMSE tables and figures are saved in `outputs/`.

## Minimal example

```python
import pandas as pd
from predict import predict, metrics

data = pd.read_csv("data/simulation_sample_10000.csv")
lst = predict(data, model="GA2008", strategy="parameterization", calibration="clean")
print(metrics(data["LST_reference_K"].to_numpy(), lst))
```

- `strategy`: `grouping` or `parameterization`.
- `calibration`: `clean` or `uncertainty_aware`.
- Available algorithm names: `MODELS` in `predict.py`.

Inputs are `T31_K`, `T32_K` (K), `emissivity31`, `emissivity32` (dimensionless),
`wvc_g_cm2` (g/cm^2), and `sec_vza` (1/cos(VZA), not degrees).
Predicted LST is returned in K; invalid or uncovered inputs return NaN.
`LST_reference_K` is used only for evaluation.

## Files

| File / folder | Contents |
| --- | --- |
| `example.ipynb` | Prediction, noise test, and plots |
| `predict.py`, `design_terms.py` | Prediction functions |
| `perturb.py` | Reproducible input perturbations |
| `coefficients/` | Final coefficients and metadata |
| `data/` | Sample CSVs and reference predictions for tests |
| `outputs/` | Metric tables and figures |
| `test_prediction.py` | Optional tests: `python -m unittest -v` |

The sample is for demonstration, not an independent held-out accuracy assessment.

## Citation

**Bridging accuracy and robustness in land surface temperature retrieval: An uncertainty-aware calibration framework for split-window algorithms**

Full citation and DOI will be added upon publication.
