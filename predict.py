"""Prediction only with frozen split-window coefficients (temperatures in K)."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from design_terms import design_matrix

ROOT = Path(__file__).resolve().parent
MODELS = ('OV1992', 'FO1996', 'PR1984', 'UC1985', 'BL-WD', 'PP1991',
          'VI1991', 'UL1994', 'WA2014', 'LY2019', 'FOW1996', 'SO1991',
          'ULW1994', 'CO1994', 'SR2000', 'MT2002', 'BL1995', 'GA2008')
CALIBRATIONS = ('clean', 'uncertainty_aware')
STRATEGIES = ('grouping', 'parameterization')
INPUTS = ('T31_K', 'T32_K', 'emissivity31', 'emissivity32', 'wvc_g_cm2', 'sec_vza')


@lru_cache(maxsize=2)
def _coefficients(strategy):
    with (ROOT / 'coefficients' / (strategy + '.json')).open(encoding='utf-8') as f:
        return json.load(f)


def _inputs(data):
    """Accept a DataFrame or mapping of broadcast-compatible numbers/arrays."""
    arrays = np.broadcast_arrays(*[np.asarray(data[k], dtype=float) for k in INPUTS])
    if any(a.ndim > 1 for a in arrays):
        raise ValueError('Inputs must be scalars or one-dimensional arrays.')
    t31, t32, e31, e32, w, sec = [np.atleast_1d(a) for a in arrays]
    valid = (np.isfinite(np.stack([t31, t32, e31, e32, w, sec])).all(axis=0)
             & (t31 > 0) & (t32 > 0) & (e31 > 0) & (e31 <= 1)
             & (e32 > 0) & (e32 <= 1) & (w >= 0) & (w < 10)
             & (sec >= 1) & (sec <= 2))
    # Invalid rows remain NaN; safe placeholders prevent divide-by-zero warnings.
    safe = [np.where(valid, a, b) for a, b in
            zip((t31, t32, e31, e32, w, sec), (300., 299., .98, .98, 1., 1.))]
    return (*safe, valid)


def predict(data, model='GA2008', strategy='parameterization', calibration='clean',
            angle_mode='quadratic'):
    """Return a 1-D LST array in K; invalid/uncovered rows are NaN.

    `sec_vza` is 1/cos(VZA), NOT degrees. WVC is in g/cm^2.
    Grouping uses two steps: predicted stage-1 LST selects stage-2 bins.
    Quadratic mode evaluates frozen angle polynomials; nodes mode uses exact
    archived coefficients at sec_vza = 1, 1.2, ..., 2 only.
    No reference LST is read by this function.
    """
    if model not in MODELS or strategy not in STRATEGIES or calibration not in CALIBRATIONS:
        raise ValueError('Unknown model, strategy, or calibration.')
    if angle_mode not in ('quadratic', 'nodes'):
        raise ValueError('angle_mode must be quadratic or nodes.')
    t31, t32, e31, e32, w, sec, valid = _inputs(data)
    mean_e = (e31 + e32) / 2
    A = design_matrix(model, t31, t32, e31, e32, w, sec)
    table = _coefficients(strategy)[calibration][model]
    if strategy == 'parameterization':
        B = np.column_stack((np.ones(len(w)), t31, t32, e31, e32,
                             w * sec, w**2 * sec, sec))
        prediction = np.sum(A * (B @ np.asarray(table).T), axis=1)
    else:
        nodes = np.array([1., 1.2, 1.4, 1.6, 1.8, 2.])
        node_index = np.abs(sec[:, None] - nodes).argmin(axis=1)
        if angle_mode == 'nodes' and np.any(valid & ~np.isclose(sec, nodes[node_index], atol=1e-10, rtol=0)):
            raise ValueError('nodes mode requires sec_vza = 1, 1.2, ..., 2.')

        def evaluate(record, mask):
            if angle_mode == 'nodes':
                c = np.asarray(record['node_coefficients'])[node_index[mask]]
            else:
                p = np.asarray(record['angle_polynomial'])
                s = sec[mask, None]
                c = (p[0] * s + p[1]) * s + p[2]
            return np.sum(A[mask] * c, axis=1)

        initial = np.full(len(w), np.nan)
        for record in table['stage1']:
            lo, hi = record['wvc_range']
            mask = valid & (w >= lo) & (w < hi)
            initial[mask] = evaluate(record, mask)
        prediction = np.full(len(w), np.nan)
        # Record order preserves the original overwrite priority in overlapping bins.
        for record in table['stage2']:
            lo, hi = record['wvc_range']
            tl, th = record['lst_range']
            el, eh = record['emissivity_range']
            mask = (valid & (w >= lo) & (w < hi) & (initial >= tl)
                    & (initial < th) & (mean_e >= el) & (mean_e < eh))
            prediction[mask] = evaluate(record, mask)
    return np.where(valid, prediction, np.nan)


def metrics(reference, prediction):
    """Bias = mean(predicted - reference); RMSE uses the same finite pairs."""
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.shape != prediction.shape or reference.ndim != 1:
        raise ValueError('Reference and prediction must be equal-length 1-D arrays.')
    valid = np.isfinite(reference) & np.isfinite(prediction)
    error = prediction[valid] - reference[valid]
    return {'N_total': len(reference), 'N_valid': int(valid.sum()),
            'Bias_K': float(error.mean()) if error.size else float('nan'),
            'RMSE_K': float(np.sqrt(np.mean(error**2))) if error.size else float('nan')}
