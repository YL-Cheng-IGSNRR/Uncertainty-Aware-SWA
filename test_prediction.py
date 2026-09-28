"""Run: python -m unittest -v. No training or external source data required."""
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from predict import INPUTS, MODELS, STRATEGIES, CALIBRATIONS, predict, metrics
from perturb import perturb_inputs

ROOT = Path(__file__).resolve().parent


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / 'data' / 'simulation_sample_10000.csv')

    def test_archived_implementation_parity(self):
        expected = np.load(ROOT / 'data' / 'reference_predictions.npz')
        for strategy in STRATEGIES:
            for calibration in CALIBRATIONS:
                for model in MODELS:
                    key = '__'.join((strategy, calibration, model))
                    actual = predict(self.data, model, strategy, calibration)
                    np.testing.assert_allclose(actual, expected[key], atol=1e-6, rtol=0, equal_nan=True)
                    noisy = predict(perturb_inputs(self.data), model, strategy, calibration)
                    np.testing.assert_allclose(noisy, expected[key + '__noisy'], atol=1e-6, rtol=0, equal_nan=True)
                    if strategy == 'grouping':
                        actual = predict(self.data, model, strategy, calibration, 'nodes')
                        np.testing.assert_allclose(actual, expected[key + '__nodes'], atol=1e-6, rtol=0, equal_nan=True)

    def test_truth_not_used(self):
        modified = self.data.copy()
        modified['LST_reference_K'] = -9999
        for strategy in STRATEGIES:
            np.testing.assert_array_equal(predict(self.data, strategy=strategy),
                                          predict(modified, strategy=strategy))
            np.testing.assert_array_equal(predict(self.data, strategy=strategy),
                predict(modified.drop(columns='LST_reference_K'), strategy=strategy))

    def test_invalid_inputs(self):
        row = self.data.iloc[:1].copy()
        for column, value in [('sec_vza', 2.1), ('emissivity31', 0),
                              ('wvc_g_cm2', 10), ('T31_K', np.nan)]:
            bad = row.copy()
            bad[column] = value
            for strategy in STRATEGIES:
                self.assertTrue(np.isnan(predict(bad, strategy=strategy)).all())
        row['sec_vza'] = 1.1
        with self.assertRaises(ValueError):
            predict(row, strategy='grouping', angle_mode='nodes')

    def test_metrics_sign(self):
        result = metrics(np.array([300., 302.]), np.array([301., 303.]))
        self.assertEqual(result['Bias_K'], 1.)
        self.assertEqual(result['RMSE_K'], 1.)

    def test_scalar_and_empty_inputs(self):
        for strategy in STRATEGIES:
            row = self.data.iloc[0].to_dict()
            actual = predict(row, strategy=strategy)
            self.assertEqual(actual.shape, (1,))
            np.testing.assert_allclose(actual, predict(self.data.iloc[:1], strategy=strategy))
            self.assertEqual(predict(self.data.iloc[:0], strategy=strategy).shape, (0,))

    def test_noise_reproducibility(self):
        original = self.data.copy(deep=True)
        a, b = perturb_inputs(self.data), perturb_inputs(self.data)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(self.data, original)
        np.testing.assert_array_equal(a.LST_reference_K, self.data.LST_reference_K)
        np.testing.assert_array_equal(a.sec_vza, self.data.sec_vza)
        self.assertFalse(np.array_equal(a.T31_K, self.data.T31_K))

    def test_minimal_data_columns(self):
        columns = list(INPUTS) + ['LST_reference_K']
        for filename in ['simulation_sample_10000.csv', 'simulation_sample_10000_noisy.csv']:
            frame = pd.read_csv(ROOT / 'data' / filename)
            self.assertEqual(list(frame.columns), columns)
            self.assertEqual(len(frame), 10000)

    def test_grouping_overlap_priority_and_boundaries(self):
        # Constant predictions make each selected group observable. Both stages
        # must overwrite earlier matches, never average overlapping predictions.
        def record(w, value, t=None, e=None):
            r = dict(wvc_range=w, node_coefficients=[[value, 0, 0]] * 6,
                     angle_polynomial=[[0, 0, 0], [0, 0, 0], [value, 0, 0]])
            if t is not None:
                r.update(lst_range=t, emissivity_range=e)
            return r
        table = {'clean': {'OV1992': {
            'stage1': [record([1, 2.5], 300), record([0, 1.5], 290)],
            'stage2': [record([0, 2.5], 10, [290, 310], [.8, .96]),
                       record([0, 2.5], 20, [275, 295], [.8, .96]),
                       record([0, 2.5], 30, [275, 295], [.94, 1.0])]}}}
        inputs = dict(T31_K=300, T32_K=299, emissivity31=.95,
                      emissivity32=.95, wvc_g_cm2=1.25, sec_vza=1.2)
        with patch('predict._coefficients', return_value=table):
            for mode in ['quadratic', 'nodes']:
                self.assertEqual(predict(inputs, 'OV1992', 'grouping', angle_mode=mode)[0], 30)
                self.assertEqual(predict(dict(inputs, wvc_g_cm2=1.5), 'OV1992', 'grouping', angle_mode=mode)[0], 10)
                self.assertEqual(predict(dict(inputs, emissivity31=.94, emissivity32=.94), 'OV1992', 'grouping', angle_mode=mode)[0], 30)
                self.assertTrue(np.isnan(predict(dict(inputs, emissivity31=1., emissivity32=1.), 'OV1992', 'grouping', angle_mode=mode)[0]))


if __name__ == '__main__':
    unittest.main()
