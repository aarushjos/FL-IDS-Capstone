# coding: utf-8
"""
Black-box + White-box integration tests for all new FL-IDS components.
Run: venv/Scripts/python.exe -m pytest tests/test_new_implementations.py -v
"""
import numpy as np
import pytest
from unittest.mock import MagicMock
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Status, Code

from src.components.server.aggregator import (
    clip_to_median_norm, compute_layer_wise_cosine_similarity,
    compute_mad_scores, extract_final_layer,
)
from src.components.server.baselines import (
    _geometric_median, GeoMedianBaseline, LayerwiseCosineKrumBaseline,
)
from src.components.server.hra_aggregator import HRABaseline
from src.components.client.attacker import min_max_attack, min_sum_attack, lie_attack


# ─── Shared fixtures ───────────────────────────────────────────────────────────

def _make_params(K=5, seed=0):
    rng = np.random.default_rng(seed)
    return [
        [rng.standard_normal((10, 3)).astype(np.float32),
         rng.standard_normal((10,)).astype(np.float32)]
        for _ in range(K)
    ]


def _flower_results(param_list):
    results = []
    for i, p in enumerate(param_list):
        proxy = MagicMock()
        proxy.cid = str(i)
        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(p),
            num_examples=100,
            metrics={},
        )
        results.append((proxy, fit_res))
    return results


def _norm(param):
    return float(np.linalg.norm(np.concatenate([a.flatten() for a in param])))


# ─── 1. clip_to_median_norm ────────────────────────────────────────────────────

class TestClipToMedianNorm:

    def test_amplified_client_capped(self):
        params = _make_params(K=5)
        params[0] = [a * 100.0 for a in params[0]]
        orig_norms = [_norm(p) for p in params]
        clipped = clip_to_median_norm(params)
        new_norms = [_norm(p) for p in clipped]
        median_n = float(np.median(orig_norms))
        assert new_norms[0] <= median_n * 1.01, \
            f"Attacker norm {new_norms[0]:.4f} > median {median_n:.4f}"

    def test_no_client_exceeds_median_after_clip(self):
        # FLAME design: clips ALL above-median clients, not just attacker.
        params = _make_params(K=5)
        params[0] = [a * 100.0 for a in params[0]]
        orig_norms = [_norm(p) for p in params]
        clipped = clip_to_median_norm(params)
        new_norms = [_norm(p) for p in clipped]
        median_n = float(np.median(orig_norms))
        assert all(n <= median_n * 1.01 for n in new_norms)

    def test_returns_same_count(self):
        params = _make_params(K=5)
        assert len(clip_to_median_norm(params)) == 5

    def test_empty_list_safe(self):
        assert clip_to_median_norm([]) == []

    def test_single_client_safe(self):
        params = _make_params(K=1)
        assert len(clip_to_median_norm(params)) == 1

    def test_shapes_preserved(self):
        params = _make_params(K=5)
        params[0] = [a * 100.0 for a in params[0]]
        clipped = clip_to_median_norm(params)
        for i in range(5):
            for j in range(len(params[i])):
                assert clipped[i][j].shape == params[i][j].shape

    def test_below_median_client_unchanged(self):
        params = _make_params(K=4, seed=99)
        params[3] = [a * 0.001 for a in params[3]]  # very small norm
        clipped = clip_to_median_norm(params)
        norms = [_norm(p) for p in params]
        smallest_idx = int(np.argmin(norms))
        # Below-median client should be the exact same object (not re-allocated)
        assert clipped[smallest_idx] is params[smallest_idx]

    def test_output_finite(self):
        params = _make_params(K=5)
        params[0] = [a * 100.0 for a in params[0]]
        clipped = clip_to_median_norm(params)
        for p in clipped:
            for a in p:
                assert np.all(np.isfinite(a))


# ─── 2. _geometric_median ──────────────────────────────────────────────────────

class TestGeometricMedian:

    def test_shape(self):
        pts = np.array([[0, 0], [1, 0], [0, 1], [10, 10]], dtype=float)
        gm = _geometric_median(pts)
        assert gm.shape == (2,)

    def test_not_pulled_to_outlier(self):
        pts = np.array([[0, 0], [1, 0], [0, 1], [10, 10]], dtype=float)
        gm = _geometric_median(pts)
        assert float(np.linalg.norm(gm - np.array([10, 10]))) > 5.0

    def test_identical_points(self):
        pts = np.ones((8, 5)) * 7.0
        gm = _geometric_median(pts)
        assert np.allclose(gm, 7.0, atol=1e-3)

    def test_single_point(self):
        pt = np.array([[3.0, 4.0]])
        gm = _geometric_median(pt)
        assert np.allclose(gm, [3.0, 4.0])

    def test_converges_on_random_data(self):
        pts = np.random.randn(20, 50)
        gm = _geometric_median(pts)
        assert np.all(np.isfinite(gm))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 1 point"):
            _geometric_median(np.empty((0, 4)))


# ─── 3. GeoMedianBaseline ──────────────────────────────────────────────────────

class TestGeoMedianBaseline:

    def setup_method(self):
        self.params5 = _make_params(K=5)
        self.init_p = ndarrays_to_parameters(self.params5[0])
        self.strat = GeoMedianBaseline(self.init_p)

    def test_correct_layer_count(self):
        agg, _ = self.strat.aggregate_fit(1, _flower_results(self.params5), [])
        assert len(parameters_to_ndarrays(agg)) == len(self.params5[0])

    def test_weight_shape_preserved(self):
        agg, _ = self.strat.aggregate_fit(1, _flower_results(self.params5), [])
        assert parameters_to_ndarrays(agg)[0].shape == self.params5[0][0].shape

    def test_bias_shape_preserved(self):
        agg, _ = self.strat.aggregate_fit(1, _flower_results(self.params5), [])
        assert parameters_to_ndarrays(agg)[1].shape == self.params5[0][1].shape

    def test_agg_values_finite(self):
        agg, _ = self.strat.aggregate_fit(1, _flower_results(self.params5), [])
        assert all(np.all(np.isfinite(a)) for a in parameters_to_ndarrays(agg))

    def test_empty_returns_none(self):
        assert self.strat.aggregate_fit(1, [], []) == (None, {})


# ─── 4. LayerwiseCosineKrumBaseline ───────────────────────────────────────────

class TestLayerwiseCosineKrum:

    def setup_method(self):
        self.params5 = _make_params(K=5)
        self.init_p = ndarrays_to_parameters(self.params5[0])

    def test_correct_layer_count(self):
        lck = LayerwiseCosineKrumBaseline(self.init_p, num_byzantine=1)
        agg, _ = lck.aggregate_fit(1, _flower_results(self.params5), [])
        assert len(parameters_to_ndarrays(agg)) == len(self.params5[0])

    def test_shapes_preserved(self):
        lck = LayerwiseCosineKrumBaseline(self.init_p, num_byzantine=1)
        agg, _ = lck.aggregate_fit(1, _flower_results(self.params5), [])
        arr = parameters_to_ndarrays(agg)
        assert arr[0].shape == self.params5[0][0].shape

    def test_empty_returns_none(self):
        lck = LayerwiseCosineKrumBaseline(self.init_p, num_byzantine=1)
        assert lck.aggregate_fit(1, [], []) == (None, {})

    def test_extreme_byzantine_floors_to_1(self):
        lck = LayerwiseCosineKrumBaseline(self.init_p, num_byzantine=1000)
        agg, _ = lck.aggregate_fit(1, _flower_results(self.params5[:2]), [])
        assert agg is not None
        assert all(np.all(np.isfinite(a)) for a in parameters_to_ndarrays(agg))

    def test_agg_values_finite(self):
        lck = LayerwiseCosineKrumBaseline(self.init_p, num_byzantine=1)
        agg, _ = lck.aggregate_fit(1, _flower_results(self.params5), [])
        assert all(np.all(np.isfinite(a)) for a in parameters_to_ndarrays(agg))


# ─── 5. HRABaseline ───────────────────────────────────────────────────────────

class TestHRABaseline:

    def setup_method(self):
        self.params5 = _make_params(K=5)
        self.init_p = ndarrays_to_parameters(self.params5[0])

    def test_correct_layer_count(self):
        hra = HRABaseline(self.init_p)
        agg, _ = hra.aggregate_fit(1, _flower_results(self.params5), [])
        assert len(parameters_to_ndarrays(agg)) == len(self.params5[0])

    def test_metrics_contain_trust_bounds(self):
        hra = HRABaseline(self.init_p)
        _, m = hra.aggregate_fit(1, _flower_results(self.params5), [])
        assert "min_trust" in m and "max_trust" in m

    def test_agg_values_finite(self):
        hra = HRABaseline(self.init_p)
        agg, _ = hra.aggregate_fit(1, _flower_results(self.params5), [])
        assert all(np.all(np.isfinite(a)) for a in parameters_to_ndarrays(agg))

    def test_empty_returns_none(self):
        hra = HRABaseline(self.init_p)
        assert hra.aggregate_fit(1, [], []) == (None, {})

    def test_reputation_tracked_across_rounds(self):
        hra = HRABaseline(self.init_p)
        for r in range(1, 5):
            hra.aggregate_fit(r, _flower_results(self.params5), [])
        assert len(hra.reputation) == 5

    def test_all_zero_phis_fallback_uniform(self):
        # Force all phis=0 by setting thresholds above possible delta range
        hra = HRABaseline(self.init_p)
        hra.t_low = 2.0
        hra.t_high = 3.0
        agg, _ = hra.aggregate_fit(1, _flower_results(self.params5), [])
        assert agg is not None
        assert all(np.all(np.isfinite(a)) for a in parameters_to_ndarrays(agg))


# ─── 6. min_max_attack ────────────────────────────────────────────────────────

class TestMinMaxAttack:

    def setup_method(self):
        self.p_local = [np.ones((10, 3), dtype=np.float32),
                        np.ones((10,), dtype=np.float32)]
        self.p_all = [
            [np.zeros((10, 3), dtype=np.float32), np.zeros((10,), dtype=np.float32)],
            [np.full((10, 3), 0.5, dtype=np.float32), np.full((10,), 0.5, dtype=np.float32)],
            self.p_local,
        ]

    def test_returns_list(self):
        result = min_max_attack(self.p_local, self.p_all, epsilon=0.5)
        assert isinstance(result, list)

    def test_correct_layer_count(self):
        result = min_max_attack(self.p_local, self.p_all, epsilon=0.5)
        assert len(result) == len(self.p_local)

    def test_shapes_preserved(self):
        result = min_max_attack(self.p_local, self.p_all, epsilon=0.5)
        for i, r in enumerate(result):
            assert r.shape == self.p_local[i].shape

    def test_finite_values(self):
        result = min_max_attack(self.p_local, self.p_all, epsilon=0.5)
        assert all(np.all(np.isfinite(a)) for a in result)

    def test_zero_direction_safe(self):
        p_zero = [np.zeros((3, 3), dtype=np.float32)]
        result = min_max_attack(p_zero, [p_zero, p_zero])
        assert len(result) == 1


# ─── 7. min_sum_attack ────────────────────────────────────────────────────────

class TestMinSumAttack:

    def setup_method(self):
        self.p_local = [np.ones((10, 3), dtype=np.float32),
                        np.ones((10,), dtype=np.float32)]
        self.p_all = [
            [np.zeros((10, 3), dtype=np.float32), np.zeros((10,), dtype=np.float32)],
            [np.full((10, 3), 0.5, dtype=np.float32), np.full((10,), 0.5, dtype=np.float32)],
            self.p_local,
        ]

    def test_returns_list(self):
        assert isinstance(min_sum_attack(self.p_local, self.p_all), list)

    def test_correct_layer_count(self):
        result = min_sum_attack(self.p_local, self.p_all)
        assert len(result) == len(self.p_local)

    def test_shapes_preserved(self):
        result = min_sum_attack(self.p_local, self.p_all)
        for i, r in enumerate(result):
            assert r.shape == self.p_local[i].shape

    def test_finite_values(self):
        result = min_sum_attack(self.p_local, self.p_all)
        assert all(np.all(np.isfinite(a)) for a in result)

    def test_norm_preserved(self):
        result = min_sum_attack(self.p_local, self.p_all)
        orig_n = np.linalg.norm(np.concatenate([a.flatten() for a in self.p_local]))
        res_n  = np.linalg.norm(np.concatenate([a.flatten() for a in result]))
        assert abs(orig_n - res_n) / (orig_n + 1e-9) < 0.01


# ─── 8. lie_attack ────────────────────────────────────────────────────────────

class TestLieAttack:

    def setup_method(self):
        self.p_all = [
            [np.zeros((10, 3), dtype=np.float32), np.zeros((10,), dtype=np.float32)],
            [np.full((10, 3), 0.5, dtype=np.float32), np.full((10,), 0.5, dtype=np.float32)],
            [np.ones((10, 3), dtype=np.float32), np.ones((10,), dtype=np.float32)],
        ]

    def test_returns_list(self):
        assert isinstance(lie_attack(self.p_all, z_clip=2.0), list)

    def test_correct_layer_count(self):
        result = lie_attack(self.p_all, z_clip=2.0)
        assert len(result) == len(self.p_all[0])

    def test_shapes_preserved(self):
        result = lie_attack(self.p_all, z_clip=2.0)
        for i, r in enumerate(result):
            assert r.shape == self.p_all[0][i].shape

    def test_finite_values(self):
        result = lie_attack(self.p_all, z_clip=2.0)
        assert all(np.all(np.isfinite(a)) for a in result)

    def test_shifts_beyond_mean(self):
        result = lie_attack(self.p_all, z_clip=2.0)
        mean_layer = np.mean([p[0] for p in self.p_all], axis=0)
        assert not np.allclose(result[0], mean_layer)

    def test_z_zero_equals_mean(self):
        result = lie_attack(self.p_all, z_clip=0.0)
        mean_layer = np.mean([p[0] for p in self.p_all], axis=0)
        assert np.allclose(result[0], mean_layer, atol=1e-5)


# ─── 9. SVD hybrid formula ────────────────────────────────────────────────────

class TestSVDHybrid:

    def test_finite_values(self):
        layers = np.random.randn(4, 20).astype(np.float32)
        sim = compute_layer_wise_cosine_similarity(layers)
        sub_mad = compute_mad_scores(sim)
        U, _, _ = np.linalg.svd(layers, full_matrices=False)
        top_proj = np.abs(U[:, 0])
        top_proj_norm = top_proj / (top_proj.max() + 1e-9)
        svd_score = 1.0 - top_proj_norm
        blended = 0.6 * sub_mad + 0.4 * (svd_score * sub_mad.std() + sub_mad.mean() - sub_mad.std())
        assert np.all(np.isfinite(blended))

    def test_same_length_as_input(self):
        layers = np.random.randn(4, 20).astype(np.float32)
        sim = compute_layer_wise_cosine_similarity(layers)
        sub_mad = compute_mad_scores(sim)
        U, _, _ = np.linalg.svd(layers, full_matrices=False)
        top_proj_norm = np.abs(U[:, 0]) / (np.abs(U[:, 0]).max() + 1e-9)
        svd_score = 1.0 - top_proj_norm
        blended = 0.6 * sub_mad + 0.4 * (svd_score * sub_mad.std() + sub_mad.mean() - sub_mad.std())
        assert len(blended) == len(sub_mad)

    def test_std_zero_edge_case(self):
        # When all suspicious clients have same MAD, std=0, formula degenerates to mean
        sub_mad_flat = np.ones(4) * -1.5
        svd_s_flat   = np.array([0.1, 0.9, 0.2, 0.8])
        blended = 0.6 * sub_mad_flat + 0.4 * (svd_s_flat * 0.0 + sub_mad_flat.mean() - 0.0)
        assert np.all(np.isfinite(blended))
        assert np.allclose(blended, sub_mad_flat.mean(), atol=1e-5)


# ─── 10. Integration pipeline ─────────────────────────────────────────────────

class TestIntegrationPipeline:

    def test_clip_then_score_pipeline(self):
        params = _make_params(K=6)
        params[0] = [a * 50.0 for a in params[0]]
        clipped = clip_to_median_norm(params)
        layers = np.stack([extract_final_layer(p) for p in clipped])
        sim = compute_layer_wise_cosine_similarity(layers)
        mad = compute_mad_scores(sim)
        assert len(mad) == 6
        assert np.all(np.isfinite(mad))

    def test_attacker_norm_capped_after_clip(self):
        params = _make_params(K=6)
        params[0] = [a * 50.0 for a in params[0]]
        orig_norms = [_norm(p) for p in params]
        orig_median = float(np.median(orig_norms))
        clipped = clip_to_median_norm(params)
        new_norms = [_norm(p) for p in clipped]
        # Attacker (index 0) must be capped to the PRE-clip cohort median norm.
        assert new_norms[0] <= orig_median * 1.01, \
            f"Attacker norm {new_norms[0]:.4f} > pre-clip median {orig_median:.4f}"


    def test_hra_three_round_run(self):
        params = _make_params(K=5)
        init_p = ndarrays_to_parameters(params[0])
        hra = HRABaseline(init_p)
        for r in range(1, 4):
            agg, metrics = hra.aggregate_fit(r, _flower_results(params), [])
            assert agg is not None
            assert "min_trust" in metrics
            assert all(np.all(np.isfinite(a)) for a in parameters_to_ndarrays(agg))

    def test_geomed_with_one_attacker(self):
        params = _make_params(K=5)
        params[0] = [a * 50.0 for a in params[0]]
        init_p = ndarrays_to_parameters(params[0])
        strat = GeoMedianBaseline(init_p)
        agg, _ = strat.aggregate_fit(1, _flower_results(params), [])
        assert agg is not None
        for a in parameters_to_ndarrays(agg):
            assert np.all(np.isfinite(a))
