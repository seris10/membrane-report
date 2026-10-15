import numpy as np
import pytest

from membrane_report import passive as pv
from membrane_report.epochs import long_square_stimulus
from membrane_report.synthetic import passive_sweep

R, TAU, V0 = 150e6, 0.012, -0.070


def steps_for(amps, **kw):
    out = []
    for k, a in enumerate(amps):
        s = passive_sweep(a, R, TAU, V0, sweep_id=k, **kw)
        out.append(pv.prepare_step(s, long_square_stimulus(s.stimulus_intervals), s.recording_end_s))
    return out


def test_noiseless_recovery_is_exact():
    fit = pv.fit_steps(steps_for((-30e-12, -50e-12)))
    assert fit.r_in_ohm == pytest.approx(R, rel=1e-8)
    assert fit.tau_s == pytest.approx(TAU, rel=1e-8)
    assert fit.c_eff_f == pytest.approx(TAU / R, rel=1e-8)
    assert fit.status == "converged"


def test_noisy_recovery_and_bootstrap():
    steps = []
    for k, a in enumerate((-30e-12, -50e-12)):
        s = passive_sweep(a, R, TAU, V0, noise_sd=0.3e-3, seed=k, sweep_id=k)
        steps.append(pv.prepare_step(s, long_square_stimulus(s.stimulus_intervals), s.recording_end_s))
    fit = pv.fit_steps(steps)
    assert fit.r_in_ohm == pytest.approx(R, rel=0.02)
    assert fit.tau_s == pytest.approx(TAU, rel=0.05)
    rs, taus = pv.block_bootstrap(steps, fit, n_boot=30, seed=1)
    assert np.all(rs > 0) and np.all(taus > 0)
    assert np.quantile(rs, 0.025) < np.quantile(rs, 0.975)


def test_held_out_prediction_of_a_linear_cell_is_exact():
    train = steps_for((-30e-12, -50e-12))
    test = steps_for((-110e-12,))[0]
    fit = pv.fit_steps(train)
    err = pv.step_errors(test, fit)
    assert err["rms_error_v"] < 1e-9
    assert err["predicted_steady_deflection_v"] == pytest.approx(err["observed_steady_deflection_v"], rel=1e-6)


def test_ode_matches_analytic_solution():
    t = np.linspace(0.4, 2.0, 5001)
    a = pv.step_response(t, 0.6, 1.6, -50e-12, R, TAU, V0)
    b = pv.simulate_ode(t, 0.6, 1.6, -50e-12, R, TAU, V0)
    assert np.max(np.abs(a - b)) < 1e-9


def test_bin_mean_factor_limits():
    assert pv.bin_mean_factor(0.01, 1, 2e-5) == 1.0
    k = pv.bin_mean_factor(0.01, 50, 2e-5)
    continuous = (2 * 0.01 / 1e-3) * np.sinh(1e-3 / (2 * 0.01))
    assert k == pytest.approx(continuous, rel=1e-6)
    assert k > 1.0


def test_sag_ratio_is_zero_for_a_passive_response():
    s = passive_sweep(-90e-12, R, TAU, V0)
    assert abs(pv.sag_ratio(s, long_square_stimulus(s.stimulus_intervals))) < 1e-6


def test_units_do_not_change_the_fit():
    from membrane_report.recording import sweep_from_arrays

    s = passive_sweep(-50e-12, R, TAU, V0)
    mv = sweep_from_arrays(s.time_seconds, s.voltage_volts * 1e3, s.injected_current_amperes * 1e12, v_unit="mV",
                           i_unit="pA", cell_id="u", sweep_id=1, source_asset_id="x", protocol_name="X1PS_SubThresh")
    a = pv.fit_steps([pv.prepare_step(s, long_square_stimulus(s.stimulus_intervals), s.recording_end_s)])
    b = pv.fit_steps([pv.prepare_step(mv, long_square_stimulus(mv.stimulus_intervals), mv.recording_end_s)])
    assert b.r_in_ohm == pytest.approx(a.r_in_ohm, rel=1e-9)
    assert b.tau_s == pytest.approx(a.tau_s, rel=1e-9)


def test_zero_current_does_not_fabricate_a_passive_estimate():
    steps = steps_for((-50e-12,))
    steps[0].delta_i = 0
    with pytest.raises(ValueError, match="nonzero"):
        pv.fit_steps(steps)


@pytest.mark.parametrize("n_boot", [0, 1, -1])
def test_invalid_bootstrap_size_is_rejected(n_boot):
    steps = steps_for((-50e-12,))
    with pytest.raises(ValueError, match="at least two"):
        pv.block_bootstrap(steps, pv.fit_steps(steps), n_boot=n_boot)


def test_failed_optimizer_start_cannot_beat_a_successful_fit(monkeypatch):
    real = pv.least_squares
    calls = 0

    def fail_later(*args, **kwargs):
        nonlocal calls
        result = real(*args, **kwargs)
        calls += 1
        if calls > 1:
            result.success, result.cost = False, 0
            result.x = np.log([1e9, .5])
        return result

    monkeypatch.setattr(pv, "least_squares", fail_later)
    fit = pv.fit_steps(steps_for((-50e-12,)))
    assert fit.r_in_ohm == pytest.approx(R, rel=1e-8)
