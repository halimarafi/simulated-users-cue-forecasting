"""Shared helpers: data for Stan, fitting, and simulating choices and response times."""
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Stan prints file paths from your computer in its progress messages; keep only warnings and errors.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
STAN_DIR, FIT_DIR = ROOT / "stan", ROOT / "data" / "derived" / "fits"
SS_AMT = 20.0
TRAITS = {  # columns of theta, per rung (unconstrained scale)
    "rung0": ["intercept", "amount_slope", "delay_slope"],
    "rung1": ["logk", "log_beta"],
    "rung2": ["logk", "log_alpha", "logit_tau", "logit_bias", "log_vmax", "log_vcoef"],
}
STAN_FILE = {"rung0": "rung0_logistic.stan", "rung1": "rung1_hyperbolic.stan", "rung2": "rung2_ddm.stan"}


def load_trials(kept_only=True):
    d = pd.read_csv(ROOT / "data" / "derived" / "trials_clean.csv")
    if kept_only:
        d = d[d.keep == 1]
    return reindex(d)


def reindex(d, n_people=None):
    """Number people 1..P and person-sessions ('units') 1..U. Optionally keep only the first n_people."""
    d = d.copy()
    people = np.sort(d.person.unique())[:n_people]
    d = d[d.person.isin(people)]
    d["p"] = d.person.map({v: i + 1 for i, v in enumerate(people)})
    d["u"] = (d.p - 1) * 2 + d.session            # person-session unit: 2 per person
    d["amount_level"] = d.groupby(["u", "condition"]).later_amount.rank(method="dense").astype(int)
    d["delay_level"] = d.groupby(["u", "condition"]).delay.rank(method="dense").astype(int)
    return d.reset_index(drop=True)


def stan_data(d, with_rt=True):
    """Build the data dictionary for Stan from a trial table (already filtered)."""
    units = d.groupby("u").p.first().sort_index()
    U = int(d.u.max())
    assert list(units.index) == list(range(1, U + 1)), "every person-session must have trials"
    out = dict(N=len(d), P=int(d.p.max()), U=U, person=units.tolist(), unit=d.u.tolist(),
               choice=d.choice.astype(int).tolist(), amt=d.later_amount.tolist(), delay=d.delay.tolist(), ss_amt=SS_AMT)
    if with_rt:
        out["rt"] = d.rt.tolist()
        out["min_rt"] = d.groupby("u").rt.min().sort_index().tolist()
    return out


RUNG2_START = [-4.0, 0.7, 0.0, 0.0, 0.2, -1.0]   # sensible group averages to start rung 2 from


def fit(rung, data, chains=4, warmup=1000, samples=1000, seed=1, show_progress=True):
    import cmdstanpy
    model = cmdstanpy.CmdStanModel(stan_file=str(STAN_DIR / STAN_FILE[rung]))
    inits = 0.2
    if rung == "rung2":
        # Start every chain near sensible values, with small random differences between chains.
        rng = np.random.default_rng(seed)
        inits = [dict(mu=(np.array(RUNG2_START) + rng.normal(0, 0.2, 6)).tolist(),
                      sigma_p=np.full(6, 0.5).tolist(), sigma_s=np.full(6, 0.2).tolist(),
                      z_p=rng.normal(0, 0.1, (data["P"], 6)).tolist(), z_s=rng.normal(0, 0.1, (data["U"], 6)).tolist())
                 for _ in range(chains)]
    return model.sample(data=data, chains=chains, parallel_chains=chains, iter_warmup=warmup, iter_sampling=samples,
                        seed=seed, inits=inits, show_progress=show_progress, adapt_delta=0.9)


def diagnostics(f):
    """One-line health check of a fit: the three numbers that say whether it can be trusted."""
    s = f.summary()
    s = s[~s.index.str.startswith(("lp__", "z_s", "z_p"))]
    mv = f.method_variables()
    return dict(max_rhat=round(float(s.R_hat.max()), 3), min_ess=int(s.ESS_bulk.min()),
                divergent=int(mv["divergent__"].sum()),
                max_treedepth_share=round(float((mv["treedepth__"] >= f.metadata.cmdstan_config.get("max_depth", 10)).mean()), 3))


def save_draws(f, rung, tag="basic"):
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIT_DIR / f"{rung}_{tag}.npz", theta=f.stan_variable("theta"), mu=f.stan_variable("mu"),
                        sigma_p=f.stan_variable("sigma_p"), sigma_s=f.stan_variable("sigma_s"),
                        theta_p=f.stan_variable("theta_p"))


def load_draws(rung, tag="basic"):
    return dict(np.load(FIT_DIR / f"{rung}_{tag}.npz"))


# ---------------------------------------------------------------- simulation
def _value_diff(logk, amt, delay):
    return amt / (1 + np.exp(logk) * delay) - SS_AMT


def p_later(rung, th, amt, delay):
    """Chance of choosing 'later' on each trial. th: (n_trials, n_traits) on the unconstrained scale."""
    if rung == "rung0":
        eta = th[:, 0] + th[:, 1] * np.log(amt / SS_AMT) + th[:, 2] * np.log(delay)
    elif rung == "rung1":
        eta = np.exp(th[:, 1]) * _value_diff(th[:, 0], amt, delay)
    else:
        raise ValueError("rung 2 choices come from the diffusion simulator")
    return 1 / (1 + np.exp(-eta))


def ddm_params(th, amt, delay, min_rt):
    # Must match stan/rung2_ddm.stan: non-decision time is at most 95% of the fastest basic response.
    alpha, tau = np.exp(th[:, 1]), 0.95 * min_rt / (1 + np.exp(-th[:, 2]))
    bias = 1 / (1 + np.exp(-th[:, 3]))
    v = np.exp(th[:, 4]) * (2 / (1 + np.exp(-np.exp(th[:, 5]) * _value_diff(th[:, 0], amt, delay))) - 1)
    return v, alpha, tau, bias


def simulate_ddm(v, alpha, tau, bias, rng, dt=0.001, max_t=60.0):
    """Simulate a diffusion decision per trial. Returns (choice, rt). Upper boundary = later (1)."""
    n = len(v)
    x = bias * alpha
    t = np.zeros(n); choice = np.zeros(n, dtype=int); live = np.arange(n)
    sq = np.sqrt(dt)
    for _ in range(int(max_t / dt)):
        if live.size == 0:
            break
        x[live] += v[live] * dt + sq * rng.standard_normal(live.size)
        t[live] += dt
        up, lo = x[live] >= alpha[live], x[live] <= 0
        choice[live[up]] = 1
        live = live[~(up | lo)]
    return choice, t + tau


def simulate(rung, th, amt, delay, rng, min_rt=None, rt_pool=None):
    """Simulated choices and response times for one pass through the trials.
    Rungs 0 and 1 have no model of time: their response times are drawn at random from rt_pool."""
    if rung == "rung2":
        return simulate_ddm(*ddm_params(th, amt, delay, min_rt), rng)
    choice = (rng.random(len(amt)) < p_later(rung, th, amt, delay)).astype(int)
    rt = rng.choice(rt_pool, size=len(amt)) if rt_pool is not None else np.full(len(amt), np.nan)
    return choice, rt
