"""Helpers for notebook 04: building copies and simulating forecasts for the cued trials."""
import numpy as np
import pandas as pd
import dd

SHIFTED = {"rung0": ["intercept"], "rung1": ["logk"], "rung2": ["logk", "log_alpha"]}  # traits the cue may move (forecast C)


def apply_shift(rung, th, shift):
    """Move traits by the cue shift. th: trials x traits (unconstrained scale). shift: dict trait -> amount."""
    th = th.copy()
    for name, amount in shift.items():
        th[:, dd.TRAITS[rung].index(name)] += amount
    return th


def rt_sampler(basic, rng):
    """Rungs 0 and 1 have no model of time. Their 'response times' are drawn from the same person-day's basic trials."""
    pools = basic.sort_values("u").groupby("u").rt.apply(np.asarray)
    start = np.concatenate([[0], np.cumsum(pools.map(len).values)[:-1]])
    flat, size = np.concatenate(pools.values), pools.map(len).values
    def draw(u):
        return flat[start[u - 1] + np.floor(rng.random(len(u)) * size[u - 1]).astype(int)]
    return draw


def simulate_copies(rung, trials, theta_draws, min_rt_unit, n_reps, rng, shift_for_rep=None, rt_draw=None, batch=25):
    """Each repeat: pick one plausible set of traits for everyone (a posterior draw), apply the cue shift if any,
    and let every copy do its own person's trials. Returns choices and response times, repeats x trials."""
    u = trials.u.values
    amt, delay = trials.later_amount.values, trials.delay.values
    picks = rng.choice(theta_draws.shape[0], size=n_reps, replace=n_reps > theta_draws.shape[0])
    choice = np.zeros((n_reps, len(trials)), dtype=np.int8)
    rt = np.zeros((n_reps, len(trials)), dtype=np.float32)
    for b0 in range(0, n_reps, batch):
        reps = range(b0, min(b0 + batch, n_reps))
        th = np.concatenate([apply_shift(rung, theta_draws[picks[r]][u - 1], shift_for_rep(r) if shift_for_rep else {})
                             for r in reps])
        A, D, U = np.tile(amt, len(reps)), np.tile(delay, len(reps)), np.tile(u, len(reps))
        if rung == "rung2":
            ch, t = dd.simulate_ddm(*dd.ddm_params(th, A, D, min_rt_unit[U - 1]), rng)
        else:
            ch = (rng.random(len(A)) < dd.p_later(rung, th, A, D)).astype(int)
            t = rt_draw(U)
        choice[b0:b0 + len(reps)] = ch.reshape(len(reps), -1)
        rt[b0:b0 + len(reps)] = t.reshape(len(reps), -1)
    return choice, rt


def make_folds(people, n_folds, seed=1):
    """Split people into n_folds groups of (nearly) equal size, at random but reproducibly."""
    order = np.random.default_rng(seed).permutation(np.sort(people))
    return {int(p): f + 1 for f, grp in enumerate(np.array_split(order, n_folds)) for p in grp}


def cue_shift_data(rung, cued, theta_mean, min_rt_unit):
    """Data for the small Stan models that measure the cue's average effect in the training people."""
    th = theta_mean[cued.u.values - 1]
    base = dict(N=len(cued), choice=cued.choice.astype(int).tolist(), th=th.tolist())
    if rung == "rung0":
        return {**base, "x_amt": np.log(cued.later_amount / dd.SS_AMT).tolist(), "x_del": np.log(cued.delay).tolist()}
    base.update(amt=cued.later_amount.tolist(), delay=cued.delay.tolist(), ss_amt=dd.SS_AMT)
    if rung == "rung1":
        return base
    tau = 0.95 * min_rt_unit[cued.u.values - 1] / (1 + np.exp(-th[:, 2]))
    # A cued response can be faster than the person's fastest basic response. Non-decision time must stay below every
    # response it explains, so for this small fit it is lowered where needed (and the count is reported).
    cap = 0.95 * cued.groupby("u").rt.transform("min").values
    base.update(rt=cued.rt.tolist(), tau=np.minimum(tau, cap).tolist())
    base["_n_capped"] = int(cued.u[tau > cap].nunique())     # person-days affected
    return base
