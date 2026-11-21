"""Helpers for notebook 05: comparing simulated copies with real people on the cued trials."""
import numpy as np
import pandas as pd

RT_MIN, RT_MAX = 0.25, 10.0          # the same exclusion rule as for real trials (notebook 01)
QS = [0.1, 0.3, 0.5, 0.7, 0.9]


def load_sim(path):
    """Simulated choices and RTs (repeats x trials). Trials outside the RT rule become missing, as for real data."""
    z = np.load(path)
    rt = z["rt"].astype(float)
    ok = (rt >= RT_MIN) & (rt <= RT_MAX)
    ch = z["choice"].astype(float)
    ch[~ok] = np.nan
    rt[~ok] = np.nan
    return ch, rt


def onehot(codes, n):
    g = np.zeros((len(codes), n))
    g[np.arange(len(codes)), codes] = 1
    return g


def share(ch, codes, n):
    """Share of 'later' per group. ch: repeats x trials (or one row). Returns repeats x groups."""
    ch = np.atleast_2d(ch)
    g = onehot(codes, n)
    ok = ~np.isnan(ch)
    return (np.where(ok, ch, 0) @ g) / np.maximum(ok @ g, 1)


def curve_error(sim_ch, real_ch, codes, n):
    """Average gap (percentage points) between simulated and real share of 'later' across the levels. One value per repeat."""
    return 100 * np.abs(share(sim_ch, codes, n) - share(real_ch, codes, n)).mean(1)


def agreement(sim_ch, real_ch):
    """Share of trials where the copy made the same choice as the real person. One value per repeat."""
    ok = ~np.isnan(sim_ch)
    return np.where(ok, sim_ch == real_ch, 0).sum(1) / ok.sum(1)


def p_later(sim_ch):
    """Each trial's predicted chance of 'later': the share of repeats in which the copy chose it (lightly smoothed)."""
    ok = ~np.isnan(sim_ch)
    return (np.nansum(sim_ch, 0) + 0.5) / (ok.sum(0) + 1)


def log_score(sim_ch, real_ch):
    """Log of the probability the copies gave to what the person actually chose, per trial."""
    p = p_later(sim_ch)
    return np.where(real_ch == 1, np.log(p), np.log(1 - p))


def rt_quantile_error(sim_ch, sim_rt, real_ch, real_rt):
    """Average gap (seconds) between simulated and real RT quantiles, for 'later' and 'now' choices. One value per repeat."""
    real = np.concatenate([np.quantile(real_rt[real_ch == c], QS) for c in (1, 0)])
    out = np.empty(len(sim_ch))
    for r in range(len(sim_ch)):
        sim = np.concatenate([np.nanquantile(np.where(sim_ch[r] == c, sim_rt[r], np.nan), QS) for c in (1, 0)])
        out[r] = np.abs(sim - real).mean()
    return out


def bootstrap_trial_se(trials, n_boot=1000, seed=0):
    """How precisely each person's real cue effect is measured, given only ~192 trials a day.
    Resample each person-day's basic and cued trials with replacement; return the standard error per person."""
    rng = np.random.default_rng(seed)
    out = {}
    for p, g in trials.groupby("person"):
        effs = np.zeros(n_boot)
        days = [(gd[gd.cued == 0].choice.values, gd[gd.cued == 1].choice.values) for _, gd in g.groupby("session")]
        for b in range(n_boot):
            effs[b] = np.mean([rng.choice(c, len(c)).mean() - rng.choice(bs, len(bs)).mean() for bs, c in days])
        out[p] = effs.std()
    return pd.Series(out)


def carry_over(rt_full, unit, block):
    """Do slow responses follow slow responses? Correlation between each trial's log RT and the next one's,
    within a session, after removing each person-day's level in each block of 24 trials (so practice is not counted)."""
    rt_full = np.atleast_2d(rt_full)
    key = unit * 100 + block
    _, k = np.unique(key, return_inverse=True)
    same_session = unit[1:] == unit[:-1]
    out = np.empty(len(rt_full))
    for r, x in enumerate(np.log(rt_full)):
        ok = ~np.isnan(x)
        m = np.bincount(k[ok], x[ok], minlength=k.max() + 1) / np.maximum(np.bincount(k[ok], minlength=k.max() + 1), 1)
        res = x - m[k]
        a, b = res[:-1], res[1:]
        use = same_session & ~np.isnan(a) & ~np.isnan(b)
        out[r] = np.corrcoef(a[use], b[use])[0, 1]
    return out


def bootstrap_people(values_by_person, n_boot=5000, seed=0):
    """Mean of a per-person quantity, with a 95% interval from resampling people."""
    v = np.asarray(values_by_person)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return v.mean(), np.quantile(boots, 0.025), np.quantile(boots, 0.975)
