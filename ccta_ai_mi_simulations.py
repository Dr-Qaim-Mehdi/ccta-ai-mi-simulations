# =============================================================================
# Simulation illustrations for:
#   "From plaque measurement to myocardial infarction prediction: a critical
#    review of artificial intelligence in coronary CT angiography with
#    simulation illustrations"  (Q. M. Rizvi)
#
# Five experiments, each tied to one methodological argument in the review:
#   A. Outcome-selected plaque thresholds with few MI events (optimism)
#   B. Platform shift: a fixed mm3 threshold applied to a scanner that measures
#      plaque ~1/3 lower (photon-counting CT)
#   C. Competing death: 1 - Kaplan-Meier vs cumulative incidence of MI
#   D. Incremental C-statistic of plaque over a strong comparator (risk score,
#      CAC, stenosis) at realistic event counts
#   E. Composite endpoints that include scan-triggered revascularisation
#
# All data are SIMULATED. No patient data are used.
#
# HOW TO RUN ON GOOGLE COLAB
#   1. New notebook -> paste this whole file into one cell -> Run.
#      (or upload the .py file and run:  !python ccta_ai_mi_simulations.py)
#   2. Full run takes about 3-15 minutes on a standard Colab CPU.
#      Set QUICK = True below for a ~1-minute smoke test.
#   3. A one-line JSON summary is printed between the two "=====" RESULTS
#      lines; the same results are saved to ccta_sim_results.json, and the
#      figure to ccta_sim_figure.png (Files pane on the left in Colab).
#   With the default seed the full run reproduces results/ccta_sim_results.json
#   in the GitHub repository.
#
# Requirements: numpy, pandas, scipy, matplotlib (all preinstalled on Colab);
# numba is used for speed when available (also preinstalled on Colab).
# =============================================================================

import json, time, sys, platform
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                   # pure-Python fallback
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

# ----------------------------------------------------------------------------- settings
QUICK = False                 # True = fast smoke test (few replications)
SEED = 20261113
LN2 = np.log(2.0)

# Replications per experiment (full run / quick run)
R_A = 60 if QUICK else 1000   # threshold-selection replications per event level
R_D_SMALL = 40 if QUICK else 500
R_D_LARGE = 15 if QUICK else 200
N_BIG = 60_000 if QUICK else 300_000   # size of "population" cohorts

# Data-generating model (SCOT-HEART-like stable chest-pain population)
#   latent plaque burden A ~ N(0,1); observed log total plaque volume (TPV)
#   = log(150 mm3) + A + measurement error (SD 0.25 on log scale)
#   MI cause-specific hazard: HR 2.0 per SD of A (= HR ~1.6 per doubling of TPV)
#   and HR 1.5 per SD of a clinical risk score z; corr(z, A) = 0.3
BASE = dict(b_A=LN2, b_z=np.log(1.5), b_dz=np.log(1.8), rho_cac=0.6, rho_z=0.3,
            meas_sd=0.25, mu=np.log(150.0), sig=1.0)
MI_PROP_47Y = 41 / 1611          # 2.55% MI over 4.7 years (SCOT-HEART-like)
HORIZON = 4.7

T0 = time.time()
master = np.random.default_rng(SEED)


def new_rng():
    return np.random.default_rng(master.integers(0, 2**63 - 1))


# ----------------------------------------------------------------------------- data
def gen(n, rng, h_mi, years, h_d=0.0, drop=0.0, b_A=BASE['b_A'], b_z=BASE['b_z'],
        b_dz=BASE['b_dz'], rho_cac=BASE['rho_cac'], rho_z=BASE['rho_z'],
        meas_sd=BASE['meas_sd'], mu=BASE['mu'], sig=BASE['sig']):
    z = rng.standard_normal(n)
    A = rho_z * z + np.sqrt(1 - rho_z**2) * rng.standard_normal(n)
    cac = rho_cac * A + np.sqrt(1 - rho_cac**2) * rng.standard_normal(n)
    sten = (0.8 * A + 0.6 * rng.standard_normal(n)) > 1.05          # ~15% obstructive
    e_meas = rng.standard_normal(n)
    log_tpv = mu + sig * A + meas_sd * e_meas
    lp_mi = b_z * z + b_A * A
    t_mi = rng.exponential(1.0, n) / (h_mi * np.exp(lp_mi))
    t_d = rng.exponential(1.0, n) / (h_d * np.exp(b_dz * z)) if h_d > 0 else np.full(n, np.inf)
    t_c = rng.exponential(1.0, n) / drop if drop > 0 else np.full(n, np.inf)
    t_c = np.minimum(t_c, years)
    t = np.minimum(np.minimum(t_mi, t_d), t_c)
    status = np.where(t == t_mi, 1, np.where(t == t_d, 2, 0))
    return dict(z=z, A=A, cac=cac, sten=sten.astype(float), e_meas=e_meas, log_tpv=log_tpv,
                lp_mi=lp_mi, t_mi=t_mi, t_d=t_d, t=t, status=status)


def calibrate_hmi(target, years, h_d=0.0, drop=0.0, n=200_000, **kw):
    """Baseline MI hazard giving the target observed MI proportion (bisection, common random numbers)."""
    lo, hi = np.log(1e-6), np.log(1.0)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        d = gen(n, np.random.default_rng(1), np.exp(mid), years, h_d=h_d, drop=drop, **kw)
        if (d['status'] == 1).mean() < target: lo = mid
        else: hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def calibrate_both(target_mi, target_d, years, n=200_000, **kw):
    h_mi, h_d = 0.005, 0.005
    for _ in range(4):
        lo, hi = np.log(1e-6), np.log(2.0)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            d = gen(n, np.random.default_rng(2), h_mi, years, h_d=np.exp(mid), **kw)
            if (d['status'] == 2).mean() < target_d: lo = mid
            else: hi = mid
        h_d = float(np.exp(0.5 * (lo + hi)))
        h_mi = calibrate_hmi(target_mi, years, h_d=h_d, n=n, **kw)
    return h_mi, h_d


# ----------------------------------------------------------------------------- Cox model
def cox_fit(X, t, ev, max_iter=40, tol=1e-9):
    """Cox PH (Breslow; continuous times). Returns beta, se, loglik, loglik0."""
    X = np.asarray(X, float)
    if X.ndim == 1: X = X[:, None]
    o = np.argsort(-t, kind="mergesort")
    X = X[o]; e = np.asarray(ev)[o].astype(bool)
    n, p = X.shape
    beta = np.zeros(p)
    ll0 = -np.log(np.arange(1, n + 1)[e]).sum()
    for _ in range(max_iter):
        eta = X @ beta
        c = eta.max(); w = np.exp(eta - c)
        S0 = np.cumsum(w)
        S1 = np.cumsum(w[:, None] * X, axis=0)
        S2 = np.cumsum(w[:, None, None] * X[:, :, None] * X[:, None, :], axis=0)
        s0, s1, s2 = S0[e], S1[e], S2[e]
        xbar = s1 / s0[:, None]
        U = (X[e] - xbar).sum(0)
        I = (s2 / s0[:, None, None] - xbar[:, :, None] * xbar[:, None, :]).sum(0)
        step = np.linalg.solve(I, U)          # raises LinAlgError if information is singular
        mx = np.max(np.abs(step))
        if mx > 1.0: step = step / mx         # damped Newton step for stability
        beta = beta + step
        if np.max(np.abs(step)) < tol: break
    eta = X @ beta; c = eta.max(); w = np.exp(eta - c)
    S0 = np.cumsum(w)
    ll = (eta[e] - c - np.log(S0[e])).sum()
    se = np.sqrt(np.diag(np.linalg.inv(I)))
    return beta, se, ll, ll0


# ----------------------------------------------------------------------------- Harrell C
@njit(cache=True)
def _harrell(t, ev, risk):
    n = t.shape[0]
    order = np.argsort(-t)
    ro = np.argsort(risk)
    rank = np.empty(n, np.int64)
    r = 0
    for k in range(n):
        if k > 0 and risk[ro[k]] != risk[ro[k - 1]]:
            r += 1
        rank[ro[k]] = r + 1
    m = r + 2
    tree = np.zeros(m + 1, np.int64)
    conc = 0.0; comp = 0.0; inserted = 0
    k = 0
    while k < n:
        # group of identical times (only administratively censored ties expected)
        j = k
        while j + 1 < n and t[order[j + 1]] == t[order[k]]:
            j += 1
        for q in range(k, j + 1):
            i = order[q]
            if ev[i] == 1 and inserted > 0:
                rk = rank[i]
                # count ranks < rk
                s_less = 0; x = rk - 1
                while x > 0:
                    s_less += tree[x]; x -= x & (-x)
                s_le = 0; x = rk
                while x > 0:
                    s_le += tree[x]; x -= x & (-x)
                conc += s_less + 0.5 * (s_le - s_less)
                comp += inserted
        for q in range(k, j + 1):
            x = rank[order[q]]
            while x <= m:
                tree[x] += 1; x += x & (-x)
            inserted += 1
        k = j + 1
    return conc / comp if comp > 0 else np.nan


def harrell_c(t, ev, risk):
    return float(_harrell(np.ascontiguousarray(t, np.float64),
                          np.ascontiguousarray(ev, np.int64),
                          np.ascontiguousarray(risk, np.float64)))


# ----------------------------------------------------------------------------- self-checks
def self_checks():
    rng = np.random.default_rng(7)
    d = gen(400, rng, 0.05, 5.0)
    ev = (d['status'] == 1).astype(int)
    # brute-force Harrell C
    t, r = d['t'], d['log_tpv']
    conc = comp = 0.0
    for i in range(len(t)):
        if ev[i]:
            m = t > t[i]
            comp += m.sum(); conc += (r[i] > r[m]).sum() + 0.5 * (r[i] == r[m]).sum()
    c_brute = conc / comp
    c_fast = harrell_c(t, ev, r)
    out = {"harrell_c_abs_diff": abs(c_brute - c_fast)}
    try:
        import statsmodels.api as sm
        X = np.column_stack([d['z'], d['log_tpv']])
        b, se, _, _ = cox_fit(X, t, ev)
        res = sm.PHReg(t, X, status=ev, ties="breslow").fit()
        out["cox_beta_abs_diff_vs_statsmodels"] = float(np.max(np.abs(b - res.params)))
    except Exception as exc:
        out["cox_check"] = f"statsmodels unavailable ({exc.__class__.__name__})"
    return out


# ============================================================================= A
def logrank_multi(x, t, ev, cuts):
    o = np.argsort(-t, kind="mergesort")
    G = (x[o][:, None] >= cuts[None, :]).astype(float)
    e = ev[o].astype(bool)
    nrisk = np.arange(1, len(t) + 1)[e][:, None]
    n1 = np.cumsum(G, axis=0)[e]
    E1 = n1 / nrisk
    O1 = G[e]
    V = (E1 * (1 - E1)).sum(0)
    return ((O1 - E1).sum(0))**2 / np.maximum(V, 1e-12)


def experiment_A(_unused=None):
    out = []
    for effect_label, bA in (("strong (HR 2.0 per SD)", LN2), ("weak (HR 1.3 per SD)", np.log(1.3))):
        h_mi = calibrate_hmi(MI_PROP_47Y, HORIZON, b_A=bA)
        out += _experiment_A_one(h_mi, bA, effect_label)
    return out


def _experiment_A_one(h_mi, bA, effect_label):
    pcts = np.arange(20, 91, 5)
    # "true" HR for any absolute threshold, from a population cohort
    pop = gen(N_BIG, new_rng(), h_mi, HORIZON, b_A=bA)
    grid_p = np.arange(10, 96, 1)
    grid_thr = np.percentile(pop['log_tpv'], grid_p)
    true_hr = []
    for thr in grid_thr:
        b, _, _, _ = cox_fit((pop['log_tpv'] >= thr).astype(float), pop['t'], pop['status'] == 1)
        true_hr.append(np.exp(b[0]))
    true_hr = np.array(true_hr)
    out = []
    for target_events in (41, 82, 200, 400):
        n = int(round(target_events / MI_PROP_47Y))
        app, val, tru, chosen, vsig, appsig, ci_ratio, med_app, med_val, evs = ([] for _ in range(10))
        for _ in range(R_A):
            rng = new_rng()
            dv = gen(n, rng, h_mi, HORIZON, b_A=bA); va = gen(n, rng, h_mi, HORIZON, b_A=bA)
            evd = dv['status'] == 1; evv = va['status'] == 1
            evs.append(evd.sum())
            if evd.sum() < 5 or evv.sum() < 5: continue
            cuts = np.percentile(dv['log_tpv'], pcts)
            chi = logrank_multi(dv['log_tpv'], dv['t'], evd, cuts)
            k = int(np.argmax(chi)); thr = cuts[k]
            gd = (dv['log_tpv'] >= thr).astype(float); gv = (va['log_tpv'] >= thr).astype(float)
            if gd.sum() == 0 or gv.sum() == 0 or evd[gd == 1].sum() == 0 or evv[gv == 1].sum() == 0 \
                    or evd[gd == 0].sum() == 0 or evv[gv == 0].sum() == 0:
                continue
            try:
                bd, sd, _, _ = cox_fit(gd, dv['t'], evd)
                bv, sv, _, _ = cox_fit(gv, va['t'], evv)
            except np.linalg.LinAlgError:
                continue
            app.append(np.exp(bd[0])); val.append(np.exp(bv[0]))
            tru.append(np.interp(thr, grid_thr, true_hr))
            chosen.append(pcts[k])
            appsig.append(abs(bd[0] / sd[0]) > 1.96); vsig.append(abs(bv[0] / sv[0]) > 1.96 and bv[0] > 0)
            ci_ratio.append(np.exp(2 * 1.96 * sd[0]))
            # prespecified median split (no outcome-based selection)
            md = np.median(dv['log_tpv'])
            b1, _, _, _ = cox_fit((dv['log_tpv'] >= md).astype(float), dv['t'], evd)
            b2, _, _, _ = cox_fit((va['log_tpv'] >= md).astype(float), va['t'], evv)
            med_app.append(np.exp(b1[0])); med_val.append(np.exp(b2[0]))
        app, val, tru = map(np.array, (app, val, tru))
        q = lambda a: [float(np.percentile(a, 25)), float(np.percentile(a, 75))]
        out.append(dict(
            effect=effect_label, target_events=target_events, n_per_cohort=n, mean_events=float(np.mean(evs)), reps=int(len(app)),
            apparent_HR_median=float(np.median(app)), apparent_HR_IQR=q(app),
            validation_HR_median=float(np.median(val)), validation_HR_IQR=q(val),
            true_HR_of_chosen_cut_median=float(np.median(tru)),
            optimism_ratio_median=float(np.median(app / val)),
            pct_apparent_sig=float(100 * np.mean(appsig)), pct_validation_sig=float(100 * np.mean(vsig)),
            chosen_percentile_median=float(np.median(chosen)),
            chosen_percentile_10_90=[float(np.percentile(chosen, 10)), float(np.percentile(chosen, 90))],
            apparent_CI_upper_to_lower_ratio_median=float(np.median(ci_ratio)),
            prespecified_median_split_apparent_HR_median=float(np.median(med_app)),
            prespecified_median_split_validation_HR_median=float(np.median(med_val)),
        ))
    return out


# ============================================================================= B
def experiment_B(h_mi):
    ref = gen(N_BIG, new_rng(), h_mi, HORIZON)
    thr75 = np.percentile(ref['log_tpv'], 75)          # fixed mm3 threshold set on reference platform
    new = gen(N_BIG, new_rng(), h_mi, HORIZON)
    ev = new['status'] == 1
    rows = []
    for label, s, sd in [("reference platform", 1.0, 0.25), ("10% lower", 0.9, 0.25),
                         ("20% lower", 0.8, 0.25), ("one-third lower", 2 / 3, 0.25),
                         ("one-third lower, less noise (PCD-like)", 2 / 3, 0.15)]:
        obs = BASE['mu'] + new['A'] + np.log(s) + sd * new['e_meas']
        hi = obs >= thr75
        hi_recal = obs >= np.percentile(obs, 75)
        def summ(h):
            ci_hi = ev[h].mean(); ci_lo = ev[~h].mean()
            return dict(pct_classified_high=float(100 * h.mean()),
                        sensitivity_for_MI=float(100 * ev[h].sum() / ev.sum()),
                        MI_incidence_high=float(100 * ci_hi), MI_incidence_low=float(100 * ci_lo),
                        risk_ratio=float(ci_hi / ci_lo))
        rows.append(dict(platform=label, scale=round(s, 3), meas_sd=sd,
                         fixed_threshold=summ(hi), platform_specific_75th=summ(hi_recal)))
    return dict(threshold_mm3=float(np.exp(thr75)), rows=rows)


# ============================================================================= C
def km_cuminc(t, ev):
    o = np.argsort(t, kind="mergesort"); e = ev[o]
    nrisk = len(t) - np.arange(len(t))
    return float(1 - np.exp(np.sum(np.log1p(-1.0 / nrisk[e]))))


def experiment_C():
    out = []
    for name, years, tmi, td in [("Stable chest pain, younger (10-y MI 5%, death 8%)", 10.0, 0.05, 0.08),
                                 ("Older / multimorbid (10-y MI 8%, death 30%)", 10.0, 0.08, 0.30),
                                 ("High-risk ischaemia (5-y MI 10%, death 10%)", 5.0, 0.10, 0.10)]:
        h_mi, h_d = calibrate_both(tmi, td, years)
        d = gen(N_BIG, new_rng(), h_mi, years, h_d=h_d)
        mi = d['status'] == 1
        cif = float(mi.mean()); km = km_cuminc(d['t'], mi)
        lp = d['lp_mi']
        dec = lp >= np.percentile(lp, 90)
        pred_cs = 1 - np.exp(-h_mi * np.exp(lp) * years)      # cause-specific model, death ignored
        out.append(dict(scenario=name, horizon_y=years,
                        death_pct=float(100 * (d['status'] == 2).mean()),
                        true_MI_cuminc_pct=100 * cif, one_minus_KM_pct=100 * km,
                        relative_overestimation_pct=100 * (km / cif - 1),
                        top_decile=dict(observed_MI_pct=float(100 * mi[dec].mean()),
                                        predicted_ignoring_death_pct=float(100 * pred_cs[dec].mean()),
                                        death_pct=float(100 * (d['status'][dec] == 2).mean()),
                                        over_prediction_ratio=float(pred_cs[dec].mean() / mi[dec].mean()))))
    return out


# ============================================================================= D
def experiment_D(h_mi):
    out = []
    for rho in (0.3, 0.6, 0.85):
        big1 = gen(N_BIG, new_rng(), h_mi, HORIZON, rho_cac=rho)
        big2 = gen(N_BIG, new_rng(), h_mi, HORIZON, rho_cac=rho)
        def mats(d):
            Xc = np.column_stack([d['z'], d['cac'], d['sten']])
            return Xc, np.column_stack([Xc, d['log_tpv']])
        Xc1, Xf1 = mats(big1); Xc2, Xf2 = mats(big2)
        e1 = big1['status'] == 1; e2 = big2['status'] == 1
        bc, _, _, _ = cox_fit(Xc1, big1['t'], e1); bf, _, _, _ = cox_fit(Xf1, big1['t'], e1)
        c_comp_true = harrell_c(big2['t'], e2, Xc2 @ bc); c_full_true = harrell_c(big2['t'], e2, Xf2 @ bf)
        true_dc = c_full_true - c_comp_true
        for target_events in (41, 82, 400):
            n = int(round(target_events / MI_PROP_47Y))
            R = R_D_LARGE if target_events >= 400 else R_D_SMALL
            dcs, lrsig, cc = [], [], []
            for _ in range(R):
                rng = new_rng()
                dv = gen(n, rng, h_mi, HORIZON, rho_cac=rho); va = gen(n, rng, h_mi, HORIZON, rho_cac=rho)
                evd = dv['status'] == 1; evv = va['status'] == 1
                if evd.sum() < 8 or evv.sum() < 8: continue
                Xcd, Xfd = mats(dv); Xcv, Xfv = mats(va)
                try:
                    b1, _, l1, _ = cox_fit(Xcd, dv['t'], evd); b2, _, l2, _ = cox_fit(Xfd, dv['t'], evd)
                except np.linalg.LinAlgError:
                    continue
                c1 = harrell_c(va['t'], evv, Xcv @ b1); c2 = harrell_c(va['t'], evv, Xfv @ b2)
                dcs.append(c2 - c1); cc.append(c1)
                lrsig.append(stats.chi2.sf(2 * (l2 - l1), 1) < 0.05)
            dcs = np.array(dcs)
            out.append(dict(rho_plaque_CAC=rho, target_events=target_events, n_per_cohort=n, reps=int(len(dcs)),
                            true_C_comparator=c_comp_true, true_delta_C=true_dc,
                            mean_C_comparator_validation=float(np.mean(cc)),
                            delta_C_median=float(np.median(dcs)),
                            delta_C_2_5=float(np.percentile(dcs, 2.5)), delta_C_97_5=float(np.percentile(dcs, 97.5)),
                            pct_delta_C_le_0=float(100 * np.mean(dcs <= 0)),
                            pct_LR_test_significant_in_development=float(100 * np.mean(lrsig))))
    return out


# ============================================================================= E
def experiment_E(h_mi_unused=None):
    years = 5.0
    h_mi, h_d = calibrate_both(0.025, 0.04, years)
    rng = new_rng()
    d = gen(N_BIG, rng, h_mi, years, h_d=h_d)
    n = N_BIG
    # scan-triggered early revascularisation (within 90 days), driven by obstructive stenosis on CCTA
    p_early = np.where(d['sten'] == 1, 0.40, 0.01)
    early = rng.random(n) < p_early
    t_early = np.where(early, rng.uniform(0.0, 0.25, n), np.inf)
    # later, symptom-driven revascularisation (modestly plaque-related)
    h_lr = 0.005
    t_late = rng.exponential(1.0, n) / (h_lr * np.exp(0.5 * d['A']))
    t_admin = np.full(n, years)
    t_mi, t_d = d['t_mi'], d['t_d']
    x = d['log_tpv']
    res = {}
    # (i) MI only: death censors; revascularisation assumed not to change MI risk
    tt = np.minimum.reduce([t_mi, t_d, t_admin]); ev = (tt == t_mi)
    # (ii) MACE including early revascularisation
    tt2 = np.minimum.reduce([t_mi, t_d, t_early, t_late, t_admin]); ev2 = tt2 < years
    # (iii) MACE excluding scan-triggered early revascularisation
    tt3 = np.minimum.reduce([t_mi, t_d, t_late, t_admin]); ev3 = tt3 < years
    for key, (T, E) in {"MI_only": (tt, ev), "MACE_incl_early_revasc": (tt2, ev2),
                        "MACE_excl_early_revasc": (tt3, ev3)}.items():
        b, se, _, _ = cox_fit(x, T, E)
        res[key] = dict(events_pct=float(100 * E.mean()), C_index=harrell_c(T, E.astype(int), x),
                        HR_per_SD_log_TPV=float(np.exp(b[0] * x.std())))
    first = np.argmin(np.column_stack([t_mi, t_d, t_early, t_late, t_admin]), axis=1)
    mace = ev2
    res["share_of_MACE_that_is_early_revasc_pct"] = float(100 * np.mean(first[mace] == 2))
    res["early_revasc_pct_of_cohort"] = float(100 * early.mean())
    return res


# ============================================================================= run
def r3(o):
    if isinstance(o, dict): return {k: r3(v) for k, v in o.items()}
    if isinstance(o, list): return [r3(v) for v in o]
    if isinstance(o, (float, np.floating)): return float(round(float(o), 4))
    if isinstance(o, (np.integer,)): return int(o)
    return o


if __name__ == "__main__" or True:
    print(f"numba={'yes' if HAVE_NUMBA else 'no (slower)'}  QUICK={QUICK}  seed={SEED}")
    chk = self_checks(); print("self-checks:", chk)
    h_mi = calibrate_hmi(MI_PROP_47Y, HORIZON)
    print(f"calibrated MI baseline hazard = {h_mi:.6f}/y"); sys.stdout.flush()
    t = time.time(); A = experiment_A(h_mi); print(f"A done in {time.time()-t:.0f}s"); sys.stdout.flush()
    t = time.time(); B = experiment_B(h_mi); print(f"B done in {time.time()-t:.0f}s"); sys.stdout.flush()
    t = time.time(); Cc = experiment_C(); print(f"C done in {time.time()-t:.0f}s"); sys.stdout.flush()
    t = time.time(); D = experiment_D(h_mi); print(f"D done in {time.time()-t:.0f}s"); sys.stdout.flush()
    t = time.time(); E = experiment_E(); print(f"E done in {time.time()-t:.0f}s"); sys.stdout.flush()

    results = r3(dict(meta=dict(seed=SEED, quick=QUICK, numba=HAVE_NUMBA, R_A=R_A, R_D_small=R_D_SMALL,
                                R_D_large=R_D_LARGE, N_big=N_BIG, python=platform.python_version(),
                                numpy=np.__version__, runtime_s=time.time() - T0, self_checks=chk,
                                model=dict(HR_per_SD_plaque=2.0, HR_per_doubling_TPV=float(np.exp(LN2 * LN2)),
                                           HR_per_SD_clinical=1.5, MI_prop=MI_PROP_47Y, horizon=HORIZON)),
                      A_threshold_optimism=A, B_platform_shift=B, C_competing_death=Cc,
                      D_incremental_C=D, E_composite_revasc=E))
    with open("ccta_sim_results.json", "w") as f:
        json.dump(results, f, indent=1)

    # ------------------------------------------------------------------ readable tables
    print("\nA. Outcome-selected threshold (optimal log-rank cut) vs independent validation")
    print(pd.DataFrame([{k: a[k] for k in ("effect", "target_events", "n_per_cohort", "apparent_HR_median", "validation_HR_median",
                                           "true_HR_of_chosen_cut_median", "optimism_ratio_median",
                                           "pct_validation_sig", "chosen_percentile_10_90")} for a in A]).to_string(index=False))
    print("\nB. Fixed mm3 threshold (75th percentile on reference platform) applied to other platforms")
    print(pd.DataFrame([{"platform": r["platform"], "%high": round(r["fixed_threshold"]["pct_classified_high"], 1),
                         "sens%": round(r["fixed_threshold"]["sensitivity_for_MI"], 1),
                         "RR": round(r["fixed_threshold"]["risk_ratio"], 2),
                         "sens% recalibrated": round(r["platform_specific_75th"]["sensitivity_for_MI"], 1)}
                        for r in B["rows"]]).to_string(index=False))
    print("\nC. Competing death")
    print(pd.DataFrame([{"scenario": c["scenario"], "CIF%": round(c["true_MI_cuminc_pct"], 2),
                         "1-KM%": round(c["one_minus_KM_pct"], 2), "over%": round(c["relative_overestimation_pct"], 1),
                         "top-decile pred/obs": round(c["top_decile"]["over_prediction_ratio"], 2)} for c in Cc]).to_string(index=False))
    print("\nD. Incremental C of plaque over risk score + CAC + stenosis (independent validation)")
    print(pd.DataFrame([{k: d[k] for k in ("rho_plaque_CAC", "target_events", "true_delta_C", "delta_C_median",
                                           "delta_C_2_5", "delta_C_97_5", "pct_delta_C_le_0",
                                           "pct_LR_test_significant_in_development")} for d in D]).round(4).to_string(index=False))
    print("\nE. Composite endpoints with scan-triggered revascularisation")
    print(json.dumps(r3(E), indent=1))

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    ax = ax.ravel()
    for eff, ls in (("strong", "-"), ("weak", ":")):
        sub = [a for a in A if a["effect"].startswith(eff)]
        ev = [a["target_events"] for a in sub]
        ax[0].plot(ev, [a["apparent_HR_median"] for a in sub], "o" + ls, color="C0", label=f"Apparent ({eff})")
        ax[0].plot(ev, [a["validation_HR_median"] for a in sub], "s" + ls, color="C1", label=f"Validation ({eff})")
        ax[0].plot(ev, [a["true_HR_of_chosen_cut_median"] for a in sub], "d" + ls, color="C2", label=f"True HR of chosen cut ({eff})")
    ax[0].set_xscale("log"); ax[0].set_xlabel("MI events"); ax[0].set_ylabel("HR above vs below cut")
    ax[0].set_title("A. Outcome-selected threshold"); ax[0].legend(fontsize=6)
    lab = [r["platform"].replace(", less noise (PCD-like)", "\n(PCD-like)") for r in B["rows"]]
    ax[1].bar(np.arange(len(lab)) - 0.2, [r["fixed_threshold"]["sensitivity_for_MI"] for r in B["rows"]], 0.4, label="Fixed mm³ threshold")
    ax[1].bar(np.arange(len(lab)) + 0.2, [r["platform_specific_75th"]["sensitivity_for_MI"] for r in B["rows"]], 0.4, label="Platform-specific 75th pct")
    ax[1].set_xticks(range(len(lab))); ax[1].set_xticklabels(lab, fontsize=6, rotation=20)
    ax[1].set_ylabel("% of MI events classified high"); ax[1].set_title("B. Platform shift"); ax[1].legend(fontsize=7)
    xs = np.arange(len(Cc))
    ax[2].bar(xs - 0.2, [c["true_MI_cuminc_pct"] for c in Cc], 0.4, label="Cumulative incidence (true)")
    ax[2].bar(xs + 0.2, [c["one_minus_KM_pct"] for c in Cc], 0.4, label="1 − Kaplan–Meier")
    ax[2].set_xticks(xs); ax[2].set_xticklabels(["Younger", "Older", "High-risk"], fontsize=8)
    ax[2].set_ylabel("MI risk at horizon (%)"); ax[2].set_title("C. Competing death"); ax[2].legend(fontsize=7)
    for j, rho in enumerate((0.3, 0.6, 0.85)):
        sub = [d for d in D if d["rho_plaque_CAC"] == rho]
        xx = np.arange(len(sub)) + j * 0.25 - 0.25
        ax[3].errorbar(xx, [d["delta_C_median"] for d in sub],
                       yerr=[[d["delta_C_median"] - d["delta_C_2_5"] for d in sub], [d["delta_C_97_5"] - d["delta_C_median"] for d in sub]],
                       fmt="o", capsize=2, label=f"corr(plaque, CAC) = {rho}")
    ax[3].axhline(0, color="grey", lw=0.8); ax[3].set_xticks(range(3)); ax[3].set_xticklabels(["41", "82", "400"])
    ax[3].set_xlabel("MI events"); ax[3].set_ylabel("ΔC (validation), median & 95% range")
    ax[3].set_title("D. Incremental C over strong comparator"); ax[3].legend(fontsize=7)
    keys = ["MI_only", "MACE_excl_early_revasc", "MACE_incl_early_revasc"]
    ax[4].bar(range(3), [E[k]["C_index"] for k in keys])
    ax[4].set_xticks(range(3)); ax[4].set_xticklabels(["MI only", "MACE excl.\nearly revasc", "MACE incl.\nearly revasc"], fontsize=8)
    ax[4].set_ylim(0.5, max(E[k]["C_index"] for k in keys) + 0.05); ax[4].set_ylabel("C-index of plaque volume")
    ax[4].set_title("E. Composite endpoints")
    ax[5].axis("off")
    ax[5].text(0, 0.5, "Simulated data only.\nSee ccta_sim_results.json", fontsize=9)
    fig.tight_layout(); fig.savefig("ccta_sim_figure.png", dpi=300)

    print("\n" + "=" * 30 + " RESULTS: COPY FROM HERE " + "=" * 30)
    print(json.dumps(results, separators=(",", ":")))
    print("=" * 30 + " RESULTS: COPY TO HERE " + "=" * 32)
    print(f"Total runtime {time.time()-T0:.0f}s. Files saved: ccta_sim_results.json, ccta_sim_figure.png")
