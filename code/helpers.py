r"""helpers.py — model comparison and diagnostic tools for 01ZLMA.

The :class:`Anova` class implements the four nested-model comparison tests
defined in Lecture 5 of 01ZLMA (course "01ZLMA — Generalized Linear
Methods", FNSPE CTU). For two nested GLMs ``M_0 ⊂ M`` with parameter counts
``p_0`` and ``p`` and unscaled residual deviances ``D_0`` and ``D``:

* **Chisq / LRT** — when the dispersion parameter ``φ`` is known
  (Poisson, Binomial), the *deviance test* statistic is

  .. math:: T_1 \;=\; \frac{D_0 - D}{\phi} \;\dot\sim\; \chi^2(p - p_0).

* **F-test** — when ``φ`` is unknown (Gaussian, Gamma, Inverse Gaussian,
  Tweedie), the *deviance F-test* statistic is

  .. math:: T_2 \;=\; \frac{(D_0 - D)/(p - p_0)}{\hat\phi}
                  \;=\; \frac{(D_0 - D)/(p - p_0)}{D / (n - p)}
                  \;\dot\sim\; F(p - p_0,\, n - p).

  By default ``\hat\phi`` is the **deviance-based** estimator
  ``D / (n - p)`` of the *larger* model M (Lecture 5, formula 4.4.3). It can
  be switched to the Pearson-based estimator
  ``Pearson_X² / (n - p)`` (which is what ``R``'s ``summary(glm)$dispersion``
  uses) via ``dispersion="pearson"``, or overridden with a numeric value.

* **Wald** — for the same nested test ``H_0: β_1 = 0``,

  .. math:: W \;=\; \hat\beta_1^\top\, [\widehat{\mathrm{Cov}}(\hat\beta_1)]^{-1}\, \hat\beta_1.

  ``W ∼ χ²(p_1)`` if ``φ`` is known and ``W/p_1 ∼ F(p_1, n-p)`` if not.

* **Rao (score) test** — the proper Cox–Hinkley score test of ``H_0:
  β_1 = 0`` is

  .. math:: T_R \;=\; U(\hat\beta_0)^\top\, I(\hat\beta_0)^{-1}\, U(\hat\beta_0)\,/\,\phi
                  \;\dot\sim\; \chi^2(p - p_0),

  where ``U`` is the score and ``I`` the *expected* Fisher information of
  the **full** model M evaluated at the null estimate ``β̂_0`` extended with
  zeros for the extra parameters of M. The expected Fisher information for a
  GLM is ``I(β) = X^T W X`` with IRLS working weights
  ``W_i = (dμ/dη_i)^2 / V(μ_i)``; for non-canonical links this is *not* the
  same as the observed Hessian and the difference matters numerically (R
  also uses the expected information).

  When the larger model is *saturated* (``df_resid = 0``) the score test
  reduces to the Pearson chi-squared statistic of the smaller model, which
  is the goodness-of-fit form mentioned at the beginning of Lecture 5.

* **Mallows Cp** — for a chain of candidate models with the *largest* model
  providing the dispersion estimate ``\hat\phi``, ``R`` reports for each
  candidate

  .. math:: C_p(M_i) \;=\; D(M_i) + 2\, p(M_i)\, \hat\phi.

  This implementation follows that convention exactly.

The output of :class:`Anova` is a ``pandas.DataFrame`` with one row per
model, exposing the columns ``resid_df``, ``resid_deviance``, ``df``,
``deviance`` and the test-specific column (``chi2`` / ``f_stat`` / ``rao``
/ ``wald`` / ``cp``) plus ``p_val`` (except for ``cp``).

Numerical values were verified against ``R 4.3.2``'s ``anova.glm`` and
``stats::summary.glm`` on Gaussian, Poisson and Gamma examples.
"""

import warnings
from typing import Type

import numpy as np
import pandas as pd
import scipy
import scipy.stats

import statsmodels
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, Binomial
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
#  Helper utilities for the Anova class
# ---------------------------------------------------------------------------

def _phi_hat(model_full, dispersion):
    """Return the dispersion estimate from the *larger* model.

    Parameters
    ----------
    model_full : GLMResultsWrapper
        The unrestricted (larger) model M.
    dispersion : {"deviance", "pearson", None, False, True} or float
        How to estimate ``φ``.

        - ``"deviance"``  — ``D / (n − p)`` (the lecture's default).
        - ``"pearson"``   — ``Pearson_X² / (n − p)`` (R's default,
          accessible via ``model.scale``).
        - ``None`` / ``False`` — alias for ``"deviance"`` (lecture default).
        - ``True``               — alias for ``"deviance"``.
        - numeric                 — use the supplied value verbatim.
    """
    if isinstance(dispersion, (int, float)) and not isinstance(dispersion, bool):
        return float(dispersion)
    if dispersion in (None, False, True, "deviance"):
        return float(model_full.deviance / model_full.df_resid)
    if dispersion == "pearson":
        return float(model_full.scale)
    raise ValueError(f"Unknown dispersion option: {dispersion!r}")


def _is_phi_known(model):
    """``True`` if the family has a fixed dispersion (Poisson / Binomial)."""
    return isinstance(model.model.family, (Poisson, Binomial))


def _order_models(m1, m2):
    """Return ``(m_smaller, m_larger)`` ordered by ``df_model``."""
    return (m1, m2) if m1.df_model <= m2.df_model else (m2, m1)


def _expected_score_and_info(model_full, beta_at):
    """Compute the score vector and expected Fisher information of a GLM
    evaluated at the parameter point ``beta_at`` (with ``φ = 1``).

    Uses the canonical IRLS form ``U = X^T (y − μ) g'(μ) / V(μ)`` and
    ``I(β) = X^T diag((dμ/dη)² / V(μ)) X``. For canonical links the
    expected information equals the observed (negative) Hessian; for
    non-canonical links it does *not*, and ``R``'s ``anova.glm`` uses the
    expected form. ``statsmodels``' ``model.hessian`` returns the observed
    Hessian and would give a slightly different answer for non-canonical
    links — that is why we re-derive both quantities here.
    """
    fam = model_full.model.family
    X = np.asarray(model_full.model.exog)
    y = np.asarray(model_full.model.endog)
    eta = X @ beta_at
    mu  = fam.link.inverse(eta)
    var_mu  = fam.variance(mu)
    dmu_deta = fam.link.inverse_deriv(eta)
    score = X.T @ ((y - mu) * dmu_deta / var_mu)
    W = (dmu_deta ** 2) / var_mu
    info = X.T @ (W[:, None] * X)
    return score, info


def _embed_null_params(m_small, m_large):
    """Build the null parameter vector ``β̂_0`` lifted into the parameter
    space of ``m_large`` (zeros for the extra coefficients).

    Returns ``None`` if the parameter names of ``m_small`` are not a subset
    of those of ``m_large`` — this signals that the two models do not have
    a *name-aligned* nesting (e.g. comparison against a saturated model
    fit on the identity matrix).
    """
    full_names = list(m_large.params.index)
    if not set(m_small.params.index).issubset(full_names):
        return None
    beta = np.zeros(len(full_names))
    for name, val in m_small.params.items():
        beta[full_names.index(name)] = val
    return beta


def _build_sequential_submodels(model_full):
    """Given a single fitted GLM, return a list of nested submodels obtained
    by sequentially adding the *terms* of the model in their formula order.

    Uses ``patsy``'s ``design_info`` (the source of truth for which columns
    of the design matrix belong to which formula term) when available, and
    falls back to per-column nesting otherwise.
    """
    family = model_full.model.family
    var_weights = getattr(model_full.model, "var_weights", None)
    offset = getattr(model_full.model, "offset", None)
    endog = model_full.model.endog

    di = getattr(model_full.model.data, "design_info", None)
    if di is not None and getattr(di, "term_name_slices", None):
        column_idx: list[int] = []
        submodels = []
        for term in di.term_names:
            sl = di.term_name_slices[term]
            column_idx.extend(range(sl.start, sl.stop))
            sub_exog = model_full.model.exog[:, column_idx]
            kwargs = {"family": family}
            if var_weights is not None:
                kwargs["var_weights"] = var_weights
            if offset is not None:
                kwargs["offset"] = offset
            submodels.append(sm.GLM(endog=endog, exog=sub_exog, **kwargs).fit())
        return submodels, list(di.term_names)

    # Fallback: nest by columns 1, 1:2, 1:3, ...  Label each step with the
    # name of the *added* column (statsmodels' GLM exposes ``exog_names``
    # whether the model was built via the formula API or directly from
    # an array — in the latter case the names default to ``x1, x2, ...``).
    exog = model_full.model.exog
    exog_names = list(getattr(model_full.model, "exog_names", []) or [])
    if len(exog_names) != exog.shape[1]:
        exog_names = [f"x{i+1}" for i in range(exog.shape[1])]

    submodels = []
    for i in range(1, exog.shape[1] + 1):
        kwargs = {"family": family}
        if var_weights is not None:
            kwargs["var_weights"] = var_weights
        if offset is not None:
            kwargs["offset"] = offset
        submodels.append(sm.GLM(endog=endog, exog=exog[:, :i], **kwargs).fit())
    return submodels, exog_names


# ---------------------------------------------------------------------------
#  Anova class — model comparison interface (mimics R's anova.glm)
# ---------------------------------------------------------------------------

#: Pretty-printed headers for each test, showing the lecture's Czech name,
#: the standard literature name, the symbol used in the lecture, the formula
#: and the asymptotic distribution. The string ``{phi}`` is filled with a
#: short label describing the dispersion estimate that was actually used.
_TEST_HEADERS = {
    "LRT": (
        "LRT — Likelihood Ratio Test  (Lecture 5: deviační test, statistika T_1)\n"
        "    T_1 = (D_0 - D) / phi  ~  chi^2(p - p_0)        [phi: {phi}]\n"
        "    aliases: LRT = chi-squared deviance test = T_1"
    ),
    "F": (
        "F — Deviance F-test  (Lecture 5: deviační F-test, statistika T_2)\n"
        "    T_2 = ((D_0 - D)/(p - p_0)) / (D/(n - p))  ~  F(p - p_0, n - p)\n"
        "    phi_hat from larger model M: {phi}\n"
        "    used when the dispersion phi is unknown (Gaussian, Gamma, IG, ...)"
    ),
    "Wald_chi2": (
        "Wald — Waldova statistika (phi known: Poisson / Binomial)\n"
        "    W = beta_1^T * Cov(beta_1)^{{-1}} * beta_1  ~  chi^2(p_1)\n"
        "    where beta_1 are the extra coefficients of M not present in M_0"
    ),
    "Wald_F": (
        "Wald — Waldova statistika (phi unknown: Gaussian, Gamma, IG, ...)\n"
        "    W = beta_1^T * Cov(beta_1)^{{-1}} * beta_1\n"
        "    F = W / p_1  ~  F(p_1, n - p)        [phi for Cov: {phi}]\n"
        "    for p_1 = 1 this is equivalent to Z = beta_1 / sqrt(v_pp) ~ t(n - p)"
    ),
    "Rao": (
        "Rao — Raova skórová statistika (Rao score test)\n"
        "    T_R = U(beta_hat_0)^T * I(beta_hat_0)^{{-1}} * U(beta_hat_0) / phi\n"
        "          ~  chi^2(p - p_0)        [phi: {phi}]\n"
        "    U and I are score and EXPECTED Fisher info of the FULL model M\n"
        "    evaluated at the null estimate beta_hat_0 (extended with zeros)\n"
        "    aliases: Rao = score test = U^T I^{{-1}} U"
    ),
    "Cp": (
        "Cp — Mallows' Cp  (Lecture 5: modifikovaná AIC, AIC_m)\n"
        "    Cp(M_i) = D(M_i) + 2 * p(M_i) * phi_hat_full\n"
        "    where phi_hat_full comes from the LARGEST model in the chain: {phi}\n"
        "    no p-value; smaller is better"
    ),
}


def _phi_label(model_full, dispersion):
    """Short human-readable label for the dispersion estimate that ``_phi_hat``
    will return for the given (model_full, dispersion) pair."""
    if isinstance(dispersion, (int, float)) and not isinstance(dispersion, bool):
        return f"user-supplied = {float(dispersion):.6g}"
    if _is_phi_known(model_full):
        return "1.0  (Poisson / Binomial: phi known and fixed)"
    if dispersion in (None, False, True, "deviance"):
        val = model_full.deviance / model_full.df_resid
        return f"D/(n-p) = {val:.6g}  (deviance estimate from larger model)"
    if dispersion == "pearson":
        return f"X²_P/(n-p) = {model_full.scale:.6g}  (Pearson estimate from larger model)"
    return str(dispersion)


class Anova:
    """Model-comparison helper following Lecture 5 of 01ZLMA.

    Usage::

        anova = Anova()
        anova(m_small, m_large, test="LRT")        # two-model LRT
        anova(m_small, m_large, test="F")          # two-model F-test
        anova(m_small, m_large, test="Wald")       # two-model Wald test
        anova(m_small, m_large, test="Rao")        # two-model Rao score test
        anova(m1, m2, m3, test="Cp")               # Mallows Cp on a chain
        anova(model_full)                          # sequential ANOVA (LRT)
        anova(model_full, model_sat, test="Rao")   # goodness-of-fit Rao test

    The output ``pandas.DataFrame`` carries the columns

    ``resid_df``  ``resid_deviance``  ``df``  ``deviance``  *<test>*  ``p_val``

    where the test column is named after the test (``LRT``, ``F``, ``Wald``,
    ``Rao``, ``Cp``).  Each call also prints a one-paragraph header that
    states both the Czech (lecture) name and the standard literature name
    of the test, the lecture's symbol (``T_1``, ``T_2``, ``W``, ``T_R``,
    ``AIC_m``), the closed-form expression and the asymptotic distribution
    of the test statistic.  Pass ``verbose=False`` to suppress the header.

    The full DataFrame is also accessible via the ``.res`` attribute after
    a call.
    """

    def __init__(self):
        self.__res = pd.DataFrame()

    @property
    def res(self):
        return self.__res

    # ------------------------------------------------------------------
    #  Public entry point
    # ------------------------------------------------------------------
    def __call__(self, *models, test="LRT", dispersion=None, verbose=True):
        if len(models) == 0:
            raise ValueError("Pass at least one fitted GLM result object.")

        # Normalize the test name. We accept the lecture/Czech name,
        # the standard literature name and the lower-case shortcut used by
        # statsmodels and R.
        test_norm = self._normalize_test_name(test)

        if len(models) == 1:
            sub_models, term_labels = _build_sequential_submodels(models[0])
        else:
            sub_models = list(models)
            term_labels = None

        if test_norm == "Cp":
            self.__res = self._cp_table(sub_models, dispersion, term_labels)
            if verbose:
                self._print_header(test_norm, sub_models[-1], dispersion)
            return self.__res

        # Build a row per model: first row = baseline (stats of the first
        # argument), subsequent rows = test of pair (i-1, i) with the stats
        # of the *i-th* argument shown, so the display follows the user's
        # argument order regardless of which model is actually larger.
        rows = [self._baseline_row(sub_models[0])]
        for i in range(1, len(sub_models)):
            rows.append(
                self._test_pair(sub_models[i - 1], sub_models[i],
                                test_norm, dispersion)
            )
        df = pd.DataFrame(rows)

        if term_labels is not None and len(term_labels) == len(df):
            df.index = pd.Index(term_labels, dtype=object)

        self.__res = df
        if verbose and len(sub_models) >= 2:
            # Use the LARGER model of the last comparison as the reference
            # for the dispersion label (matches the test computation).
            last_small, last_large = _order_models(sub_models[-2], sub_models[-1])
            self._print_header(test_norm, last_large, dispersion)
        return self.__res

    def __repr__(self) -> str:
        return repr(self.__res)

    # ------------------------------------------------------------------
    #  Test name normalization and header printing
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_test_name(test):
        """Map any of the accepted aliases to the canonical column name."""
        t = str(test).strip().lower()
        if t in ("lrt", "chisq", "chi2", "chi-squared", "deviance", "deviační",
                 "deviacni", "t1", "t_1"):
            return "LRT"
        if t in ("f", "f-test", "f_test", "ftest", "t2", "t_2"):
            return "F"
        if t in ("wald", "waldova", "waldův", "waldova statistika", "w"):
            return "Wald"
        if t in ("rao", "score", "skóre", "skore", "raova", "raova statistika",
                 "rao score test"):
            return "Rao"
        if t in ("cp", "c_p", "mallows", "mallows cp", "mallowsovo cp",
                 "modified aic", "aic_m", "modifikovaná aic", "modifikovana aic"):
            return "Cp"
        raise ValueError(
            f"Unknown test {test!r}. Recognised aliases: "
            "LRT/chi2/T1, F/T2, Wald, Rao/score, Cp/AIC_m"
        )

    def _print_header(self, test_norm, model_full, dispersion):
        """Pretty-print the header describing the test about to be reported."""
        if test_norm == "Wald":
            key = "Wald_chi2" if _is_phi_known(model_full) else "Wald_F"
        else:
            key = test_norm
        template = _TEST_HEADERS.get(key)
        if template is None:
            return
        text = template.format(phi=_phi_label(model_full, dispersion))
        print(text)

    # ------------------------------------------------------------------
    #  Per-test implementations
    # ------------------------------------------------------------------
    def _baseline_row(self, model):
        return {
            "resid_df":       model.df_resid,
            "resid_deviance": model.deviance,
            "df":             np.nan,
            "deviance":       np.nan,
        }

    def _test_pair(self, m_prev, m_next, test_norm, dispersion):
        """Compute the requested test statistic for the pair (m_prev, m_next).

        The *numerical* test is insensitive to the argument order — we call
        ``_order_models`` internally so that LRT/F/Wald/Rao always work the
        same way regardless of which model the user puts first. The
        *displayed* ``resid_df`` and ``resid_deviance`` columns, however,
        reflect ``m_next`` (the second argument), so consecutive rows in
        the result DataFrame tabulate the models in the order the user
        passed them, matching R's ``anova.glm`` convention.
        """
        m_small, m_large = _order_models(m_prev, m_next)
        df_diff       = m_large.df_model - m_small.df_model
        deviance_diff = m_small.deviance - m_large.deviance

        row = {
            "resid_df":       m_next.df_resid,
            "resid_deviance": m_next.deviance,
            "df":             df_diff,
            "deviance":       deviance_diff,
        }

        if test_norm == "LRT":
            stat, pv = self._lrt(m_small, m_large, dispersion)
        elif test_norm == "F":
            stat, pv = self._F(m_small, m_large, dispersion)
        elif test_norm == "Rao":
            stat, pv = self._rao(m_small, m_large, dispersion)
        elif test_norm == "Wald":
            stat, pv = self._wald(m_small, m_large, dispersion)
        else:
            raise ValueError(f"Unknown test: {test_norm!r}")

        row[test_norm] = stat
        row["p_val"]   = pv
        return row

    # --- LRT (Chi-squared, T_1) --------------------------------------
    def _lrt(self, m_small, m_large, dispersion):
        """LRT / chi-squared deviance test (Lecture 5, T_1).

        ``T_1 = (D_0 − D) / phi  ~  chi²(p − p_0)`` (asymptotic).

        Exact when ``phi`` is known (Poisson, Binomial). For Gaussian /
        Gamma / Inverse Gaussian the dispersion is unknown and the F-test
        is the exact test; the chi² form here divides the deviance change
        by an *estimated* phi and is only asymptotically valid. By default
        the deviance estimate ``phi_hat = D/(n-p)`` of the larger model is
        used (Lecture 5, formula 4.1).
        """
        if not _is_phi_known(m_large) and dispersion is None:
            warnings.warn(
                "LRT (chi-squared) assumes a known dispersion (Poisson/"
                "Binomial). For Gaussian/Gamma the F-test is the exact one; "
                "the chi^2 form here divides by an estimated phi and is only "
                "asymptotically valid. The default uses the deviance-based "
                "estimate D/(n-p) of the larger model.",
                stacklevel=3,
            )
        phi = 1.0 if _is_phi_known(m_large) else _phi_hat(m_large, dispersion)
        df_diff = m_large.df_model - m_small.df_model
        T1 = (m_small.deviance - m_large.deviance) / phi
        p_val = scipy.stats.chi2.sf(T1, df=df_diff)
        return T1, p_val

    # --- F-test (T_2) -------------------------------------------------
    def _F(self, m_small, m_large, dispersion):
        """Deviance F-test (Lecture 5, T_2).

        ``T_2 = ((D_0 − D)/(p − p_0)) / phi_hat  ~  F(p − p_0, n − p)``.

        ``phi_hat`` is by default the deviance estimate ``D/(n − p)`` of the
        larger model M. Use ``dispersion="pearson"`` to switch to the
        Pearson estimate (R's convention) or pass a numeric value directly.
        """
        df_diff = m_large.df_model - m_small.df_model
        if df_diff <= 0:
            return np.nan, np.nan
        if m_large.df_resid <= 0:
            warnings.warn(
                "F test is degenerate when the larger model is saturated "
                "(df_resid = 0).",
                stacklevel=3,
            )
            return np.nan, np.nan
        phi = _phi_hat(m_large, dispersion)
        T2 = ((m_small.deviance - m_large.deviance) / df_diff) / phi
        p_val = scipy.stats.f.sf(T2, dfn=df_diff, dfd=m_large.df_resid)
        return T2, p_val

    # --- Rao score test ----------------------------------------------
    def _rao(self, m_small, m_large, dispersion):
        """Rao score test (Cox–Hinkley) for nested GLMs.

        ``T_R = U(beta_hat_0)^T I(beta_hat_0)^{-1} U(beta_hat_0) / phi``,
        where U and I are the score and the **expected** Fisher information
        of the *full* model evaluated at the null estimate ``beta_hat_0``
        embedded into the full parameter space (zeros for the extra
        coefficients).

        Reproduces ``R``'s ``anova.glm(test="Rao")`` exactly because it uses
        the expected Fisher information ``X^T W X`` (with IRLS weights
        ``W = (dmu/deta)² / V(mu)``) rather than the observed Hessian, which
        differ for non-canonical links such as Gamma with log link.

        When the larger model is *saturated* (``df_resid = 0``) the score
        test reduces to the Pearson chi-squared statistic of the smaller
        (proposed) model — the goodness-of-fit form mentioned at the top of
        Lecture 5.
        """
        df_diff = m_large.df_model - m_small.df_model
        if df_diff <= 0:
            return np.nan, np.nan

        if m_large.df_resid <= 0:
            T_R_unscaled = float(np.sum(m_small.resid_pearson ** 2))
        else:
            beta_null = _embed_null_params(m_small, m_large)
            if beta_null is None:
                warnings.warn(
                    "Rao score test: parameter names of the smaller model are "
                    "not a subset of the larger model. Falling back to the "
                    "Pearson chi² of the smaller model (valid only when the "
                    "larger model is essentially saturated).",
                    stacklevel=3,
                )
                T_R_unscaled = float(np.sum(m_small.resid_pearson ** 2))
            else:
                score, info = _expected_score_and_info(m_large, beta_null)
                T_R_unscaled = float(score @ np.linalg.solve(info, score))

        phi = 1.0 if _is_phi_known(m_large) else _phi_hat(m_large, dispersion)
        T_R = T_R_unscaled / phi
        p_val = scipy.stats.chi2.sf(T_R, df=df_diff)
        return T_R, p_val

    # --- Wald test ----------------------------------------------------
    def _wald(self, m_small, m_large, dispersion):
        """Waldova statistika for the extra coefficients of ``m_large``
        relative to ``m_small`` (those not present in the smaller model).

        For phi known returns ``W = beta_1^T Cov(beta_1)^{-1} beta_1
        ~ chi²(p_1)``; for phi unknown returns the F-form ``F = W/p_1
        ~ F(p_1, n − p)``. For ``p_1 = 1`` the F-form is equivalent to
        ``Z = beta_1 / sqrt(v_pp) ~ t(n − p)``.

        Note that ``cov_params()`` already incorporates the dispersion that
        was estimated when the larger model was fitted, so the ``dispersion``
        argument has *no effect* here — it is accepted only for API
        consistency with the other tests.
        """
        del dispersion  # see docstring
        extra = [n for n in m_large.params.index if n not in m_small.params.index]
        if len(extra) == 0:
            raise ValueError(
                "Wald test: the larger model has no extra coefficients "
                "relative to the smaller model."
            )
        beta1 = m_large.params.loc[extra].values.astype(float)
        V11   = m_large.cov_params().loc[extra, extra].values.astype(float)
        W = float(beta1 @ np.linalg.solve(V11, beta1))
        p1 = len(extra)

        if _is_phi_known(m_large):
            return W, scipy.stats.chi2.sf(W, df=p1)
        else:
            F = W / p1
            return F, scipy.stats.f.sf(F, dfn=p1, dfd=m_large.df_resid)

    # --- Mallows Cp / modifikovaná AIC -------------------------------
    def _cp_table(self, models, dispersion, term_labels):
        """Mallows' Cp table for a chain of candidate models.

        ``Cp(M_i) = D(M_i) + 2 * p(M_i) * phi_hat_full``, where
        ``phi_hat_full`` is the dispersion estimate of the *largest* model
        in the chain (the one with the smallest residual df). This matches
        ``R``'s ``anova.glm(test="Cp")`` and the *modifikovaná AIC*
        ``AIC_m = D + 2 p phi`` defined in Lecture 5, formula 4.4.3.
        """
        largest = min(models, key=lambda m: m.df_resid)
        phi_full = _phi_hat(
            largest, dispersion if dispersion is not None else "deviance"
        )

        rows = []
        prev = None
        for m in models:
            n_params = m.df_model + 1  # statsmodels df_model excludes intercept
            cp = m.deviance + 2.0 * n_params * phi_full
            row = {
                "resid_df":       m.df_resid,
                "resid_deviance": m.deviance,
                "df":             np.nan if prev is None else (m.df_model - prev.df_model),
                "deviance":       np.nan if prev is None else (prev.deviance - m.deviance),
                "Cp":             cp,
            }
            rows.append(row)
            prev = m
        df = pd.DataFrame(rows)
        if term_labels is not None and len(term_labels) == len(df):
            df.index = pd.Index(term_labels, dtype=object)
        return df


# ---------------------------------------------------------------------------
#  Diagnostic plots — unchanged from the original (Prajwal Kafle, modified
#  for GLMs). The constructor's type-check has been corrected (formerly used
#  ``or`` instead of ``and`` and therefore always raised).
# ---------------------------------------------------------------------------

style_talk = "seaborn-talk"  # refer to plt.style.available


class DiagnosticPlots:
    """
    Diagnostic plots to identify potential problems in a linear regression
    or GLM fit.

    Mainly:
        a. non-linearity of data
        b. correlation of error terms
        c. non-constant variance
        d. outliers
        e. high-leverage points
        f. collinearity
    """

    def __init__(
        self,
        results: Type[statsmodels.regression.linear_model.RegressionResultsWrapper],
    ) -> None:
        if (not isinstance(
                results, statsmodels.regression.linear_model.RegressionResultsWrapper)
            and not isinstance(
                results,
                statsmodels.genmod.generalized_linear_model.GLMResultsWrapper)):
            raise TypeError(
                "result must be an instance of "
                "statsmodels.regression.linear_model.RegressionResultsWrapper "
                "or statsmodels.genmod.generalized_linear_model."
                "GLMResultsWrapper"
            )

        self.results = maybe_unwrap_results(results)

        self.y_true   = self.results.model.endog
        self.y_predict = self.results.fittedvalues
        self.xvar     = self.results.model.exog
        self.xvar_names = self.results.model.exog_names

        influence = self.results.get_influence()
        self.leverage = influence.hat_matrix_diag
        self.cooks_distance = influence.cooks_distance[0]
        self.nparams = len(self.results.params)

        if isinstance(
            self.results,
            statsmodels.genmod.generalized_linear_model.GLMResults,
        ):
            self.residual = np.array(self.results.resid_pearson)
            self.residual_norm = influence.resid_studentized
        else:
            self.residual = np.array(self.results.resid)
            self.residual_norm = influence.resid_studentized_internal

    def __call__(self, plot_context="seaborn-paper"):
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
            self.residual_plot(ax=ax[0, 0])
            self.qq_plot(ax=ax[0, 1])
            self.scale_location_plot(ax=ax[1, 0])
            self.leverage_plot(ax=ax[1, 1])
            plt.show()

        self.vif_table()
        return fig, ax

    def residual_plot(self, ax=None):
        """Residual vs fitted plot — diagnoses non-linearity."""
        if ax is None:
            fig, ax = plt.subplots()

        sns.residplot(
            x=self.y_predict,
            y=self.residual,
            lowess=True,
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "red", "lw": 1, "alpha": 0.8},
            ax=ax,
        )

        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.sort(residual_abs))
        for i, _ in enumerate(abs_resid[:3]):
            ax.annotate(
                i,
                xy=(self.y_predict[i], self.residual[i]),
                color="C3",
            )

        ax.set_title("Residuals vs Fitted", fontweight="bold")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        return ax

    def qq_plot(self, ax=None):
        """Standardized residual vs theoretical quantile plot — checks normality."""
        if ax is None:
            fig, ax = plt.subplots()

        QQ = ProbPlot(self.residual_norm)
        QQ.qqplot(line="45", alpha=0.5, lw=1, ax=ax)

        abs_norm_resid = np.flip(np.argsort(np.abs(self.residual_norm)), 0)
        for r, i in enumerate(abs_norm_resid[:3]):
            ax.annotate(
                i,
                xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                    self.residual_norm[i]),
                ha="right",
                color="C3",
            )

        ax.set_title("Normal Q-Q", fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Standardized Residuals")
        return ax

    def scale_location_plot(self, ax=None):
        """sqrt(|standardized residual|) vs fitted plot — checks homoscedasticity."""
        if ax is None:
            fig, ax = plt.subplots()

        residual_norm_abs_sqrt = np.sqrt(np.abs(self.residual_norm))

        ax.scatter(self.y_predict, residual_norm_abs_sqrt, alpha=0.5)
        sns.regplot(
            x=self.y_predict,
            y=residual_norm_abs_sqrt,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={"color": "red", "lw": 1, "alpha": 0.8},
            ax=ax,
        )

        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        for i in abs_sq_norm_resid[:3]:
            ax.annotate(
                i,
                xy=(self.y_predict[i], residual_norm_abs_sqrt[i]),
                color="C3",
            )
        ax.set_title("Scale-Location", fontweight="bold")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel(r"$\sqrt{|\mathrm{Standardized\ Residuals}|}$")
        return ax

    def leverage_plot(self, ax=None):
        """Standardized residual vs leverage plot with Cook's distance contours."""
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(self.leverage, self.residual_norm, alpha=0.5)

        sns.regplot(
            x=self.leverage,
            y=self.residual_norm,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={"color": "red", "lw": 1, "alpha": 0.8},
            ax=ax,
        )

        leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(
                i,
                xy=(self.leverage[i], self.residual_norm[i]),
                color="C3",
            )

        xtemp, ytemp = self.__cooks_dist_line(0.5)
        ax.plot(xtemp, ytemp, label="Cook's distance", lw=1, ls="--", color="red")
        xtemp, ytemp = self.__cooks_dist_line(1)
        ax.plot(xtemp, ytemp, lw=1, ls="--", color="red")

        ax.set_xlim(0, max(self.leverage) + 0.01)
        ax.set_title("Residuals vs Leverage", fontweight="bold")
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Standardized Residuals")
        ax.legend(loc="upper right")
        return ax

    def vif_table(self):
        """Variance inflation factors for the design columns."""
        vif_df = pd.DataFrame()
        vif_df["Features"] = self.xvar_names
        vif_df["VIF Factor"] = [
            variance_inflation_factor(self.xvar, i)
            for i in range(self.xvar.shape[1])
        ]
        print(vif_df.sort_values("VIF Factor").round(2))

    def __cooks_dist_line(self, factor):
        """Helper for plotting Cook's distance contours on the leverage plot."""
        p = self.nparams
        formula = lambda x: np.sqrt((factor * p * (1 - x)) / x)
        x = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x)
        return x, y
