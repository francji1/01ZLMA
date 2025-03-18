import numpy as np
import statsmodels
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels

from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Type
import copy
import math
import warnings


# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/anova.glm

class Anova():
    """Anova callable object to mimic glm.anova function from R for GLM comparison.
       Note that this object does not perform one-way or any other type of anova testing it only serves for comparing GLM models
    """
    def __init__(self):
        self.__res = pd.DataFrame(data={})

    @property
    def res(self):
        return self.__res
        
    def __call__(self, *models, test='chisq', dispersion=False):
        if len(models) == 1:
            """Testing all submodels"""

            link = models[0].model.family.link
            var_weights = models[0].model.var_weights
            offset = None
            if hasattr(models[0].model, 'offset'):
                offset = models[0].model.offset
            family = models[0].model.family
            exog = models[0].model.exog
            un = None
            if hasattr(models[0].params, 'index'):
                var_names = list(models[0].params.index)
                vv = [i.split('_')[0] for i in var_names]
                indexes = np.unique(vv, return_index=True)[1]
                un = [vv[index] for index in sorted(indexes)]
                

                sub_mod_vars = [np.isin(vv, un[:i]) for i in range(1, len(un)+1)]
                sub_mods = [sm.GLM(endog=models[0].model.endog, exog=exog[:, i], family=family, var_weights=var_weights, offset=offset).fit() for i in sub_mod_vars]
            else:
                sub_mods = [sm.GLM(endog=models[0].model.endog, exog=exog[:, :i], family=family, var_weights=var_weights, offset=offset).fit() for i in range(1, exog.shape[1]+1)]


            pairs = self._pairs_(*sub_mods)


        elif len(models) > 1:
            """Comparing models"""
            pairs = self._pairs_(*models)
        else:
            raise Exception(f'None model selected {len(models)}')

    
        out = [self._choice(*i, test=test, dispersion=dispersion) for i in pairs]
        
        if test.lower() == 'cp':
            res = {k: [v if 'resid' in k else v[0] if 'cp' in k else np.nan] for k, v in out[0].items()}
            for i, j in enumerate(out):
                for k, v in j.items():
                    if 'resid' in k:
                        c = k.split('_')[1]
                        res[k].append(res[k][-1] - j[c])
                    elif 'resid' not in k and 'cp' not in k:
                        res[k].append(v)
                    elif 'cp' in k:
                        res[k].append(v[1])

                    
        else:
            res = {k: [v if 'resid' in k else np.nan] for k, v in out[0].items()}
            for i, j in enumerate(out):
                for k, v in j.items():
                    if 'resid' in k:
                        c = k.split('_')[1]
                        res[k].append(res[k][-1] - j[c])
                    elif 'resid' not in k:
                        res[k].append(v)

        self.__res = pd.DataFrame(data=res)

        if len(models) == 1:
            if un is not None:
                self.__res = self.__res.set_index([pd.Index(un, dtype='object')])
            else:
                self.__res = self.__res.set_index([pd.Index([f'x{i}' for i in range(0, exog.shape[1])], dtype='object')])
        #display(self.__res)
        return self.__res
    
    def __repr__(self) -> str:
        return repr(self.__res)

    def _choice(self, *models, test, dispersion):
        if test.lower() in ('chisq'.lower(), 'lrt'):
            res = self._chisq_(*models)
        elif test.lower() == 'f':
            res = self._F_(*models, dispersion=dispersion)
        elif test.lower() == 'rao':
            res = self._rao_(*models)
        elif test.lower() == 'cp':
            res = self._cp_(*models)
        else:
            raise Exception(f'Not such test implemented: {test}')
        return res

    def _pairs_(self, *models):
        return [[models[i], models[i+1]] for i in range(len(models) -1)]

    def _F_(self, *models, dispersion=False):
        """Performs deviance F-test most appropriate when scale param (phi) is not known"""
        phi_hat = models[-1].scale
        if models[0].df_model > models[-1].df_model:
            phi_hat = models[0].scale
        if isinstance(dispersion, bool) and dispersion:
            # deviance estimate of scale factor
            phi_hat  = models[-1].deviance / models[-1].df_resid
        elif isinstance(dispersion, float):
            phi_hat = dispersion

        f_stat = (models[0].deviance - models[-1].deviance) / ((models[-1].df_model - models[0].df_model)* phi_hat)
        p_val = scipy.stats.f.sf(f_stat, dfn=np.abs(models[-1].df_model - models[0].df_model), 
                                 dfd=np.abs(models[-1].nobs - models[-1].df_model))

        return {'resid_df': models[0].df_resid, 'resid_deviance':  models[0].deviance, 'df': models[-1].df_model - models[0].df_model, 'deviance': models[0].deviance - models[-1].deviance,
                'f_stat': f_stat, 'p_val': p_val,  
                }
    def _chisq_(self, *models):
        """Performs deviance LRT test leading to Chisq test statistic (known dispersion param)
        Note that if disperzion param is not known then deviance LRT test leads to F test and
        performing Chisq test is inappropriate
        """
        phi_hat = models[-1].scale
        if models[0].df_model > models[-1].df_model:
            phi_hat = models[0].scale
        chi2_stat = np.abs((models[0].deviance - models[-1].deviance) / phi_hat)  
        p_val = scipy.stats.chi2.sf(chi2_stat, df=np.abs(models[-1].df_model - models[0].df_model))

        return {'resid_df': models[0].df_resid, 'resid_deviance':  models[0].deviance, 'df': models[-1].df_model - models[0].df_model, 'deviance': models[0].deviance - models[-1].deviance,
                'chi2': chi2_stat,'p_val': p_val}

    def _rao_(self, *models):
        """Performs rao score test"""
        # TODO possible incorrect
        phi_hat = models[-1].scale
        if models[0].df_model > models[-1].df_model:
            phi_hat = models[0].scale
       
        raos = [i.resid_pearson.T @ i.resid_pearson for i in models] #
        # reference: https://link.springer.com/content/pdf/10.1007/BF02763005.pdf
        
        #scores = [i.resid_response / i.model.family.variance(i.predict()) for i in models]  # calculates score vectors for both models
        #raos = [i.T  @ (np.diag(j.model.family.variance(j.predict())) ) @ i for i, j in zip(scores, models)] # calculates rao statistic for both models ()

        chi2_stat = (raos[0] - raos[-1])
        p_val = scipy.stats.chi2.sf(np.abs(chi2_stat) / phi_hat, df=np.abs(models[-1].df_model - models[0].df_model))

        warnings.warn("Note that rao score statistic may be inccorect. Currently it is Generalized Pearson statistic")

        return {'resid_df': models[0].df_resid, 'resid_deviance':  models[0].deviance, 'df': models[-1].df_model - models[0].df_model, 'deviance': models[0].deviance - models[-1].deviance,
                'rao': chi2_stat,'p_val': p_val}

    def _cp_(self, *models):
        """No p-values just Cp values"""
        
        return {'resid_df': models[0].df_resid, 'resid_deviance':  models[0].deviance, 'df': models[-1].df_model - models[0].df_model, 'deviance': models[0].deviance - models[-1].deviance,
                'cp': [i.deviance + 2*i.scale*i.df_model for i in models]}


def drop1():
    pass

def add1():
    pass

def step(model, scope, direction='backward'):
    #https://medium.com/@garrettwilliams90/stepwise-feature-selection-for-statsmodels-fda269442556
    pass

def F(*models, estim_scale=False):
        """Performs deviance F-test most appropriate when scale param (phi) is not known"""
        phi_hat = models[0].scale
        if estim_scale:
            phi_hat  = models[-1].deviance / (models[-1].nobs - models[-1].df_model)
    
        f_stat = (models[0].deviance - models[-1].deviance) / ((models[-1].df_model - models[0].df_model)* phi_hat)
        p_val = scipy.stats.f.sf(f_stat, dfn=models[-1].df_model - models[0].df_model, 
                                 dfd=models[-1].nobs - models[-1].df_model)
        print(f'Estimated F statistic is: {f_stat} \n'
              f'P-value is: {p_val} \n'
              )
def chisq(*models):
    """Performs deviance LRT test leading to Chisq test statistic (known dispersion param)
        Note that if disperzion param is not known then deviance LRT test leads to F test and
        performing Chisq test is inappropriate
    """
    chi2_stat = (models[0].deviance - models[-1].deviance) / models[0].scale
    p_val = scipy.stats.chi2.sf(chi2_stat, df=models[-1].df_model - models[0].df_model)
    print(f'Estimated Chi2 statistic is: {chi2_stat} \n'
          f'P-value is: {p_val} \n'
          )

def rao(*models):
    scores = [i.resid_response / i.model.family.variance(i.predict()) for i in models]
    raos = [i.T  @ np.diag(j.model.family.variance(j.predict())) @ i for i, j in zip(scores, models)]
    
    chi2_stat = (raos[0] - raos[-1]) / models[0].scale
    p_val = scipy.stats.chi2.sf(chi2_stat, df=models[-1].df_model - models[0].df_model)
    print(f'Estimated Rao statistic is: {chi2_stat} \n'
          f'P-value is: {p_val} \n'
          )

def cp(*models):
    cps = [i.resid_deviance for i in models]
    pass


# FOLLOWING CODE WAS SHAMELESLY COPY PASTED FROM https://robert-alvarez.github.io/2018-06-04-diagnostic_plots/

# ANOTHER PLOTTING CAPABILITIES
# https://www.statsmodels.org/devel/examples/notebooks/generated/categorical_interaction_plot.html
# https://www.statsmodels.org/devel/examples/notebooks/generated/lowess.html
# https://www.statsmodels.org/devel/examples/notebooks/generated/glm.html

# TAKEN FROM # https://www.statsmodels.org/devel/examples/notebooks/generated/linear_regression_diagnostics_plots.html
#  AND ADJUSTED TO WORK WITH  GLMs


style_talk = 'seaborn-talk'    #refer to plt.style.available

class DiagnosticPlots():
    """
    Diagnostic plots to identify potential problems in a linear regression fit.
    Mainly,
        a. non-linearity of data
        b. Correlation of error terms
        c. non-constant variance
        d. outliers
        e. high-leverage points
        f. collinearity

    Author:
        Prajwal Kafle (p33ajkafle@gmail.com, where 3 = r)
        Does not come with any sort of warranty.
        Please test the code one your end before using.
    """

    def __init__(self,
                 results: Type[statsmodels.regression.linear_model.RegressionResultsWrapper]) -> None:
        """
        For a linear regression model / GLM, generates following diagnostic plots:

        a. residual
        b. qq
        c. scale location and
        d. leverage

        and a table

        e. vif

        Args:
            results (Type[statsmodels.regression.linear_model.RegressionResultsWrapper]):
                must be instance of statsmodels.regression.linear_model object

        Raises:
            TypeError: if instance does not belong to above object

        Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import statsmodels.formula.api as smf
        >>> x = np.linspace(-np.pi, np.pi, 100)
        >>> y = 3*x + 8 + np.random.normal(0,1, 100)
        >>> df = pd.DataFrame({'x':x, 'y':y})
        >>> res = smf.ols(formula= "y ~ x", data=df).fit()
        >>> cls = DiagnosticPlots(res)
        >>> cls(plot_context="seaborn-paper")

        In case you do not need all plots you can also independently make an individual plot/table
        in following ways

        >>> cls = DiagnosticPlots(res)
        >>> cls.residual_plot()
        >>> cls.qq_plot()
        >>> cls.scale_location_plot()
        >>> cls.leverage_plot()
        >>> cls.vif_table()
        """

        if not isinstance(results, statsmodels.regression.linear_model.RegressionResultsWrapper) or \
           not isinstance(results, statsmodels.genmod.generalized_linear_model.GLMResultsWrapper):
            raise TypeError("result must be instance of statsmodels.regression.linear_model.RegressionResultsWrapper  or"
                            " statsmodels.genmod.generalized_linear_model.GLMResultsWrapper object")

        self.results = maybe_unwrap_results(results)

        self.y_true = self.results.model.endog
        self.y_predict = self.results.fittedvalues
        self.xvar = self.results.model.exog
        self.xvar_names = self.results.model.exog_names

        influence = self.results.get_influence()
        self.leverage = influence.hat_matrix_diag
        self.cooks_distance = influence.cooks_distance[0]
        self.nparams = len(self.results.params)

        if isinstance(self.results, statsmodels.genmod.generalized_linear_model.GLMResults):

            self.residual = np.array(self.results.resid_pearson)  # Not sure if pearson is best choise | we can use anscombe or other instead
            self.residual_norm = influence.resid_studentized

        else:
            self.residual = np.array(self.results.resid)
            self.residual_norm = influence.resid_studentized_internal
            


    def __call__(self, plot_context='seaborn-paper'):
        # print(plt.style.available)
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
            self.residual_plot(ax=ax[0,0])
            self.qq_plot(ax=ax[0,1])
            self.scale_location_plot(ax=ax[1,0])
            self.leverage_plot(ax=ax[1,1])
            plt.show()

        self.vif_table()
        return fig, ax


    def residual_plot(self, ax=None):
        """
        Residual vs Fitted Plot

        Graphical tool to identify non-linearity.
        (Roughly) Horizontal red line is an indicator that the residual has a linear pattern
        """
        if ax is None:
            fig, ax = plt.subplots()

        sns.residplot(
            x=self.y_predict,
            y=self.residual,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.sort(residual_abs))
        abs_resid_top_3 = abs_resid[:3]
        for i, _ in enumerate(abs_resid_top_3):
            ax.annotate(
                i,
                xy=(self.y_predict[i], self.residual[i]),
                color='C3')

        ax.set_title('Residuals vs Fitted', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        return ax

    def qq_plot(self, ax=None):
        """
        Standarized Residual vs Theoretical Quantile plot

        Used to visually check if residuals are normally distributed.
        Points spread along the diagonal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        QQ = ProbPlot(self.residual_norm)
        QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(self.residual_norm)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for r, i in enumerate(abs_norm_resid_top_3):
            ax.annotate(
                i,
                xy=(np.flip(QQ.theoretical_quantiles, 0)[r], self.residual_norm[i]),
                ha='right', color='C3')

        ax.set_title('Normal Q-Q', fontweight="bold")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax

    def scale_location_plot(self, ax=None):
        """
        Sqrt(Standarized Residual) vs Fitted values plot

        Used to check homoscedasticity of the residuals.
        Horizontal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        residual_norm_abs_sqrt = np.sqrt(np.abs(self.residual_norm))

        ax.scatter(self.y_predict, residual_norm_abs_sqrt, alpha=0.5);
        sns.regplot(
            x=self.y_predict,
            y=residual_norm_abs_sqrt,
            scatter=False, ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            ax.annotate(
                i,
                xy=(self.y_predict[i], residual_norm_abs_sqrt[i]),
                color='C3')
        ax.set_title('Scale-Location', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$');
        return ax

    def leverage_plot(self, ax=None):
        """
        Residual vs Leverage plot

        Points falling outside Cook's distance curves are considered observation that can sway the fit
        aka are influential.
        Good to have none outside the curves.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            self.leverage,
            self.residual_norm,
            alpha=0.5);

        sns.regplot(
            x=self.leverage,
            y=self.residual_norm,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(
                i,
                xy=(self.leverage[i], self.residual_norm[i]),
                color = 'C3')

        xtemp, ytemp = self.__cooks_dist_line(0.5) # 0.5 line
        ax.plot(xtemp, ytemp, label="Cook's distance", lw=1, ls='--', color='red')
        xtemp, ytemp = self.__cooks_dist_line(1) # 1 line
        ax.plot(xtemp, ytemp, lw=1, ls='--', color='red')

        ax.set_xlim(0, max(self.leverage)+0.01)
        ax.set_title('Residuals vs Leverage', fontweight="bold")
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        ax.legend(loc='upper right')
        return ax

    def vif_table(self):
        """
        VIF table

        VIF, the variance inflation factor, is a measure of multicollinearity.
        VIF > 5 for a variable indicates that it is highly collinear with the
        other input variables.
        """
        vif_df = pd.DataFrame()
        vif_df["Features"] = self.xvar_names
        vif_df["VIF Factor"] = [variance_inflation_factor(self.xvar, i) for i in range(self.xvar.shape[1])]

        print(vif_df
                .sort_values("VIF Factor")
                .round(2))


    def __cooks_dist_line(self, factor):
        """
        Helper function for plotting Cook's distance curves
        """
        p = self.nparams
        formula = lambda x: np.sqrt((factor * p * (1 - x)) / x)
        x = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x)
        return x, y

