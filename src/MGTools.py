from __future__ import print_function
import numpy as np
import logging
from src.core import BuildMG
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)


class MGTools(object):

    # def __init__(self, table):
    #     super().__init__(table)  # super class can be called by this or "ProTools.__init__(self, pdbid)"

    @staticmethod
    def eigenVect(M):
        """ Return the eigenvectors and eigenvalues, ordered by decreasing values of the
        eigenvalues, for a real symmetric matrix M. The sign of the eigenvectors is fixed
        so that the mean of its components is non-negative.

        :Example:
         eigenVectors, eigenValues = eigenVect(M)

        """
        eigenValues, eigenVectors = np.linalg.eigh(M)
        idx = (-eigenValues).argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        for k in range(eigenVectors.shape[1]):
            if np.sign(np.mean(eigenVectors[:, k])) != 0:
                eigenVectors[:, k] = np.sign(np.mean(eigenVectors[:, k])) * eigenVectors[:, k]
        return eigenValues, eigenVectors

    def t_eigenVect(self, stab, SS=False):
        """ Return csm_mat, eigenVectors and eigenValues of all windows """
        Nwind = stab.columns.size
        Npos = stab.index.get_level_values('resI').unique().size
        start = stab.index.get_level_values("resI")[0]  # start residue number
        end = stab.index.get_level_values("resI")[-1]  # end residue number
        if SS:
            Npos = end
        t_mat = np.zeros((Nwind, Npos, Npos))
        t_vec = np.zeros((Nwind, Npos, Npos))
        t_val = np.zeros((Nwind, Npos))
        for i in range(Nwind):
            time_mat = BuildMG.mgt_mat(stab.iloc[:, i])
            if SS:
                time_mat = time_mat.reindex(range(start, end+1)).T.reindex(range(start, end+1)).replace(np.nan, 0.0)
            tval, tvec = self.eigenVect(time_mat)
            t_val[i, :] = tval
            t_vec[i, :, :] = tvec
            t_mat[i, :, :] = time_mat
        return t_mat, t_vec, t_val

    @staticmethod
    def calc_pers(pers_mat, Npos, Nwind):
        # calculate persistance and leakage
        mpers = []
        for m in range(Npos):
            # print("mode %d" %m)
            ps = []
            for w in range(Nwind):
                max1 = sorted(np.square(pers_mat[w][m]))[-1]
                pers = max1
                ps.append(pers)
            mpers.append(np.asarray(ps).mean())
        return mpers

    @staticmethod
    def significant_residues(eigenVector, modes, cutoff):
        # Get sigificant residues from significant modes
        r = list()
        for j in modes:
            for i in np.where(eigenVector[:, j - 1] * eigenVector[:, j - 1] >= cutoff):
                r.append(i + 1)
        top_res = np.sort(np.unique(np.concatenate(r, axis=0)))
        top_res = top_res[top_res < 224]
        # print "Residues from significant modes: \n {} and size is {}".format(top_res, top_res.size)
        return top_res

    @staticmethod
    def hitcov(sca_res, cs_res):
        """hitrate rate with pySCA"""
        common = np.intersect1d(sca_res, cs_res).size
        hit = np.float(common) / np.float(cs_res.size)
        cov = np.float(common) / np.float(sca_res.size)
        return hit, cov

    @staticmethod
    def annotate_modes(eigen_vec, modes, ndx, aa):
        res = list()
        count = 0
        for i in np.array(modes):
            if count == 0:
                for m, l, n in zip(ndx, aa, (eigen_vec[:, i - 1] * eigen_vec[:, i - 1])):
                    res.append([l, n])
                count += 1
            else:
                for m, l, n in zip(ndx, aa, (eigen_vec[:, i - 1] * eigen_vec[:, i - 1])):
                    if res[m][1] < n:
                        res[m][1] = n
        return res

#########################################################################################
#########################################################################################
import scipy.stats
import warnings
import random
import math

# just for surpressing warnings
warnings.simplefilter('ignore')

from joblib import Parallel, delayed

class CheckDistribution(object):
    """
        # title           :distribution_checkX.py
        # description     :Checks a sample against 80 distributions by applying the Kolmogorov-Smirnov test.
        # author          :Andre Dietrich, modified by Nixon
        # version         :0.1
        # usage           :distribution_check.run()
        # original gitlab project link: https://gitlab.com/OvGU-ESS/distribution-check
        # python_version  :2.* and 3.*
    """
    def __init__(self, data, iteration=1, exclude=10.0, verbose=False, top=10, processes=-1):
        # list of all available distributions

        self.data = data
        self.iteration = iteration
        self.exclude = exclude
        self.verbose = verbose
        self.top = top
        self.processes = processes
        self.pcuts = dict()
        self.cdfs = {
            "alpha": {"p": [], "D": []},  # Alpha
            "anglit": {"p": [], "D": []},  # Anglit
            "arcsine": {"p": [], "D": []},  # Arcsine
            "beta": {"p": [], "D": []},  # Beta
            "betaprime": {"p": [], "D": []},  # Beta Prime
            "bradford": {"p": [], "D": []},  # Bradford
            "burr": {"p": [], "D": []},  # Burr
            "cauchy": {"p": [], "D": []},  # Cauchy
            "chi": {"p": [], "D": []},  # Chi
            "chi2": {"p": [], "D": []},  # Chi-squared
            "cosine": {"p": [], "D": []},  # Cosine
            "dgamma": {"p": [], "D": []},  # Double Gamma
            "dweibull": {"p": [], "D": []},  # Double Weibull
            "erlang": {"p": [], "D": []},  # Erlang
            "expon": {"p": [], "D": []},  # Exponential
            "exponweib": {"p": [], "D": []},  # Exponentiated Weibull
            "exponpow": {"p": [], "D": []},  # Exponential Power
            "f": {"p": [], "D": []},  # F (Snecdor F)
            "fatiguelife": {"p": [], "D": []},  # Fatigue Life (Birnbaum-Sanders)
            "fisk": {"p": [], "D": []},  # Fisk
            "foldcauchy": {"p": [], "D": []},  # Folded Cauchy
            "foldnorm": {"p": [], "D": []},  # Folded Normal
            "frechet_r": {"p": [], "D": []},  # Frechet Right Sided, Extreme Value Type II
            "frechet_l": {"p": [], "D": []},  # Frechet Left Sided, Weibull_max
            "gamma": {"p": [], "D": []},  # Gamma
            "gausshyper": {"p": [], "D": []},  # Gauss Hypergeometric
            "genexpon": {"p": [], "D": []},  # Generalized Exponential
            "genextreme": {"p": [], "D": []},  # Generalized Extreme Value
            "gengamma": {"p": [], "D": []},  # Generalized gamma
            "genhalflogistic": {"p": [], "D": []},  # Generalized Half Logistic
            "genlogistic": {"p": [], "D": []},  # Generalized Logistic
            "genpareto": {"p": [], "D": []},  # Generalized Pareto
            "gilbrat": {"p": [], "D": []},  # Gilbrat
            "gompertz": {"p": [], "D": []},  # Gompertz (Truncated Gumbel)
            "gumbel_l": {"p": [], "D": []},  # Left Sided Gumbel, etc.
            "gumbel_r": {"p": [], "D": []},  # Right Sided Gumbel
            "halfcauchy": {"p": [], "D": []},  # Half Cauchy
            "halflogistic": {"p": [], "D": []},  # Half Logistic
            "halfnorm": {"p": [], "D": []},  # Half Normal
            "hypsecant": {"p": [], "D": []},  # Hyperbolic Secant
            "invgamma": {"p": [], "D": []},  # Inverse Gamma
            "invgauss": {"p": [], "D": []},  # Inverse Normal
            "invweibull": {"p": [], "D": []},  # Inverse Weibull
            "johnsonsb": {"p": [], "D": []},  # Johnson SB
            "johnsonsu": {"p": [], "D": []},  # Johnson SU
            "laplace": {"p": [], "D": []},  # Laplace
            "logistic": {"p": [], "D": []},  # Logistic
            "loggamma": {"p": [], "D": []},  # Log-Gamma
            "loglaplace": {"p": [], "D": []},  # Log-Laplace (Log Double Exponential)
            "lognorm": {"p": [], "D": []},  # Log-Normal
            "lomax": {"p": [], "D": []},  # Lomax (Pareto of the second kind)
            "maxwell": {"p": [], "D": []},  # Maxwell
            "mielke": {"p": [], "D": []},  # Mielke's Beta-Kappa
            "nakagami": {"p": [], "D": []},  # Nakagami
            "ncx2": {"p": [], "D": []},  # Non-central chi-squared
            "ncf": {"p": [], "D": []},  # Non-central F
            "nct": {"p": [], "D": []},  # Non-central Student's T
            "norm": {"p": [], "D": []},  # Normal (Gaussian)
            "pareto": {"p": [], "D": []},  # Pareto
            "pearson3": {"p": [], "D": []},  # Pearson type III
            "powerlaw": {"p": [], "D": []},  # Power-function
            "powerlognorm": {"p": [], "D": []},  # Power log normal
            "powernorm": {"p": [], "D": []},  # Power normal
            "rdist": {"p": [], "D": []},  # R distribution
            "reciprocal": {"p": [], "D": []},  # Reciprocal
            "rayleigh": {"p": [], "D": []},  # Rayleigh
            "rice": {"p": [], "D": []},  # Rice
            "recipinvgauss": {"p": [], "D": []},  # Reciprocal Inverse Gaussian
            "semicircular": {"p": [], "D": []},  # Semicircular
            "t": {"p": [], "D": []},  # Student's T
            "triang": {"p": [], "D": []},  # Triangular
            "truncexpon": {"p": [], "D": []},  # Truncated Exponential
            "truncnorm": {"p": [], "D": []},  # Truncated Normal
            "tukeylambda": {"p": [], "D": []},  # Tukey-Lambda
            "uniform": {"p": [], "D": []},  # Uniform
            "vonmises": {"p": [], "D": []},  # Von-Mises (Circular)
            "wald": {"p": [], "D": []},  # Wald
            "weibull_min": {"p": [], "D": []},  # Minimum Weibull (see Frechet)
            "weibull_max": {"p": [], "D": []},  # Maximum Weibull (see Frechet)
            "wrapcauchy": {"p": [], "D": []},  # Wrapped Cauchy
            "ksone": {"p": [], "D": []},  # Kolmogorov-Smirnov one-sided (no stats)
            "kstwobign": {"p": [], "D": []}}  # Kolmogorov-Smirnov two-sided test for Large N

    ########################################################################################

    def check(self, fct, verbose=False):
        """

        :param fct: distribution to test
        :param verbose: verbosity
        :return: tuple of (distribution name, probability, D)
        """
        # fit our data set against every probability distribution
        parameters = eval("scipy.stats." + fct + ".fit(data)");
        # Applying the Kolmogorov-Smirnof two sided test
        D, p = scipy.stats.kstest(self.data, fct, args=parameters);

        if math.isnan(p): p = 0
        if math.isnan(D): D = 0

        if verbose:
            print(fct.ljust(16) + "p: " + str(p).ljust(25) + "D: " + str(D))

        return (fct, p, D)

    ########################################################################################

    def plot(self, fcts, pd_cut=0.95):
        """
        :param fcts: distribution to plot
        :param pd_cut: cumulative density function cutoff

        :return plots image and returns pcut values
        """
        # plot data
        plt.hist(self.data, normed=True, bins=max(10, int(len(self.data) / 10)))

        # plot fitted probability
        for i in range(len(fcts)):
            fct = fcts[i][0]
            params = eval("scipy.stats." + fct + ".fit(data)")
            f = eval("scipy.stats." + fct + ".freeze" + str(params))
            x = np.linspace(f.ppf(0.001), f.ppf(0.999), len(self.data))
            plt.plot(x, f.pdf(x), lw=3, label=fct)
            cd = f.cdf(x)
            tmp = f.pdf(x).argmax()
            if abs(max(self.data)) > abs(min(self.data)):
                tail = cd[tmp:len(cd)]
            else:
                cd = 1 - cd
                tail = cd[0:tmp]
            diff = abs(tail - pd_cut)
            x_pos = diff.argmin()
            p_cut = np.round(x[x_pos + tmp], 2)
            self.pcuts[fct] = p_cut
            plt.axvline(p_cut, color='k', linestyle='--')
        plt.legend(loc='best', frameon=False)
        plt.title("Top " + str(len(fcts)) + " Results")
        plt.show()
        return self.pcuts

    def run(self, sort_it=True):
        """
        :param sort_it: whether to sort the results
        :return: sort_it: True: sorted list of tuples with (distribution name, probability, D)
                          False: dictionary with distribution functions {"distribution name": {"p":float, "D":float}}
        """
        for i in range(self.iteration):
            if self.iteration == 1:
                data = self.data
            else:
                data = [value for value in self.data if random.random() >= self.exclude / 100]

            results = Parallel(n_jobs=self.processes)(
                delayed(self.check)(data, fct, self.verbose) for fct in self.cdfs.keys())

            for res in results:
                key, p, D = res
                self.cdfs[key]["p"].append(p)
                self.cdfs[key]["D"].append(D)
            if sort_it:
                print("-------------------------------------------------------------------")
                print("Top %d after %d iteration(s)" % (self.top, i + 1,))
                print("-------------------------------------------------------------------")
                best = sorted(self.cdfs.items(), key=lambda elem: scipy.median(elem[1]["p"]), reverse=True)
                for t in range(self.top):
                    fct, values = best[t]
                    print(str(t + 1).ljust(4), fct.ljust(16),
                          "\tp: ", scipy.median(values["p"]),
                          "\tD: ", scipy.median(values["D"]),
                          end="")
                    if len(values["p"]) > 1:
                        print("\tvar(p): ", scipy.var(values["p"]),
                              "\tvar(D): ", scipy.var(values["D"]), end="")
                    print()
            return best
        return self.cdfs