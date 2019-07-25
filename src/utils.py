from __future__ import print_function
import numpy as np
import logging
from core import BuildMG
import matplotlib.pyplot as plt
import itertools

logging.basicConfig(level=logging.INFO)


def t_eigenVect(stab, SS=False):
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
        tval, tvec = eigenVect(time_mat)
        t_val[i, :] = tval
        t_vec[i, :, :] = tvec
        t_mat[i, :, :] = time_mat
    return t_mat, t_vec, t_val


def significant_residues(eigenVector, pers_modes, cutoff):
    """
    Get significant residues from persistant modes

    :param eigenVector: numpy eigenvector matrix
    :param pers_modes: index of significant modes to check for important residues
    :param cutoff: weight cutoff
    :return: significant residues
    :rtype: list

    """
    # Get sigificant residues from significant modes
    r = list()
    for j in pers_modes:
        for i in np.where(eigenVector[:, j - 1] * eigenVector[:, j - 1] >= cutoff):
            r.append(i + 1)
    top_res = np.sort(np.unique(np.concatenate(r, axis=0)))
    # print "Residues from significant modes: \n {} and size is {}".format(top_res, top_res.size)
    return top_res


def hitcov(sca_res, cs_res):
    """
    Calculate Hitrate and Coverage of MGT residues compared with pySCA

    :param sca_res: pySCA residues
    :type sca_res: numpy array
    :param cs_res: significant residues from MGT
    :type cs_res: numpy array
    :return: hitrate and coverage

    """
    common = np.intersect1d(sca_res, cs_res).size
    hit = np.float(common) / np.float(cs_res.size)
    cov = np.float(common) / np.float(sca_res.size)
    return hit, cov


def annotate_modes(eigen_vec, modes, ndx, aa):
    """
    Annotate the modes in eigenvector

    :param eigen_vec: eigenvector numpy matrix
    :param modes: sindices of modes
    :param ndx: continuous indices (ex: range(1,224))
    :param aa: 3 letter aminoacid array

    """
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


def get_kbHmodes(mean_table, eig_vec, kBcut=5, resnoIndexAdd=None, Npos=None):

    """
    Get eigenmodes having inter-residue coupling strength greater than 5 kcal/mol/A^2

    :param mean_table: the mean table used for creating MGT
    :param eig_vec: Eigen vectors
    :param kBcut: coupling strength(kB) cutoff to select interactions greater than it.
    :param resnoIndexAdd: Number to be added with index to match the residue numbering.
                           (ex): For PDZ3, 0+301, as the index starts frßom 0 and resno starts from 301

    :return: Dictionary of modes and it details and List of Modes having inter-residue kB greter than kBCut

    """

    mode_details = list()
    ddd = mean_table.copy(deep=True)  # input M
    ddd = ddd.reset_index()
    ddd = ddd.drop(columns=["segidI", "segidJ"], axis=1)
    ddd = ddd.set_index(["resI", "resJ"])
    for i in range(Npos):
        elem_ndx = np.asarray(
            np.where(eig_vec[:, i] * eig_vec[:, i] > 0.05))  # index of eigvec elements with weight gt 0.05
        elem_ndx = elem_ndx[0]
        # print("Mode {}".format(i+1))
        for j in itertools.combinations(elem_ndx, 2):  # iterate throught combinations of the elements obtained
            # print(j[0]+301,j[1]+301)
            mode_res = dict()
            j = np.asarray(j)
            residue_I = j[0] + resnoIndexAdd
            residue_J = j[1] + resnoIndexAdd
            if residue_I in ddd.index.get_level_values("resI"):
                rI = ddd.loc[j[0] + resnoIndexAdd]  # resI selected
                if residue_J in rI.index.get_level_values("resJ"):  # if resJ in selected resI
                    if (rI.loc[residue_J].values > kBcut):  # if the kB of pair resI and resJ gt 5 kcal/mol/A2
                        # print("residue:{},{}: {}\n".format(j[0]+301, j[1]+301, rI.loc[j[1]+301].values[0]))
                        # if i+1 not in mmm:
                        mode_res["mode"] = i + 1
                        mode_res["I"] = residue_I
                        mode_res["J"] = residue_J
                        mode_res["kb"] = rI.loc[residue_J].values[0]
                        mode_res["weightI"] = (eig_vec[:, i] * eig_vec[:, i])[residue_I - resnoIndexAdd]
                        mode_res["weightJ"] = (eig_vec[:, i] * eig_vec[:, i])[residue_J - resnoIndexAdd]
                        if not mode_res in mode_details:
                            mode_details.append(mode_res)
    modes = [d["mode"] for d in mode_details if "mode" in d]
    return mode_details, modes


def calc_pers(dot_mat, Npos, Nwind):
    """
    Calculate Persistence of eigenmodes

    :param dot_mat: matrix with dot product values of each mode.
    .. math::
        \mathbf{M_{ij} = {U^r_i \cdot U^w_j}}
    :param Npos: number of positions(ie. the total number of residues positions)
    :param Nwind: number of windows (ie. total number of time windows)
    :return: persistence of each mode
    :rtype: list
    """
    # calculate persistance and leakage
    mpers = []
    for m in range(Npos):
        # print("mode %d" %m)
        ps = []
        for w in range(Nwind):
            max1 = sorted(np.square(dot_mat[w][m]))[-1]
            pers = max1
            ps.append(pers)
        mpers.append(np.asarray(ps).mean())
    return mpers


def eigenVect(kmat):
    """
    Return the eigenvectors and eigenvalues, ordered by decreasing values of the
    eigenvalues, for a real symmetric matrix M. The sign of the eigenvectors is fixed
    so that the mean of its components is non-negative.

    :param kmat: symmetric matrix to perform eigenvalue decomposition

    :return: eigenvalues and eigenvectors

    :Example:

    eigenVectors, eigenValues = eigenVect(kmat)

    """
    eigenValues, eigenVectors = np.linalg.eigh(kmat)
    idx = (-eigenValues).argsort()
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    for k in range(eigenVectors.shape[1]):
        if np.sign(np.mean(eigenVectors[:, k])) != 0:
            eigenVectors[:, k] = np.sign(np.mean(eigenVectors[:, k])) * eigenVectors[:, k]
    return eigenValues, eigenVectors


#########################################################################################
#########################################################################################
import scipy.stats
import warnings
import random
import math
from joblib import Parallel, delayed

# just for surpressing warnings
warnings.simplefilter('ignore')


class CheckDistribution(object):
    """
        - description       :Checks a sample against 80 distributions by applying the Kolmogorov-Smirnov test.
        - use in MGT        :Checks the distribution of the persistence values
        - author            :Andre Dietrich, modified by Nixon
        - gitlab            : https://gitlab.com/OvGU-ESS/distribution-check
        - python_version    : 2.* and 3.*

    :Example:
    .. highlight:: python
    .. code-block:: python

        >>> from utils import CheckDistribution
        >>> chk_dist = CheckDistribution()
        >>> best = chk_dist.run()

    """

    def __init__(self, iteration=1, exclude=10.0, verbose=False, top=10, processes=-1):

        """
        This is :func:`__init__` docstring

        :param iteration: iterations (default=1)
        :param exclude:
        :param verbose: verbosity (default=False)
        :param top: rank and give top results (default=10)
        :param processes: number o processes (default=-1) meaning use all
        """


        # list of all available distributions

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

    def check(self, data, fct, verbose=False):
        """

        :param data: data to check
        :param fct: distribution to test
        :param verbose: verbosity
        :return: tuple of (distribution name, probability, D)
        """
        # fit our data set against every probability distribution
        parameters = eval("scipy.stats." + fct + ".fit(data)");
        # Applying the Kolmogorov-Smirnof two sided test
        D, p = scipy.stats.kstest(data, fct, args=parameters);

        if math.isnan(p): p = 0
        if math.isnan(D): D = 0

        if verbose:
            print(fct.ljust(16) + "p: " + str(p).ljust(25) + "D: " + str(D))

        return (fct, p, D)

    ########################################################################################

    def plot(self, fcts, data, pd_cut=0.95):
        """
        :param fcts: distribution to plot
        :param data: data to check
        :param pd_cut: cumulative density function cutoff

        :return plots image and returns pcut values
        """
        # plot data
        plt.hist(data, normed=True, bins=max(10, int(len(data) / 10)), color='k')

        # plot fitted probability
        for i in range(len(fcts)):
            fct = fcts[i][0]
            params = eval("scipy.stats." + fct + ".fit(data)")
            f = eval("scipy.stats." + fct + ".freeze" + str(params))
            x = np.linspace(f.ppf(0.001), f.ppf(0.999), len(data))
            plt.plot(x, f.pdf(x), lw=3, label=fct)
            cd = f.cdf(x)
            tmp = f.pdf(x).argmax()
            if abs(max(data)) > abs(min(data)):
                tail = cd[tmp:len(cd)]
            else:
                cd = 1 - cd
                tail = cd[0:tmp]
            diff = abs(tail - pd_cut)
            x_pos = diff.argmin()
            p_cut = np.round(x[x_pos + tmp], 2)
            self.pcuts[fct] = p_cut
            plt.axvline(p_cut, color='r', linestyle='--')
        plt.legend(loc='best', frameon=False)
        plt.title("Top " + str(len(fcts)) + " Results")
        plt.show()
        return self.pcuts

    def run(self, data, sort_it=True):
        """
        :param data: data to check distribution
        :param sort_it: whether to sort the results
        :return: sort_it: True: sorted list of tuples with (distribution name, probability, D)
                          False: dictionary with distribution functions {"distribution name": {"p":float, "D":float}}
        """
        for i in range(self.iteration):
            if self.iteration == 1:
                data = data
            else:
                data = [value for value in data if random.random() >= self.exclude / 100]

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