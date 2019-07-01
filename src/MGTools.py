import numpy as np
import logging
from src.core import BuildMG

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