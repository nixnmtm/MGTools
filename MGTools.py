import pandas as pd
import numpy as np
import MDAnalysis as mda
import logging
from ProTools import ProTools

class MGTools(object):

    def __init__(self, table, ressep):

        self.grouping = ["segidI", "resI", "segidJ", "resJ"]
        self._index = ["segidI", "resI", "I", "segidJ", "resJ", "J"]
        self.table = table
        self.ressep = ressep

    def split_sec_struc(self, pdbid=None, write=False, prefix=None):
        # split table into three tables based on BB,BS and SS
        tmp = self.table.copy(deep=True)
        if pdbid is not None:
            self.pt = ProTools(pdbid)
            u = self.pt.univ
            # if atom names are not defined in the table
            atomnames = u.atoms.names
            atomids = np.unique(self.table["I"])
            atmname_atmid = dict(list(zip(atomids, atomnames)))
            tmp['I'].replace(atmname_atmid, inplace=True)
            tmp['J'].replace(atmname_atmid, inplace=True)

        # BACKBONE-BACKBONE
        BB = tmp[((tmp["I"] == 'N') | (tmp["I"] == 'O') | (tmp["I"] == 'ions')) \
                  & ((tmp["J"] == 'N') | (tmp["J"] == 'O') | (tmp["J"] == 'ions'))]

        # BACKBONE-SIDECHAIN
        BS = tmp[((tmp["I"] == "N") | (tmp["I"] == 'O') | (tmp["I"] == "ions")) & (tmp["J"] == 'CB')]
        SB = tmp[(tmp["I"] == 'CB') & ((tmp["J"] == "N") | (tmp["J"] == 'O') | (tmp["J"] == "ions"))]
        BB_side = pd.concat([BS, SB], axis=0, ignore_index=True)

        # SIDECHAIN-SIDECHAIN
        SS = tmp[(tmp["I"] == "CB") & (tmp["J"] == "CB")]

        if write:
            if prefix is None:
                logging.error("prefix is not defined")
                exit(1)
            else:
                # write the files in current directory
                BB.to_csv(prefix + "_" + "kb_BB.txt", header=True, sep=" ")
                BB_side.to_csv(prefix + "_" + "kb_BS.txt", header=True, sep=" ")
                SS.to_csv(prefix + "_" + "kb_SS.txt", header=True, sep=" ")

        return BB, BB_side, SS

    def sepres(self):
        tmp = self.table[self.table["segidI"] == self.table["segidJ"]]
        tmp = tmp[
            (tmp["resI"] >= tmp["resJ"] + self.ressep) |
            (tmp["resJ"] >= tmp["resI"] + self.ressep)
            ]
        diff = self.table[self.table["segidI"] != self.table["segidJ"]]
        df = pd.concat([tmp, diff], axis=0)
        return df

    def sum_mean(self, segid=None):
        """Returns the sum, mean and standard deviation of residues based on the self.grouping"""
        tab_sep = self.sepres()
        if segid:
            tab_sep = tab_sep[(tab_sep["segidI"] == segid) & (tab_sep["segidJ"] == segid)]
        tab_sum = tab_sep.groupby(self.grouping).sum()
        tab_mean = tab_sum.mean(axis=1)
        tab_std = tab_sum.std(axis=1)
        return tab_sum, tab_mean, tab_std

    def csm_mat(self, tab):
        """Returns symmetric diagonally dominant residue-residue coupling strength matrix (CSM)"""

        try:
            tab.ndim == 1
        except TypeError:
            print('Dimension of the mat should not exceed 1, as we are stacking from each column')
        else:
            _tab = tab.reset_index()
            diag_val = _tab.groupby("resI").sum().drop("resJ", axis=1).values.ravel()
            ref_mat = _tab.drop(["segidI", "segidJ"], axis=1).set_index(['resI', 'resJ']).unstack(fill_value=0).values
            row, col = np.diag_indices(ref_mat.shape[0])
            ref_mat[row, col] = diag_val
            return pd.DataFrame(ref_mat, index=np.unique(_tab.reset_index().resI.values),
                                columns=np.unique(_tab.reset_index().resI.values))

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

    def t_eigenVect(self, Nwind, Npos, SS=False):
        """ Return csm_mat, eigenVectors and eigenValues of all windows """

        stab, _, _ = self.sum_mean()
        t_mat = np.zeros((Nwind, Npos, Npos))
        t_vec = np.zeros((Nwind, Npos, Npos))
        t_val = np.zeros((Nwind, Npos))
        for i in range(Nwind):
            time_mat = self.csm_mat(stab.iloc[:, i])
            start = self.pt.get_residNname(segid=self.pt.univ.segments[0].segid)[1][0]
            end = self.pt.get_residNname(segid=self.pt.univ.segments[0].segid)[1][-1]
            if SS:
                time_mat = time_mat.reindex(np.arange(start, end)).T.reindex(np.arange(start, end)).replace(np.nan, 0.0)
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

    def pos_neg_res(self, mode, U, pdb=False, wcut=0.1):
        ''' modes: give exactly which eigenmode needed
            idx : is normal numbering
            pdb: if needed in pdb numbering
            U : the eigenvector matrix to analyze
            returns:  residues coupled positively and negatively in the given eigenmodes even after taking the norm'''
        pres = list()
        nres = list()
        sres = list()

        if pdb:
            for m, n, s in zip(self.pt.get_residNname(self.pt.univ.segments[0].segid[1]),
                               U[:, mode - 1],
                               (U[:, mode - 1] * U[:, mode - 1])):
                if s > wcut:
                    sres.append(m)
                if n > 0.0:
                    pres.append(m)
                if n < 0.0:
                    nres.append(m)
            return sres, np.sort(np.intersect1d(pres, sres)), np.sort(np.intersect1d(nres, sres))
        else:
            for m, n, s in zip(range(self.pt.get_residNname(segid=self.pt.univ.segments[0].segid)[1][0],
                                     self.pt.get_residNname(segid=self.pt.univ.segments[0].segid)[1][-1]),
                               U[:, mode - 1],
                               (U[:, mode - 1] * U[:, mode - 1])):
                if s > wcut:
                    sres.append(m + 1)
                if n > 0.0:
                    pres.append(m + 1)
                if n < 0.0:
                    nres.append(m + 1)
            return sres, np.sort(np.intersect1d(pres, sres)), np.sort(np.intersect1d(nres, sres))

