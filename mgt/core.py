import pandas as pd
import logging
import numpy as np
from mgt.base import BaseMG, LoadKbTable

logging.basicConfig(level=logging.INFO)


class MGCore(BaseMG):
    """
    Core Class for The Molecular Graph Theory Analysis

    """
    def __init__(self, table, **kwargs):
        super(MGCore, self).__init__(table, **kwargs)
        self.segid = kwargs.get("segid")
        self.sskey = kwargs.get("sskey")
        self.res_idx, self.nres = self.get_nres_seg()

    def get_sum_table(self):
        """
        Get table sum of specific secondary structure and segid

        :param segid: segid
        :param sskey: secondary structure key
        :return: Dataframe of sum table
        """
        return self.table_sum()[self.segid][self.sskey]

    def get_mean_table(self):
        """
        Get mean table of specific secondary structure and segid

        :param segid: segid
        :param sskey: secondary structure key
        :return: Dataframe of mean table
        """
        return self.table_mean()[self.segid][self.sskey]

    def table_intraseg(self):
        tmp = self.table.copy(deep=True)
        if isinstance(self.table.index, pd.core.index.MultiIndex):
            mask = (tmp.index.get_level_values("segidI") == self.segid) & \
                   (tmp.index.get_level_values("segidJ") == self.segid)
            return tmp[mask].reset_index()
        else:
            mask = (tmp["segidI"] == self.segid) & (tmp["segidJ"] == self.segid)
            return tmp[mask]

    def get_nres_seg(self):
        """
        Ger number of residues in a segment
        :return:
        """

        tmp = self.table_sum()[self.segid]["BB"]  # BB will have all residue interactions, so resnum will be intact
        res_idx = tmp.index.get_level_values("resI").unique().tolist()
        nres = len(res_idx)
        return res_idx, nres

    def molg_mat(self, tab=None):
        """
        Build MGT Matrix.
        | Input should be a :class:`pd.Series`

        :param tab: Mean series or window series
        :return:  MGT matrix, type dataframe

        :Example:

        >>> load = LoadKbTable(filename="holo_pdz.txt.bz2")
        >>> kb_aa = load.load_table()
        >>> core = MGCore(kb_aa, segid="CRPT", sskey="BB", ressep=1)
        >>> print(core.molg_mat())
                    5           6           7           8           9
        5  172.692126  169.063123    3.482413    0.139217    0.007373
        6  169.063123  364.543558  193.112981    2.314533    0.052921
        7    3.482413  193.112981  390.274191  192.792781    0.886016
        8    0.139217    2.314533  192.792781  390.518684  195.272153
        9    0.007373    0.052921    0.886016  195.272153  196.218462

        """
        if tab is None:
            tab = self.get_mean_table()
        if isinstance(tab, pd.Series) and isinstance(tab.index, pd.core.index.MultiIndex):
            tab = tab.reset_index()
        if tab.groupby("resI").sum().shape[0] > 1:
            diag_val = tab.groupby("resI").sum().drop("resJ", axis=1).values.ravel()
        else:
            diag_val = tab.groupby("resI").sum().values.ravel()
        ref_mat = tab.drop(["segidI", "segidJ"], axis=1).set_index(['resI', 'resJ']).unstack(fill_value=0).values
        row, col = np.diag_indices(ref_mat.shape[0])
        ref_mat[row, col] = diag_val
        mat = pd.DataFrame(ref_mat, index=np.unique(tab.resI.values), columns=np.unique(tab.resI.values))
        npos = tab.resI.unique().size
        if npos != self.nres:
            mat = mat.reindex(self.res_idx).T.reindex(self.res_idx).replace(np.nan, 0.0)
        return mat

    def eigh_decom(self, kmat=None):
        """
        Return the eigenvectors and eigenvalues, ordered by decreasing values of the
        eigenvalues, for a real symmetric matrix M. The sign of the eigenvectors is fixed
        so that the mean of its components is non-negative.

        :param kmat: symmetric matrix to perform eigenvalue decomposition

        :return: eigenvalues and eigenvectors

        :Example:

        >>> load = LoadKbTable(filename="holo_pdz.txt.bz2")
        >>> kb_aa = load.load_table()
        >>> core = MGCore(kb_aa, segid="CRPT", sskey="BB", ressep=1)
        >>> egval, egvec = core.eigh_decom()
        >>> assert egval.shape[0] == egvec.shape[0]

        """
        if kmat is None:
            eigval, eigvec = np.linalg.eigh(self.molg_mat())
        else:
            eigval, eigvec = np.linalg.eigh(kmat)
        idx = (-eigval).argsort()
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        for k in range(eigvec.shape[1]):
            if np.sign(np.mean(eigvec[:, k])) != 0:
                eigvec[:, k] = np.sign(np.mean(eigvec[:, k])) * eigvec[:, k]
        return eigval, eigvec

    def windows_eigen_decom(self):
        """
        Return csm_mat, eigenVectors and eigenValues of all windows

        .. todo::
            1. Check for smarter ways to speed it up
            2. So far this is the best way
        """

        stab = self.get_sum_table()
        nwind = stab.columns.size
        npos = stab.index.get_level_values('resI').unique().size
        t_mat = np.zeros((nwind, self.nres, self.nres))
        t_vec = np.zeros((nwind, self.nres, self.nres))
        t_val = np.zeros((nwind, self.nres))

        for i in range(nwind):
            time_mat = self.molg_mat(tab=stab.iloc[:, i])
            if npos != self.nres:
                time_mat = time_mat.reindex(self.res_idx).T.reindex(self.res_idx).replace(np.nan, 0.0)
            tval, tvec = self.eigh_decom(time_mat)
            t_val[i, :] = tval
            t_vec[i, :, :] = tvec
            t_mat[i, :, :] = time_mat
        return t_mat, t_val, t_vec

    def evec_dotpdts(self):
        """
        Evaluate dot products between eigen vectors of decomposed mean mgt matrix and each window mgt matrix


        .. math::
            \\mathbf{M_{ij} = {U^r_i \\cdot U^w_j}}


        r - reference mean matrix eigen vectors

        w - windows eigen vectors

        """
        logging.info("Decomposition of each window MGT matrix started")
        logging.info("This may take some time based on the number of windows analyzed")
        tmat, tval, tvec = self.windows_eigen_decom()
        logging.info("Windows decomposition completed")
        meval, mevec = self.eigh_decom()
        logging.info("Decomposition of Mean MGT matrix done")
        dps = np.zeros(tvec.shape)
        for t in range(tvec.shape[0]):
            dps[t] = np.dot(mevec.T, tvec[t])
        logging.info("Dot products of eigenmodes done")
        return dps

    def calc_persistence(self, dot_mat=None):
        """
        Calculate Persistence of eigenmodes

        :param dot_mat: matrix with dot product values of each mode.
        :return: persistence of each mode
        :rtype: list
        """

        if dot_mat is None:
            dot_mat = self.evec_dotpdts()
        mpers = []
        logging.info("Calculating Persistence")
        for m in range(dot_mat.shape[1]):
            # sort and get max value (ie) last element
            ps = [sorted(np.square(dot_mat[w][m]))[-1] for w in range(dot_mat.shape[0])]
            mpers.append(np.asarray(ps).mean())
        return mpers

