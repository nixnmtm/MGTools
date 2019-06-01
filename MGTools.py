import pandas as pd
import numpy as np
import MDAnalysis as mda
from Bio.PDB import PDBList


class MGTools(object):

    def __init__(self, table, ressep):

        self.grouping = ["segidI", "resI", "segidJ", "resJ"]
        self._index = ["segidI", "resI", "I", "segidJ", "resJ", "J"]
        self.table = table
        self.ressep = ressep
        self.pdbno = np.array(['16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26',
                               '27', '28', '29', '30', '31', '32', '33', '34', '37', '38', '39',
                               '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                               '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61',
                               '62', '63', '64', '66', '67', '68', '69', '70', '71', '72', '73',
                               '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84',
                               '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95',
                               '96', '97', '98', '99', '100', '101', '102', '103', '104', '105',
                               '106', '107', '108', '109', '110', '111', '112', '113', '114',
                               '115', '116', '117', '118', '119', '120', '121', '122', '123',
                               '124', '125', '127', '128', '129', '130', '132', '133', '134',
                               '135', '136', '137', '138', '139', '140', '141', '142', '143',
                               '144', '145', '146', '147', '148', '149', '150', '151', '152',
                               '153', '154', '155', '156', '157', '158', '159', '160', '161',
                               '162', '163', '164', '165', '166', '167', '168', '169', '170',
                               '171', '172', '173', '174', '175', '176', '177', '178', '179',
                               '180', '181', '182', '183', '184', '184A', '185', '186', '187',
                               '188', '188A', '189', '190', '191', '192', '193', '194', '195',
                               '196', '197', '198', '199', '200', '201', '202', '203', '204',
                               '209', '210', '211', '212', '213', '214', '215', '216', '217',
                               '219', '220', '221', '221A', '222', '223', '224', '225', '226',
                               '227', '228', '229', '230', '231', '232', '233', '234', '235',
                               '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', 'CA'])

        self.pdbaa = np.array(['I16', 'V17', 'G18', 'G19', 'Y20', 'T21', 'C22', 'Q23', 'E24',
                               'N25', 'S26', 'V27', 'P28', 'Y29', 'Q30', 'V31', 'S32', 'L33',
                               'N34', 'S37', 'G38', 'Y39', 'H40', 'F41', 'C42', 'G43', 'G44',
                               'S45', 'L46', 'I47', 'N48', 'D49', 'Q50', 'W51', 'V52', 'V53',
                               'S54', 'A55', 'A56', 'H57', 'C58', 'Y59', 'K60', 'S61', 'R62',
                               'I63', 'Q64', 'V66', 'R67', 'L68', 'G69', 'E70', 'H71', 'N72',
                               'I73', 'N74', 'V75', 'L76', 'E77', 'G78', 'N79', 'E80', 'Q81',
                               'F82', 'V83', 'N84', 'A85', 'A86', 'K87', 'I88', 'I89', 'K90',
                               'H91', 'P92', 'N93', 'F94', 'D95', 'R96', 'K97', 'T98', 'L99',
                               'N100', 'N101', 'D102', 'I103', 'M104', 'L105', 'I106', 'K107',
                               'L108', 'S109', 'S110', 'P111', 'V112', 'K113', 'L114', 'N115',
                               'A116', 'R117', 'V118', 'A119', 'T120', 'V121', 'A122', 'L123',
                               'P124', 'S125', 'S127', 'C128', 'A129', 'P130', 'A132', 'G133',
                               'T134', 'Q135', 'C136', 'L137', 'I138', 'S139', 'G140', 'W141',
                               'G142', 'N143', 'T144', 'L145', 'S146', 'S147', 'G148', 'V149',
                               'N150', 'E151', 'P152', 'D153', 'L154', 'L155', 'Q156', 'C157',
                               'L158', 'D159', 'A160', 'P161', 'L162', 'L163', 'P164', 'Q165',
                               'A166', 'D167', 'C168', 'E169', 'A170', 'S171', 'Y172', 'P173',
                               'G174', 'K175', 'I176', 'T177', 'D178', 'N179', 'M180', 'V181',
                               'C182', 'V183', 'G184', 'F184A', 'L185', 'E186', 'G187', 'G188', 'K188A',
                               'D189', 'S190', 'C191', 'Q192', 'G193', 'D194', 'S195', 'G196',
                               'G197', 'P198', 'V199', 'V200', 'C201', 'N202', 'G203', 'E204',
                               'L209', 'Q210', 'G211', 'I212', 'V213', 'S214', 'W215', 'G216',
                               'Y217', 'G219', 'C220', 'A221', 'B221A', 'P222', 'D223', 'N224',
                               'P225', 'G226', 'V227', 'Y228', 'T229', 'K230', 'V231', 'C232',
                               'N233', 'Y234', 'V235', 'D236', 'W237', 'I238', 'Q239', 'D240',
                               'T241', 'I242', 'A243', 'A244', 'N245', 'CA'])

    def sepres(self):
        tmp = self.table[self.table["segidI"] == self.table["segidJ"]]
        tmp = tmp[
            (tmp["resI"] >= tmp["resJ"] + self.ressep) |
            (tmp["resJ"] >= tmp["resI"] + self.ressep)
            ]
        diff = self.table[self.table["segidI"] != self.table["segidJ"]]
        df = pd.concat([tmp, diff], axis=0)
        # df.set_index(self._index, inplace=True)
        return df

    def sum_mean(self, segid=None):
        tab_sep = self.sepres()
        if segid:
            tab_sep = tab_sep[(tab_sep["segidI"] == segid) & (tab_sep["segidJ"] == segid)]
        tab_sum = tab_sep.groupby(self.grouping).sum()
        tab_mean = tab_sum.mean(axis=1)
        tab_var = tab_sum.var(axis=1)
        tab_std = tab_sum.std(axis=1)
        return tab_sum, tab_mean, tab_std, tab_var

    @staticmethod
    def csm_mat(tab):
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
           >>> eigenVectors, eigenValues = eigenVect(M)

        """
        eigenValues, eigenVectors = np.linalg.eigh(M)
        idx = (-eigenValues).argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        for k in range(eigenVectors.shape[1]):
            if np.sign(np.mean(eigenVectors[:, k])) != 0:
                eigenVectors[:, k] = np.sign(np.mean(eigenVectors[:, k])) * eigenVectors[:, k]
        return eigenValues, eigenVectors

    def t_eigenVect(self, sum_mat, Nwind, start, end, Npos, SS=False):
        """ Return csm_mat, eigenVectors and eigenValues of all windows """
        t_mat = np.zeros((Nwind, Npos, Npos))
        t_vec = np.zeros((Nwind, Npos, Npos))
        t_val = np.zeros((Nwind, Npos))
        for i in range(Nwind):
            time_mat = self.csm_mat(sum_mat.iloc[:, i])
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
        mleak = []
        mpers = []
        for m in range(Npos):
            # print("mode %d" %m)
            ps = []
            lk = []
            for w in range(Nwind):
                max1 = sorted(np.square(pers_mat[w][m]))[-1]
                pers = max1
                leak = 1 - max1
                # if sorted(np.square(pers_mat[w][m]))[-2] > 0.4:
                #    max2 = sorted(np.square(pers_mat[w][m]))[-2]
                #    pers = max1 + max2
                #   leak = 1-(max1 + max2)
                lk.append(leak)
                ps.append(pers)
            mleak.append(np.asarray(lk).mean())
            mpers.append(np.asarray(ps).mean())
        return mpers, mleak

    def split_sec_struc(self, psf, prefix, write=False):
        # split table into three tables based on BB,BS and SS
        cg = mda.Universe(psf)
        atomnames = cg.atoms.names()
        atomids = np.unique(self.table["I"])
        atmname_atmid = dict(list(zip(atomids, atomnames)))
        tmp = self.table.copy(deep=True)
        tmp['I'].replace(atmname_atmid, inplace=True)
        tmp['J'].replace(atmname_atmid, inplace=True)
        # BACKBONE-BACKBONE
        B_B = tmp[((tmp["I"] == 'N') | (tmp["I"] == 'O') | (tmp["I"] == 'ions')) \
                  & ((tmp["J"] == 'N') | (tmp["J"] == 'O') | (tmp["J"] == 'ions'))]

        # BACKBONE-SIDECHAIN
        B_S = tmp[((tmp["I"] == "N") | (tmp["I"] == 'O') | (tmp["I"] == "ions")) & (tmp["J"] == 'CB')]
        S_B = tmp[(tmp["I"] == 'CB') & ((tmp["J"] == "N") | (tmp["J"] == 'O') | (tmp["J"] == "ions"))]
        BB_side = pd.concat([B_S, S_B], axis=0, ignore_index=True)

        # SIDECHAIN-SIDECHAIN
        S_S = tmp[(tmp["I"] == "CB") & (tmp["J"] == "CB")]

        if write:
            # write the files in current directory
            B_B.to_csv(prefix + "_" + "kb_BB.txt", header=True, sep=" ")
            BB_side.to_csv(prefix + "_" + "kb_BS.txt", header=True, sep=" ")
            S_S.to_csv(prefix + "_" + "kb_SS.txt", header=True, sep=" ")

        return B_B, BB_side, S_S

    @staticmethod
    def significant_residues(eigenVector, modes, cutoff):
        # Get sigificant residues from significant modes
        r = list()
        for j in modes:
            # for i in np.where(eigenVector[:,j-1] >= cutoff): #signifcnt residues only from +ive residues, not used anymore
            for i in np.where(eigenVector[:, j - 1] * eigenVector[:, j - 1] >= cutoff):
                r.append(i + 1)
        top_res = np.sort(np.unique(np.concatenate(r, axis=0)))
        top_res = top_res[top_res < 224]
        # print "Residues from significant modes: \n {} and size is {}".format(top_res, top_res.size)
        return top_res

    @staticmethod
    def hitcov(sca_res, cs_res):
        # hitrate and coverage rate with pySCA
        # input pySCA residues and significant csm residues as an array
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

    def pos_neg_res(self, mode, idx, U, pdb=False, wcut=0.1):
        ''' modes: give exactly which eigenmode needed
            idx : is normal numbering
            pdb: if needed in pdb numbering
            U : the eigenvector matrix to analyze
            returns:  residues coupled positively and negatively in the given eigenmodes even after taking the norm'''
        pres = list()
        nres = list()
        sres = list()
        if pdb:
            for m, n, s in zip(self.pdbno, U[:, mode - 1], (U[:, mode - 1] * U[:, mode - 1])):
                if s > wcut:
                    sres.append(m)
                if n > 0.0:
                    pres.append(m)
                if n < 0.0:
                    nres.append(m)
            return sres, np.sort(np.intersect1d(pres, sres)), np.sort(np.intersect1d(nres, sres))
        else:
            for m, n, s in zip(idx, U[:, mode - 1], (U[:, mode - 1] * U[:, mode - 1])):
                if s > wcut:
                    sres.append(m + 1)
                if n > 0.0:
                    pres.append(m + 1)
                if n < 0.0:
                    nres.append(m + 1)
            return sres, np.sort(np.intersect1d(pres, sres)), np.sort(np.intersect1d(nres, sres))

    def map_aa(self, aa2map, aa=False, aanum=True, contno=False):
        aa_need = []
        if aanum is True:
            aaDict = dict(zip(range(1, 225), self.pdbno))
            for aa in aa2map:
                aa_need.append(aaDict[aa])
        if aa is True:
            aaDict = dict(zip(range(1, 225), self.pdbaa))
            for aa in aa2map:
                aa_need.append(aaDict[aa])
        if contno is True:
            aaDict = dict(zip(self.pdbno, range(1, 225)))
            for aa in aa2map:
                aa_need.append(aaDict[aa])
        return aa_need

