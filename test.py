#from MGTools import MGTools
import pandas as pd
import numpy as np
import logging

# below two functions are written to avil withoutsplitMGT also, noe decided to make spilMGT as default

def table_sum(self):
    """
    Returns the sum table based on the self.grouping

    :return: splitMGT==True : dict of segids with  tables with keys ["BB", "BS", "SS"]
             splitMGT==False : dict of segids with sum of complete table

    """
    smtable = dict()
    if self.splitMGT:
        sstable = self.splitSS()
    for seg in self.table.segidI.unique():
        smtable[seg] = dict()
        if self.splitMGT:
            for key in self.splitkeys:
                tmp = self.sepres(table=sstable[key]).groupby(self.grouping).sum()
                mask = (tmp.index.get_level_values("segidI") == seg) & \
                       (tmp.index.get_level_values("segidJ") == seg)
                smtable[seg][key] = tmp[mask]
        else:
            tmp = self.sepres(table=self.table).groupby(self.grouping).sum()
            mask = (tmp.index.get_level_values("segidI") == seg) & \
                   (tmp.index.get_level_values("segidJ") == seg)
            smtable[seg] = tmp[mask]
    return smtable


def table_mean(self):
    """

    :return: mean table
    """

    table = self.table_sum()
    mntable = dict()
    for seg in table.keys():
        mntable[seg] = dict()
        if isinstance(table[seg], dict) and self.splitMGT:
            for key in table[seg].keys():
                mntable[seg][key] = table[seg][key].mean(axis=1)
        elif isinstance(table[seg], pd.DataFrame):
            mntable[seg] = table[seg].mean(axis=1)
        else:
            logging.warning("Unknown table format")
    return mntable

def mgt_mat(self):
    """
    Build MGT Matrix

    :return: MGT dataframe

    """

    tab = self.table_mean()
    mats = dict()
    for seg in tab.keys():
        mats[seg] = dict()
        for key in tab[seg].keys():
            if isinstance(tab[seg][key].index, pd.core.index.MultiIndex):
                tab[seg][key] = tab[seg][key].reset_index()
            diag_val = tab[seg][key].groupby("resI").sum().drop("resJ", axis=1).values.ravel()
            ref_mat = tab[seg][key].drop(["segidI", "segidJ"], axis=1).set_index(['resI', 'resJ']).unstack(fill_value=0).values
            row, col = np.diag_indices(ref_mat.shape[0])
            ref_mat[row, col] = diag_val
            mats[seg][key] = pd.DataFrame(ref_mat, index=np.unique(tab[seg][key].resI.values), columns=np.unique(tab[seg][key].resI.values))

    return mats


from src.core import MGCore

mgc = MGCore(filename="spholo_test_submat.txt")
print(mgc.ressep, mgc.input_path, mgc._segids())
