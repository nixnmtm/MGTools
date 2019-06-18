import pandas as pd
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


class BuildMG(object):
    """
    1. Base class for loading Coupling strength dataframe.
    """
    def __init__(self, filename: str, ressep=3, splitMgt=None, segid=None):
        """
        :param filename: Name of the file to be loaded
        :param ressep: residue separation( >= I,I + ressep), default is I,I+3
        :type int
        :param splitMgt: split mgt into BB or BS or SS
        :type str
        :param segid: segment id to analyze, if not mentioned all segid is used
        :type str
        """
        self.filename = filename
        self.dir_path = '../Inputs/'
        self.table = self.load_table()
        self.grouping = ["segidI", "resI", "segidJ", "resJ"]
        self._index = ["segidI", "resI", "I", "segidJ", "resJ", "J"]
        self.ressep = ressep
        self.splitMgt = splitMgt
        self.segid = segid

    def load_table(self, verbose: bool = False) -> pd.DataFrame:
        """
        Load Coupling Strength DataFrame
        :param verbose: verbosity
        :return: Structured coupling strength DataFrame

        """
        if verbose:
            logging.info("Loading file: {} from {}".format(self.filename, self.dir_path))
        _, fileext = os.path.splitext(self.filename)
        if not (fileext[-3:] == "txt") or (fileext[-3:] == "bz2"):
            logging.error("Please provide a appropriate file, with extension either txt or bz2")
        filepath = os.path.join(self.dir_path, self.filename)
        os.path.exists(filepath)
        table = pd.read_csv(os.path.join(filepath), sep=' ')
        if str(table.columns[0]).startswith("Unnamed"):
            table = table.drop(table.columns[0], axis=1)
        if verbose:
            logging.info("File loaded.")
        return table

    def splitSS(self, df: pd.DataFrame = None, write: bool = False, prefix: str = None) -> tuple:
        """
        Split based on secondary structures.
        BB - Backbone-Backbone Interactions
        BS - Backbone-Sidechain Interactions
        SS - Sidechain-Sidehain Interactions

        :param df: Dataframe to split. If None, df initialized during class instance is taken
        :param write: write after splitting
        :param prefix: prefix for writing file
        :return: tuple of split DataFrames

        """
        # split table into three tables based on BB,BS and SS
        if df is None:
            tmp = self.table.copy(deep=True)
        else:
            tmp = df.copy(deep=True)
        try:
            # BACKBONE-BACKBONE
            BB = tmp[((tmp["I"] == 'N') | (tmp["I"] == 'O') | (tmp["I"] == 'ions')) \
                      & ((tmp["J"] == 'N') | (tmp["J"] == 'O') | (tmp["J"] == 'ions'))]

            # BACKBONE-SIDECHAIN
            BS = tmp[((tmp["I"] == "N") | (tmp["I"] == 'O') | (tmp["I"] == "ions")) & (tmp["J"] == 'CB')]
            SB = tmp[(tmp["I"] == 'CB') & ((tmp["J"] == "N") | (tmp["J"] == 'O') | (tmp["J"] == "ions"))]
            BB_side = pd.concat([BS, SB], axis=0, ignore_index=True)

            # SIDECHAIN-SIDECHAIN
            SS = tmp[(tmp["I"] == "CB") & (tmp["J"] == "CB")]

            # write the file, if needed
            if write:
                if prefix is None:
                    logging.error("prefix is not defined")
                    exit(1)
                else:
                    # write the files in current directory
                    BB.to_csv(prefix + "_" + "kb_BB.txt", header=True, sep=" ", index=False)
                    BB_side.to_csv(prefix + "_" + "kb_BS.txt", header=True, sep=" ", index=False)
                    SS.to_csv(prefix + "_" + "kb_SS.txt", header=True, sep=" ", index=False)

            return BB, BB_side, SS

        except Exception as e:
            logging.warning("Error in splitting secondary structures --> {}".format(str(e)))

    def sepres(self, table=None, ressep=None):
        """
        :param table: table for sequence separation
        :param ressep: sequence separation to include (eg.  >= I,I + ressep), default is I,I+3)
        :return: DataFrame after separation
        """
        if table is None:
            table = self.table
        if ressep is None:
            ressep = self.ressep

        tmp = table[table["segidI"] == table["segidJ"]]
        tmp = tmp[
            (tmp["resI"] >= tmp["resJ"] + ressep) |
            (tmp["resJ"] >= tmp["resI"] + ressep)
            ]
        diff = table[table["segidI"] != table["segidJ"]]
        df = pd.concat([tmp, diff], axis=0)
        return df

    def sum_mean(self, segid: object = None) -> object:
        """
        Returns the sum, mean and standard deviation of residues based on the self.grouping

        """

        tab_sep = self.sepres()
        if self.splitMgt is not None:
            (BB, BS, SS) = self.splitSS()
            tab = self.splitMgt
            if tab == 'BB':
                tab_sep = self.sepres(table=BB)
            elif tab == 'BS':
                tab_sep = self.sepres(table=BS)
            elif tab == 'SS':
                tab_sep = self.sepres(table=SS)
            else:
                logging.warning("splitMGT not recognized")
        if segid:
            tab_sep = tab_sep[(tab_sep["segidI"] == segid) & (tab_sep["segidJ"] == segid)]
        tab_sum = tab_sep.groupby(self.grouping).sum()
        tab_mean = tab_sum.mean(axis=1)
        tab_std = tab_sum.std(axis=1)
        return tab_sum, tab_mean, tab_std

    def csm_mat(self, tab=None, segid=None, ressep=None):

        """
        Returns symmetric diagonally dominant residue-residue coupling strength matrix (CSM)

        """
        if tab is None:
            _, tab, _ = self.sum_mean(segid)

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


mgt = BuildMG(filename="apo_pdz.txt", splitMgt="BS", ressep=2)
# bb, bs, ss = mgt.splitSS()
# print("{}\n {}\n {}\n{}+".format(bb.shape[0], bs.shape[0], ss.shape[0], bb.shape[0]+bs.shape[0]+ss.shape[0]))
# print(mgt.table.shape[0])
# tab_sum, tab_mean, tab_std = mgt.sum_mean()
# print(tab_sum.head())
print(mgt.csm_mat())
