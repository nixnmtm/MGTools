import pandas as pd
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


class BuildMG(object):
    """
    1. Base class for loading Coupling strength dataframe.
    """
    def __init__(self, filename: str, **kwargs):
        """
        :param filename: Name of the file to be loaded
        :param dirpath: Directory Path of file
        """
        self.filename = filename
        self.dir_path = '../Inputs/'
        self.table = self.load_table()
        if "ressep" in kwargs.keys():
            self.ressep = kwargs["ressep"]

    def load_table(self, verbose: bool = False) -> pd.DataFrame:
        """
        Load Coupling Strength DataFrame
        :param verbose: verbosity
        :return: Structured coupling strength DataFrame

        """
        if verbose:
            logging.info("Loading file: {} from {}".format(self.filename, self.dir_path))
        _, fileext = os.path.splitext(self.filename)
        if fileext[-3:] != "txt":
            logging.error("Please provide a txt file")
        filepath = os.path.join(self.dir_path, self.filename)
        os.path.exists(filepath)
        table = pd.read_csv(os.path.join(filepath), sep=' ')
        if str(table.columns[0]).startswith("Unnamed"):
            table = table.drop(table.columns[0], axis=1)
        if verbose:
            logging.info("File loaded")
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

    # TODO: Need to think about including ressep efficiently
    # TODO: MGBuild is just upto building the symmetric matrix
    # TODO: So Sum, Mean and CSM construction need to be included in this - think and do
    def sepres(self):
        """
        Separate residues with given sequence separation number (ressep)

        :return: DataFrame after residue separation
        """
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

    def csm_mat(self, tab, segid=None, ressep=None):
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
for file in os.listdir('../Inputs/'):
    mgt = BuildMG(filename=file)
    print(file)
    bb, bs, ss = mgt.splitSS()
    print(mgt.table.shape[0])
    print("{}\n {}\n {}\n{}+".format(bb.shape[0], bs.shape[0], ss.shape[0], bb.shape[0]+bs.shape[0]+ss.shape[0]))
