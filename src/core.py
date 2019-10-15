import pandas as pd
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


class BuildMG(object):
    """
    Base class for building Coupling strength dataframe.

    :Example:
    .. highlight:: python
    .. code-block:: python

        >>> from src.core import BuildMG
        >>> mgt = BuildMG(filename="holo_pdz.txt.bz2", ressep=1, splitMgt="SS")
        >>> print(mgt.mgt_mat())

                  5         6         7         8         9
        5  0.025300  0.022754  0.002543  0.000003  0.000000
        6  0.022754  0.105062  0.068219  0.013300  0.000788
        7  0.002543  0.068219  0.239772  0.086187  0.082822
        8  0.000003  0.013300  0.086187  0.370769  0.271280
        9  0.000000  0.000788  0.082822  0.271280  0.354890

    """

    def __init__(self, filename: str, ressep=3, splitMgt=None, segid=None):
        """
        :func:`__init__` method docstring.
        Creates a new :class:`BuildMG` instance.

        :param filename: Name of the file to be loaded
        :param ressep: residue separation( >= I,I + ressep), (default=3)
        :type int
        :param splitMgt: split mgt into BB or BS or SS
        :type str
        :param segid: segment id to analyze, if not mentioned all segid is used
        :type str
        """
        self.filename = filename
        self.module_path = os.path.dirname(os.path.realpath(__file__))
        self.input_path = os.path.join(self.module_path + "/../Inputs")
        self.table = self.load_table()
        self.grouping = ["segidI", "resI", "segidJ", "resJ"]
        self._index = ["segidI", "resI", "I", "segidJ", "resJ", "J"]
        self.ressep = ressep
        self.splitMgt = splitMgt
        self.segid = segid

    def load_table(self) -> pd.DataFrame:
        """
        Load Coupling Strength DataFrame

        :return: Processed Coupling Strength DataFrame

        """
        logging.info("Loading file: {} from {}".format(self.filename, self.input_path))
        _, fileext = os.path.splitext(self.filename)

        if not (fileext[-3:] == "txt" or fileext[-3:] == "bz2"):
            logging.error("Please provide a appropriate file, with extension either txt or bz2")
        filepath = os.path.join(self.input_path, self.filename)
        os.path.exists(filepath)
        table = pd.read_csv(os.path.join(filepath), sep=' ')
        if str(table.columns[0]).startswith("Unnamed"):
            table = table.drop(table.columns[0], axis=1)
        logging.info("File loaded.")
        return table

    def splitSS(self, df: pd.DataFrame = None, write: bool = False) -> dict:
        """
        Split based on secondary structures.

        BB - Backbone-Backbone Interactions
        BS - Backbone-Sidechain Interactions
        SS - Sidechain-Sidehain Interactions

        :param df: Dataframe to split. If None, df initialized during class instance is taken
        :param write: write after splitting
        :return: dict of split DataFrames

        """
        # split table into three tables based on BB,BS and SS
        sstable = dict()

        if df is None:
            tmp = self.table.copy(deep=True)
        else:
            tmp = df.copy(deep=True)
        try:
            # BACKBONE-BACKBONE
            sstable['BB'] = tmp[((tmp["I"] == 'N') | (tmp["I"] == 'O') | (tmp["I"] == 'ions')) \
                      & ((tmp["J"] == 'N') | (tmp["J"] == 'O') | (tmp["J"] == 'ions'))]

            # BACKBONE-SIDECHAIN
            BS = tmp[((tmp["I"] == "N") | (tmp["I"] == 'O') | (tmp["I"] == "ions")) & (tmp["J"] == 'CB')]
            SB = tmp[(tmp["I"] == 'CB') & ((tmp["J"] == "N") | (tmp["J"] == 'O') | (tmp["J"] == "ions"))]
            sstable['BS'] = pd.concat([BS, SB], axis=0, ignore_index=True)

            # SIDECHAIN-SIDECHAIN
            sstable['SS'] = tmp[(tmp["I"] == "CB") & (tmp["J"] == "CB")]

            # write the file, if needed
            if write:
                # write the files in current directory
                sstable['BB'].to_csv("kb_BB.txt", header=True, sep=" ", index=False)
                sstable['BS'].to_csv("kb_BS.txt", header=True, sep=" ", index=False)
                sstable['SS'].to_csv("kb_SS.txt", header=True, sep=" ", index=False)

            return sstable

        except Exception as e:
            logging.warning("Error in splitting secondary structures --> {}".format(str(e)))

    def sepres(self, table: object = None, ressep: object = None) -> object:
        """
        :param table: table for sequence separation
        :param ressep: sequence separation to include (eg.  >= I,I + ressep), default is I,I+3)
        :return: DataFrame after separation
        """
        if table is None:
            table = self.table
        if ressep is None:
            ressep = self.ressep
        logging.info("DataFrame is populated with ressep: {}".format(ressep))
        tmp = table[table["segidI"] == table["segidJ"]]
        tmp = tmp[
            (tmp["resI"] >= tmp["resJ"] + ressep) |
            (tmp["resJ"] >= tmp["resI"] + ressep)
            ]
        diff = table[table["segidI"] != table["segidJ"]]
        df = pd.concat([tmp, diff], axis=0)
        return df

    def sum_mean(self, table=None, segid=None, ressep=None):
        """
        Returns the sum, mean and standard deviation of residues based on the self.grouping

        :param table: DataFrame to sum
        :param segid: segment id
        :param ressep: Number of residues separation
        :return: list of DataFrames [sum, mean, stand deviation]
        """
        if table is None:
            if self.splitMgt is not None:
                sstable = self.splitSS()
                tab = self.splitMgt
                if tab == 'BB':
                    tab_sep = self.sepres(table=sstable['BB'])
                elif tab == 'BS':
                    tab_sep = self.sepres(table=sstable['BS'])
                elif tab == 'SS':
                    tab_sep = self.sepres(table=sstable['SS'])
                else:
                    logging.warning("splitMGT not recognized")
            else:
                tab_sep = self.sepres(ressep)
        else:
            tab_sep = self.sepres(table, ressep)

        if segid is not None:
            logging.info("only segid: {} is populated".format(segid))
            tab_sep = tab_sep[(tab_sep["segidI"] == segid) & (tab_sep["segidJ"] == segid)]
        tab_sum = tab_sep.groupby(self.grouping).sum()
        tab_mean = tab_sum.mean(axis=1)
        tab_std = tab_sum.std(axis=1)
        return tab_sum, tab_mean, tab_std

    def mgt_mat(self, segid=None, ressep=None):
        """
        Build MGT Matrix

        :param segid: input segment id
        :param ressep: input residue separation
        :param segsplit: True: split matrix based on segids(ie. if 2 segids present, list of 2 dataframes are returned)
                         False: A single matrix with inter- & intra- segement interactions is returned
        :return: MGT dataframe

        """
        def mgtcore(df):
            """

            :param df:
            :return:
            """
            #tmp = df.copy(deep=True)
            diag_val = df.groupby("resI").sum().drop("resJ", axis=1).values.ravel()
            ref_mat = df.drop(["segidI", "segidJ"], axis=1).set_index(['resI', 'resJ']).unstack(fill_value=0).values
            row, col = np.diag_indices(ref_mat.shape[0])
            ref_mat[row, col] = diag_val
            ref_mat = pd.DataFrame(ref_mat, index=np.unique(df.resI.values), columns=np.unique(df.resI.values))
            # # Maintain table shape by adding zero for glycine residues missing in SS
            # if self.splitMgt == "SS":
            #     start = np.unique(tmp["resI"].values)[0]  # start residue number
            #     end = np.unique(tmp["resI"].values)[-1]  # end residue number
            #     ref_mat = ref_mat.reindex(np.arange(start, end+1)).T.reindex(np.arange(start, end+1)).replace(np.nan, 0.0)
            return ref_mat

        _, tab, _ = self.sum_mean(segid=segid, ressep=ressep)

        try:
            tab.ndim == 1
        except TypeError:
            logging.error('Dimension of the mat should not exceed 1, as we are stacking from each column')
        else:
            _tab = tab.reset_index()
            segments = np.unique(_tab["segidI"].values)
            print(segments)
            if len(segments) > 1:
                dfs = list()
                for seg in segments:
                    tmp = _tab[(_tab["segidI"] == seg) & (_tab["segidJ"] == seg)]
                    dfs.append(mgtcore(tmp))
                return dfs
            else:
                return mgtcore(_tab)
