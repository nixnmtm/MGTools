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
        >>> mgt = BuildMG(filename="holo_pdz.txt.bz2", ressep=1)
        >>> print(mgt.mgt_mat())

                  5         6         7         8         9
        5  0.025300  0.022754  0.002543  0.000003  0.000000
        6  0.022754  0.105062  0.068219  0.013300  0.000788
        7  0.002543  0.068219  0.239772  0.086187  0.082822
        8  0.000003  0.013300  0.086187  0.370769  0.271280
        9  0.000000  0.000788  0.082822  0.271280  0.354890

    """

    def __init__(self, filename: str, ressep=3, **kwargs):
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
        self.interSegs = kwargs.get('interSegs')  # should be a tuple
        self.splitkeys = ["BB", "BS", "SS"]

    def load_table(self) -> pd.DataFrame:
        """
        Load Coupling Strength DataFrame

        :return: Processed Coupling Strength DataFrame

        """
        logging.info("Loading '{}' from {}".format(self.filename, self.input_path))
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

    def splitSS(self, write: bool = False) -> dict:
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
        # if not self.splitMGT:
        #     raise ValueError("splitMGT must be True to run splitSS method")
        #     exit(1)

        sstable = dict()
        tmp = self.table.copy(deep=True)

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

    def sepres(self, table) -> object:
        """
        :param table: table for sequence separation
        :param ressep: sequence separation to include (eg.  >= I,I + ressep), default is I,I+3)
        :return: DataFrame after separation
        """

        ressep = self.ressep
        # logging.info("DataFrame is populated with ressep: {}".format(ressep))
        tmp = table[table["segidI"] == table["segidJ"]]
        tmp = tmp[
            (tmp["resI"] >= tmp["resJ"] + ressep) |
            (tmp["resJ"] >= tmp["resI"] + ressep)
            ]
        diff = table[table["segidI"] != table["segidJ"]]
        df = pd.concat([tmp, diff], axis=0)
        return df

    def table_sum(self):
        """
        Returns the sum table based on the self.grouping

        :return: dict of sunm tables

        :Example:
        .. highlight:: python
        .. code-block:: python

        >>> from src.core import BuildMG
        >>> mgt = BuildMG(filename="holo_pdz.txt.bz2", ressep=3, interSegs=("PDZ3", "CRPT"))
        >>> print(mgt.table_sum())

        {"CRPT": {"BB" :  df},
                 {"BS" :  df},
                 {"SS" :  df}
        ,
        "PDZ3": {"BB" :  df},
                {"BS" :  df},
                {"SS" :  df}
        },
        ("PDZ3", "CRPT"): {"BB" :  df},
                          {"BS" :  df},
                          {"SS" :  df}
        },

        """
        smtable = dict()
        sstable = self.splitSS()

        for seg in self.table.segidI.unique():
            smtable[seg] = dict()
            for key in self.splitkeys:
                tmp = self.sepres(table=sstable[key]).groupby(self.grouping).sum()
                mask = (tmp.index.get_level_values("segidI") == seg) & \
                       (tmp.index.get_level_values("segidJ") == seg)
                smtable[seg][key] = tmp[mask]

        if self.interSegs is not None and isinstance(self.interSegs, tuple):
            smtable[self.interSegs] = dict()
            if self.interSegs[0] == self.interSegs[1]:
                raise IOError("Inter segments should not be same")
            for key in self.splitkeys:
                tmp = self.sepres(table=sstable[key]).groupby(self.grouping).sum()
                mask = (tmp.index.get_level_values("segidI") == self.interSegs[0]) & \
                       (tmp.index.get_level_values("segidJ") == self.interSegs[1])
                revmask = (tmp.index.get_level_values("segidI") == self.interSegs[1]) & \
                       (tmp.index.get_level_values("segidJ") == self.interSegs[0])
                intertab = pd.concat([tmp[mask], tmp[revmask]], axis=0)
                fullsm = pd.concat([smtable[self.interSegs[0]][key], smtable[self.interSegs[1]][key]], axis=0)
                smtable[self.interSegs][key] = pd.concat([fullsm, intertab], axis=0)

        return smtable

    def table_mean(self):
        """
        Return Mean of table

        :return: dict of mean tables, format as table_sum()

        """

        table = self.table_sum()
        mntable = dict()
        for seg in table.keys():
            mntable[seg] = {key: table[seg][key].mean(axis=1) for key in table[seg].keys()}
        return mntable

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
