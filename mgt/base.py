import pandas as pd
import os
import logging
import numpy as np
import string


class LoadKbTable(object):

    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.module_path = os.path.dirname(os.path.realpath(__file__))
        self.input_path = kwargs.get("input_path", os.path.join(self.module_path + "/../Inputs"))

    def load_table(self) -> pd.DataFrame:
        """
        Load Coupling Strength DataFrame

        :return: Processed Coupling Strength DataFrame

        """
        logging.info("Loading '{}' from {}".format(self.filename, self.input_path))
        _, fileext = os.path.splitext(self.filename)

        if not (fileext[-3:] == "txt" or fileext[-3:] == "bz2"):
            logging.error("Please provide a appropriate file, with extension either txt or bz2")
            exit(1)
        filepath = os.path.join(self.input_path, self.filename)
        try:
            table = pd.read_csv(filepath, sep=' ')
            if str(table.columns[0]).startswith("Unnamed"):
                table = table.drop(table.columns[0], axis=1)
            logging.info("File loaded.")
            return table
        except IOError as e:
            logging.error(f'Error in loading file: {str(e)}')


class BaseMG(object):
    """
    Base class for building Coupling strength Table sum and mean for all segids including
    their secondary structure splits.

    :param filename: Name of the file to be loaded
    :key ressep: residue separation( >= I,I + ressep), (default=3)
    :key interSegs: two segments names for inter segment analysis, should be a tuple
    :key input_path: path of file input, should be a str, if not given searches filename in Inputs folder
    :returns:


    :Example:
    .. highlight:: python
    .. code-block:: python

        >>> from mgt.core import BaseMG
        >>> base = BaseMG(filename="holo_pdz.txt.bz2", ressep=1)
        >>> print(base.table_mean()["CRPT"]["BB"].head())
        segidI  resI  segidJ  resJ
        CRPT    5     CRPT    6       169.063123
                              7         3.482413
                              8         0.139217
                              9         0.007373
                6     CRPT    5       169.063123
        dtype: float64

    """

    def __init__(self, table: pd.DataFrame, **kwargs):

        self.grouping = ["segidI", "resI", "segidJ", "resJ"]
        self._index = ["segidI", "resI", "I", "segidJ", "resJ", "J"]
        self._complete_idx = ["segidI", "resI", "resnI", "I", "segidJ", "resJ", "resnJ", "J"]
        self.splitkeys = ["BB", "BS", "SS"]

        # kwargs
        self.ressep = kwargs.get('ressep', 3)
        self.interSegs = kwargs.get('interSegs', None)  # should be a tuple
        self.table = table
        self.tot_nres = len(self.table.resI.unique())

    def splitSS(self, write: bool = False, exclude_disul: bool = True) -> dict:
        """
        Split based on secondary structures.

        | BB - Backbone-Backbone Interactions
        | BS - Backbone-Sidechain Interactions
        | SS - Sidechain-Sidehain Interactions

        :param df: Dataframe to split. If None, df initialized during class instance is taken
        :param write: write after splitting
        :param exclude_disul: exclude disulphide interactions (default: True)
        :return: dict of split DataFrames

        .. todo::
            1. Try to include ion interaction with SS
            2. Remove ion interactions from BS

        """
        # split table into three tables based on BB,BS and SS
        # if not self.splitMGT:
        #     raise ValueError("splitMGT must be True to run splitSS method")
        #     exit(1)

        sstable = dict()
        tmp = self.table.copy(deep=True)
        try:
            # BACKBONE-BACKBONE
            sstable['BB'] = tmp[((tmp["I"] == 'N') | (tmp["I"] == 'O') | (tmp["I"] == 'ions'))
                                & ((tmp["J"] == 'N') | (tmp["J"] == 'O') | (tmp["J"] == 'ions'))]

            # BACKBONE-SIDECHAIN
            BS = tmp[((tmp["I"] == "N") | (tmp["I"] == 'O') | (tmp["I"] == "ions")) & (tmp["J"] == 'CB')]
            SB = tmp[(tmp["I"] == 'CB') & ((tmp["J"] == "N") | (tmp["J"] == 'O') | (tmp["J"] == "ions"))]
            sstable['BS'] = pd.concat([BS, SB], axis=0, ignore_index=True)

            # SIDECHAIN-SIDECHAIN
            tmp_SS = tmp[(tmp["I"] == "CB") & (tmp["J"] == "CB")]

            if exclude_disul:
                tmp_SS = tmp_SS.set_index(self._complete_idx)
                tmp_SS = tmp_SS[(tmp_SS < 100).any(axis=1)]
                sstable['SS'] = tmp_SS.reset_index()

            else:
                sstable['SS'] = tmp_SS


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
        Residue Separation

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

    def _nres(self, segid=None, ss=None):
        """
        Return number of residues in a segmet or inter-segments

        :param segid: segment id
        :param ss: splitkey
        :return: number of residue (int)

        """
        return len(self.table_sum()[segid][ss].index.get_level_values("resI").unique())

    def _resids(self, segid=None, ss=None):
        """
        Return resids of given segments.

        :param seg: segid
        :return: array of residue ids

        """

        return self.table_sum[segid][ss].index.get_level_values("resI").unique()

    def _segids(self):
        """
        Return segids of given table.

        :return: list of segids of given table.

        """

        return self.table.segidI.unique()

    def _refactor_resid(self, df):
        """
        Rename the resids of inter segments.
        | Method should be called only if segments has overlapping resids

        :param df: Dataframe with overalpping resids in segments
        :return: DataFrame with renamed resids
        """
        alphs = list(string.ascii_uppercase)
        if self.interSegs is not None:
            if isinstance(df.index, pd.core.index.MultiIndex):
                df = df.reset_index()
            for n, seg in enumerate(self.interSegs):
                resids = df[df.segidI == seg].resI.unique()
                if not resids.dtype == str:
                    mapseg = [alphs[n] + str(i) for i in resids]
                    mapd = dict(zip(resids, mapseg))
                    df.loc[df['segidI'] == seg, 'resI'] = df['resI'].map(mapd)
                    df.loc[df['segidJ'] == seg, 'resJ'] = df['resJ'].map(mapd)
            renamed = df.set_index(self.grouping)
            return renamed
        else:
            logging.warning(f"interSegs argument is None, but {self._refactor_resid.__name__} invoked")

    def _comp_resid(self, df):
        """
        Compare resids of the two segments and return True if overlap exists in resids

        :param df: DataFrame to check for comparision
        :return: Boolean
        """

        if self.interSegs is not None:
            if isinstance(df.index, pd.core.index.MultiIndex):
                df = df.reset_index()
            r1 = df[df.segidI == self.interSegs[0]].resI.unique()
            r2 = df[df.segidI == self.interSegs[1]].resI.unique()
            return np.intersect1d(r1, r2).size > 0

    def table_sum(self):
        """
        Returns the sum table based on the self.grouping

        :return: dict of sum tables

        """

        smtable = dict()
        sstable = self.splitSS()
        if self.interSegs is not None:
            seg1 = self.interSegs[0]
            seg2 = self.interSegs[1]

        for seg in self.table.segidI.unique():
            smtable[seg] = dict()
            for key in self.splitkeys:
                tmp = self.sepres(table=sstable[key]).groupby(self.grouping).sum()
                mask = (tmp.index.get_level_values("segidI") == seg) & \
                       (tmp.index.get_level_values("segidJ") == seg)
                smtable[seg][key] = tmp[mask]

        if self.interSegs is not None and isinstance(self.interSegs, tuple):
            smtable[self.interSegs] = dict()
            if seg1 == seg2:
                raise IOError("Inter segments should not be same")
            for key in self.splitkeys:
                tmp = self.sepres(table=sstable[key]).groupby(self.grouping).sum()
                mask = (tmp.index.get_level_values("segidI") == seg1) & \
                       (tmp.index.get_level_values("segidJ") == seg2)
                revmask = (tmp.index.get_level_values("segidI") == seg2) & \
                          (tmp.index.get_level_values("segidJ") == seg1)
                diff = pd.concat([tmp[mask], tmp[revmask]], axis=0)
                same = pd.concat([smtable[seg1][key], smtable[seg2][key]], axis=0)
                inter = pd.concat([same, diff], axis=0)
                if self._comp_resid(inter):
                    logging.warning("resids overlap, refactoring resids")
                    inter = self._refactor_resid(inter)
                smtable[self.interSegs][key] = inter
        return smtable

    def table_mean(self):
        """
        Return Mean of table

        :return: dict of mean tables, format as table_sum()

        """
        stab = self.table_sum()
        mntable = dict()
        for seg in stab.keys():
            mntable[seg] = {key: stab[seg][key].mean(axis=1) for key in stab[seg].keys()}
        return mntable