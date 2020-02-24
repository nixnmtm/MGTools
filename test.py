# #from MGTools import MGTools
# import pandas as pd
# import numpy as np
# import logging
#
# # below two functions are written to avil withoutsplitMGT also, noe decided to make spilMGT as default
#
# def table_sum(self):
#     """
#     Returns the sum table based on the self.grouping
#
#     :return: splitMGT==True : dict of segids with  tables with keys ["BB", "BS", "SS"]
#              splitMGT==False : dict of segids with sum of complete table
#
#     """
#     smtable = dict()
#     if self.splitMGT:
#         sstable = self.splitSS()
#     for seg in self.table.segidI.unique():
#         smtable[seg] = dict()
#         if self.splitMGT:
#             for key in self.splitkeys:
#                 tmp = self.sepres(table=sstable[key]).groupby(self.grouping).sum()
#                 mask = (tmp.index.get_level_values("segidI") == seg) & \
#                        (tmp.index.get_level_values("segidJ") == seg)
#                 smtable[seg][key] = tmp[mask]
#         else:
#             tmp = self.sepres(table=self.table).groupby(self.grouping).sum()
#             mask = (tmp.index.get_level_values("segidI") == seg) & \
#                    (tmp.index.get_level_values("segidJ") == seg)
#             smtable[seg] = tmp[mask]
#     return smtable
#
#
# def table_mean(self):
#     """
#
#     :return: mean table
#     """
#
#     table = self.table_sum()
#     mntable = dict()
#     for seg in table.keys():
#         mntable[seg] = dict()
#         if isinstance(table[seg], dict) and self.splitMGT:
#             for key in table[seg].keys():
#                 mntable[seg][key] = table[seg][key].mean(axis=1)
#         elif isinstance(table[seg], pd.DataFrame):
#             mntable[seg] = table[seg].mean(axis=1)
#         else:
#             logging.warning("Unknown table format")
#     return mntable
#
# def mgt_mat(self):
#     """
#     Build MGT Matrix
#
#     :return: MGT dataframe
#
#     """
#
#     tab = self.table_mean()
#     mats = dict()
#     for seg in tab.keys():
#         mats[seg] = dict()
#         for key in tab[seg].keys():
#             if isinstance(tab[seg][key].index, pd.core.index.MultiIndex):
#                 tab[seg][key] = tab[seg][key].reset_index()
#             diag_val = tab[seg][key].groupby("resI").sum().drop("resJ", axis=1).values.ravel()
#             ref_mat = tab[seg][key].drop(["segidI", "segidJ"], axis=1).set_index(['resI', 'resJ']).unstack(fill_value=0).values
#             row, col = np.diag_indices(ref_mat.shape[0])
#             ref_mat[row, col] = diag_val
#             mats[seg][key] = pd.DataFrame(ref_mat, index=np.unique(tab[seg][key].resI.values), columns=np.unique(tab[seg][key].resI.values))
#
# #     return mats
# #
# #
# #
#
# from mgt.core import LoadKbTable, BaseMG, MGCore
# from mgt import utils
#
#
# load = LoadKbTable(filename="s1a_holo_200.txt", ressep=3)
# kb_tab = load.load_table()
#
# core = MGCore(kb_tab, segid="S1A", sskey="SS")
# print(core.calc_persistence())


#For getting exact num of subplots

import math
import numpy as np
from matplotlib import pyplot as plt



# https://stackoverflow.com/questions/28738836/matplotlib-with-odd-number-of-subplots

# x_variable = list(range(-5, 6))
# parameters = list(range(0, 13))
#
# figure, axes = generate_subplots(len(parameters), row_wise=True)
# for parameter, ax in zip(parameters, axes):
#     ax.plot(x_variable, [x**parameter for x in x_variable])
#     ax.set_title(label="y=x^{}".format(parameter))
#
# plt.tight_layout()
# plt.show()

from mgt.base import BaseMG
# from mgt.core import LoadKbTable, MGCore
# from mgt.utils.pers import PersistenceUtils
# from mgt.extras import get_aa_format
#
# tab = LoadKbTable(filename="s1a_holo_200.txt", input_path="./Inputs")
# kb_tab = tab.load_table()
# core = MGCore(kb_tab, segid="S1A", sskey="BB")
# print(core.windows_eigen_decom())
# pers = core.calc_persistence()
#
# # Get the eigenvalues of mean mgt matrix
# eigval, eigvec = core.eigh_decom()


# ss_struc = "SS"
# cdf_cut = {"SS": 0.85, "BS": 0.90, "BB": 0.90}
# pcut_range = np.arange(0.5, 0.81, 0.01)
# wt_cut_range = np.arange(0., 0.41, 0.01)
# wt_cut = 0.15
# pu = PersistenceUtils(eigenval=eigval, pers=pers, eigcut="outliers")
# pcut, _ = pu.get_pcutoff(cdf_cut=cdf_cut[ss_struc])
#
# # Get the persistence eigenmodes
# permodes = pu.get_pers_modes(pcut_range)
# # Get the amino acid formatted for labelling
# pdb_aa = get_aa_format("./PDB/3TGI.pdb")
# pdb_aa = pdb_aa[:core.nres - 1].tolist()
# pdb_aa.insert(len(pdb_aa), "CA")
#
# # Plot the persistance eigenmodes
# pu.plot_persmodes(eigvec, permodes, cdf_cut=cdf_cut[ss_struc], wt_cut=wt_cut, pdbaa=pdb_aa)
# all_res = pu.get_residues_persmodes(eigvec, pcut_range, wt_cut_range)
#
#
# print(f"Total number of residues obtained in PCUT:{pcut} and WCUT:{wt_cut} is :\n {len(all_res[str(pcut)][wt_cut])}")

# fontsize = 24
# legend_properties = {'weight': 'bold', 'size': fontsize}
# font = {'family': 'Arial', 'weight': 'bold','size': fontsize}
# font_ticks = {'family': 'Arial', 'weight': 'bold', 'size': fontsize}
#
# plt.rc('font', **font)
# fig, axes = generate_subplots(len(permodes[str(pcut)]), row_wise=True)
# for m, ax in zip(permodes[str(pcut)], axes):
#     aa_residues = []
#     cont_residues = []
#     m = m - 1  # modes are exact indices
#     ax.plot(np.arange(1, len(eigval) + 1), eigvec[:, m] * eigvec[:, m], label=r'$U^{%s}$' % (m + 1))
#     ax.axhline(wt_cut, color='k', linestyle='--', linewidth=0.5)
#     for k, p, l in zip(range(1, len(eigval) + 1), pdb_aa, eigvec[:, m] * eigvec[:, m]):
#         if l > wt_cut:
#             ax.annotate(str(p), xy=(k, l), rotation=90)
#             aa_residues.append(str(p))
#             cont_residues.append(str(k))
#     print(f"{aa_residues} <--> {cont_residues}")
#     ax.set_xlabel("Residues", fontdict=font)
#     ax.set_ylabel("Weight", fontdict=font)
#     ax.legend(loc='best', frameon=False, prop=legend_properties)
#     xticklab = [str(indx) for indx in range(0, len(eigval))][::50]
#     yticklab = [str(np.round(yindx, 2)) for yindx in np.arange(0.0, 1.0, 0.2)]
#     xticks = [indx for indx in range(0, len(eigval))][::50]
#     yticks = [np.round(yindx, 2) for yindx in np.arange(0.0, 1.0, 0.2)]
#     ax.set_xticks(xticks)
#     ax.set_yticks(yticks)
#     ax.set_xticklabels(xticklab, fontdict=font_ticks)
#     ax.set_yticklabels(yticklab, fontdict=font_ticks)
# fig.suptitle(title, fontsize=fontsize)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()


######## DISTMAT SNIPPET ###########
import MDAnalysis as mda
# from src.fluctmatch.fluctmatch import utils as fmutils
# from MDAnalysis.lib.distances import distance_array
# from src.fluctmatch.models.core import modeller
#
# u = modeller(("/Users/nix/mphysics/protease/fluct/holo/10/data/30/fluctmatch.xplor.psf"), "/Users/nix/mphysics/protease/fluct/holo/10/data/30/cg.dcd",
#              com="com", model=["ENM"])
# positions = fmutils.AverageStructure(u.atoms).run().result
# distmat = distance_array(positions, positions, backend="OpenMP")
# print(distmat.shape)
# print(distmat)
# bins = np.arange(1., 10, 0.5)
# import numpy as np
# np.savetxt("dist_mat_w30.txt", distmat)

import mgt.extras as ex
a = ex.get_pdb_resid("./PDB/3TGI.pdb")

print(np.insert(a, 223, "CA"))