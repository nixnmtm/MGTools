# from MGTools import MGTools
# import pandas as pd
# import numpy as np
#
# pdz3 = "C:/Users/nixon/Google Drive/coding/nw_notebooks/pdz3_kb_complete_apo_local_copy.txt"
# kb_BB = "C:/Users/nixon/Google Drive/coding/nw_notebooks/kb_SS.txt"
# #protease = "C:/Users/nixon/Google Drive/coding/nw_notebooks/kb_5ms_apo_local_copy.txt"
# """The table should be a complete table with residue name and atom name already included"""
# mgt = MGTools(pdbid="1BE9", table_path=pdz3, ressep=3)
# mgt1 = MGTools(pdbid="3TGI", table_path=kb_BB, ressep=3)
#
# print(mgt.table.head())
# print(mgt1.table.head())
# print(mgt.table.columns[0])
# if str(mgt1.table.columns[0]).startswith("Unnamed"):
#     mgt1.table.drop(mgt.table.colums, axis=1)
# print(mgt1.table)
# # bb, bs, ss = mgt.split_sec_struc()
# # s, m, sd = mgt.sum_mean()
# #tmat, tvec, tval = mgt.t_eigenVect(SS=True)
# #print(tval[0])
# # print(len(mgt.get_residNname(segid=mgt.univ.segments[0].segid)[1]))
# # print(len(np.unique(mgt.table.resI)))
# # print((mgt.univ.segments[0].residues.resnames == "GLY").sum())
# # print(mgt.univ.segments[0].residues.resnames[:10])

didi = dict()
didi["ressep"] = 3

if "ressep" in didi.keys():
    print(True)