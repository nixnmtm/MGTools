from ProTools import ProTools
from MGTools import MGTools
prot = ProTools("1BE9")
print(prot.univ.residues.resids)
print(prot.univ.segments[1].segid)
print(prot.get_residNname(prot.univ.segments[0].segid)[1][0])
print(prot.get_residNname(prot.univ.segments[0].segid)[1][-1])
u = prot.univ
# import pandas as pd
# import numpy as np
# dd = pd.read_csv("C:/Users/nixon/Google Drive/coding/nw_notebooks/kb_5ms_apo_local_copy.txt", sep=" ")
#
# print(dd.head())
# atomids = np.unique(dd["I"])
# atmnames = u.atoms.names
# atmname_atmid = dict(zip(atomids, atmnames))
# dd['I'].replace(atmname_atmid, inplace=True)
# dd['J'].replace(atmname_atmid, inplace=True)
# print(dd.head())

# mgt = MGTools(dd, ressep=3)
# bb, bs, ss = mgt.split_sec_struc()

# print(bb.head())