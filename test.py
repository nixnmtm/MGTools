from ProTools import ProTools

prot = ProTools("1BE9")
res = prot.get_residNname()
print(res["A"])
print(prot.univ.residues)