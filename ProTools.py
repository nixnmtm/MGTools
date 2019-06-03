import MDAnalysis as mda
from Bio.PDB import PDBList
import os
import urllib.request


class ProTools(object):

    def __init__(self, pdbid):
        self.pdbid = pdbid
        self.dir_name = "./PDB/"
        self.filename = self.pdbid + ".pdb"
        self.filepath = os.path.join(self.dir_name, self.filename)
        if not os.path.isfile(self.filepath):
            self.get_pdb(format="pdb")
        self.univ = mda.Universe(self.filepath)

    def get_pdb(self, format=None):
        """This gives only the .cif format"""
        if format == "cif":
            pdbl = PDBList()
            pdbl.retrieve_pdb_file(self.pdbid, pdir='PDB')

        """This gives pdb format"""
        if format == "pdb":
            urllib.request.urlretrieve('https://files.rcsb.org/download/' + self.pdbid + '.pdb',
                                       './PDB/' + self.pdbid + '.pdb')

    def get_residNname(self, segid=None):
        """
        Retreive the resids and resnames.
        if segid given retreives only that segment or
        retreives all the segments.

        Returns dictionary of resids and resnames with segids as keys
        """
        protseg = dict()
        if segid is None:
            segs = self.univ.segments.segids
            for seg in segs:
                _resids = self.univ.select_atoms("segid {} and not (resname HOH)".format(seg)).residues.resids
                _resnames = self.univ.select_atoms("segid {} and not (resname HOH)".format(seg)).residues.resnames
                protseg[seg] = dict(zip(_resids, _resnames))
        else:
            _resids = self.univ.select_atoms("segid {} and not (resname HOH)".format(segid)).residues.resids
            _resnames = self.univ.select_atoms("segid {} and not (resname HOH)".format(segid)).residues.resnames
            protseg[segid] = dict(zip(_resids, _resnames))
        return protseg