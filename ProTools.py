import MDAnalysis as mda
from Bio.PDB import PDBList
import os
import urllib.request
import logging


class ProTools(object):

    def __init__(self, pdbid):
        self.dir_name = "./PDB/"
        self.pdbid = pdbid
        if pdbid is None:
            logging.error("No pdbid provided")
            exit(1)
        self.filename = pdbid + ".pdb"
        self.filepath = os.path.join(self.dir_name, self.filename)
        if not os.path.isfile(self.filepath):
            self.download_pdb(format="pdb")
        self.univ = mda.Universe(self.filepath)

    def download_pdb(self, pdbid=None, format=None):
        """This gives pdb format"""
        if pdbid is None:
            pdbid = self.pdbid

        if format == "pdb" or format is None:
            urllib.request.urlretrieve('https://files.rcsb.org/download/' + pdbid + '.pdb',
                                       './PDB/' + pdbid + '.pdb')

        """This gives only the .cif format"""
        if format == "cif":
            pdbl = PDBList()
            pdbl.retrieve_pdb_file(pdbid, pdir='PDB')

    def n_residues(self, segid=None):
        if segid is None:
            total_res = dict()
            segs = self.univ.segments.segids
            for seg in segs:
                total_res[seg] = len(self.univ.select_atoms("segid {} and protein".format(seg)).residues.resids)
            return total_res
        else:
            return len(self.univ.select_atoms("segid {} and protein".format(segid)).residues.resids)

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
        return protseg, _resids