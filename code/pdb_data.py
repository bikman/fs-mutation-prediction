"""
Module for parsing the structure PDB files
"""
import os
from pathlib import Path

from Bio.PDB import PDBParser

from data_model import DistanceMatrix, ProteinData, ChainData, ResidueData
from utils import ECOD_FOLDER, get_protein_files_dict


class PdbDataParser(object):
    """
    Parser for PDB ECOD files
    """

    def __init__(self, log, closest_count):
        self.log = log
        self.closest_count = closest_count  # attn_len

    def parse_ecod_pdb_file(self, file_path):
        """
        Create list of PDB data objects from PDB file
        @param file_path: pdb file path
        @return: list of pdb datas
        """
        protein_name = self.get_protein_name_from_pdb_file(file_path)
        protein_datas = []
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('tmp', file_path)
            filename = os.path.basename(file_path)
            for chain in structure.get_chains():
                all_ca_atoms = [a for a in chain.get_atoms() if a.name == 'CA']
                if len(all_ca_atoms) == 0:
                    continue
                chain_name = chain.id
                p = ProteinData()
                p.file = filename
                p.name = protein_name
                chain_obj = ChainData()
                chain_obj.name = chain_name
                for ca_atom in all_ca_atoms:
                    residue = ResidueData()
                    residue.load(ca_atom)
                    chain_obj.ca_residues.append(residue)
                chain_obj.distance_matrix = DistanceMatrix()
                chain_obj.distance_matrix.load(all_ca_atoms)
                chain_obj.distance_matrix.calculate_closest_serials(self.closest_count)
                p.chain = chain_obj
                protein_datas.append(p)

        except Exception as e:
            self.log(f'Cannot parse file {file_path}')
            self.log(f'Error: {e}')
        return protein_datas

    @staticmethod
    def get_protein_name_from_pdb_file(file_path):
        """
        Returns short protein name from PDB file
        @param file_path: path to pdb file
        @return: name of the protein as string
        """
        file_name = os.path.basename(file_path)
        tokens = file_name.split('_')
        protein_name: str = tokens[0]
        return protein_name

    def create_pdb_datas_dict(self, ):
        """
        Parsing of structure PDB files
        @return: dictionary [protein name -> pdb data object]
        """
        pname_to_pdb_datas = {}
        for f in Path(ECOD_FOLDER).rglob('*.pdb'):
            protein_name = self.get_protein_name_from_pdb_file(f)
            created_pdb_data = False
            for file_name in get_protein_files_dict().values():
                if protein_name in file_name:
                    pdb_datas = self.parse_ecod_pdb_file(f)
                    self.log(f'Parsed {protein_name}: {len(pdb_datas)} chains')
                    if protein_name not in pname_to_pdb_datas:
                        pname_to_pdb_datas[protein_name] = pdb_datas
                    else:
                        pname_to_pdb_datas[protein_name] += pdb_datas
                    created_pdb_data = True
            if not created_pdb_data:
                self.log(f'- Skipped {protein_name} PDB parsing')
        return pname_to_pdb_datas


if __name__ == '__main__':
    # parser = PdbDataParser(print, 6)
    # pdata = parser._parse_ecod_pdb_file(r'C:\DATASETS\MAVE_for_michael.tar\MAVE_chains\IF-1_1AH9_A.pdb')[0]
    # print(pdata.chain.get_string())
    # matrix = pdata.chain.distance_matrix
    # print(matrix.serial_to_closest)
    pass
