# DATA PREPARATION

import os
import numpy as np
from scipy.spatial.distance import squareform, pdist # needed for 3d case
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from Bio.PDB.PDBParser import PDBParser


class DataPreparation:
    RECEPTORS = "D:/Projects/Molecular docking (colab with Borodin)/DATA/RECEPTORS/"
    LIGANDS = "D:/Projects/Molecular docking (colab with Borodin)/DATA/LIGANDS/"
    ANNOTATIONS = "D:/Projects/Molecular docking (colab with Borodin)/DATA/BioLiP_2013-03-6.txt"

    
    NUMBER_OF_SAMPLES = 3000 # The number of pairs we want to prepare
    
    count_receptors = 0
    count_ligands = 0
    count_not_ligands = 0
    
    receptors = []
    ligands = []
    not_ligands = []
    
    training_data = []


    def distance_matrix(self, file, interpolation_size, interpolation_type):
        coords = []
        parser = PDBParser()
        structure = parser.get_structure("molecule", file.path)
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coords.append(atom.coord)
        coords_array = np.asarray(coords, dtype="float")
        distance = squareform(pdist(coords_array, 'euclidean'))
        resized = cv2.resize(distance, (interpolation_size,interpolation_size), interpolation=interpolation_type)

        return resized
    
    def make_data(self):
        count_rec = 0
        count_lig = 0

        receptor_names = []
        ligand_names = []

        with open(self.ANNOTATIONS) as fh: # Open file and read N-lines
            cnt = 0
            while cnt < self.NUMBER_OF_SAMPLES: # Set NUMBER_OF_SAMPLES or set "while True:" to read whole file
                line = fh.readline()
                tuple = line.split()
                pdb_id = tuple[0]
                chain = tuple[1]
                ligand_id = tuple[4]
                ligand_chain = tuple[5]
                ligand_number = tuple[6]
                receptor_names.append(pdb_id + chain + ".pdb")
                count_rec += 1
                ligand_names.append(pdb_id + "_" + ligand_id + "_" + ligand_chain + "_" + ligand_number + ".pdb")
                count_lig += 1
#                 print("Receptor: ", pdb_id + chain + ".pdb", "Ligand: ", pdb_id + "_" + ligand_id + "_" + ligand_chain + "_" + ligand_number + ".pdb")
                cnt += 1
        
        print(f"There are {count_rec} receptors and {count_lig} ligands in annotations")
        
        for receptor_name in tqdm(receptor_names):
            for receptor_file in os.scandir(self.RECEPTORS):
                if receptor_name in receptor_file.name:
#                     self.receptors.append(receptor_file.name)
                    self.receptors.append(self.distance_matrix(receptor_file, 256, cv2.INTER_LINEAR))
                    self.count_receptors += 1
        
        for ligand_name in tqdm(ligand_names):
            for ligand_file in os.scandir(self.LIGANDS):
                if ligand_name in ligand_file.name:
#                     self.ligands.append(ligand_file.name)
                    self.ligands.append(self.distance_matrix(ligand_file, 30, cv2.INTER_AREA))
                    self.count_ligands += 1
                    
        print(f"Length of receptors {len(self.receptors)}, length of ligands {len(self.ligands)}")
                    
        # Add molecules that cannot interact with receptors

        for ligand_name in tqdm(ligand_names):
            for ligand_file in os.scandir(self.LIGANDS):
                if ligand_name[:3] not in ligand_file.name and len(self.not_ligands) < len(self.receptors):
#                     self.not_ligands.append(ligand_file.name)
                    self.not_ligands.append(self.distance_matrix(ligand_file, 30, cv2.INTER_AREA)) # not_ligands
                    self.count_not_ligands += 1
        
        print(f"Number of receptors {self.count_receptors}; number of ligands {self.count_ligands}, number of not_ligands {self.count_not_ligands}")
        
        # Join receptors and ligands, add label
        for receptor, ligand, not_ligand in zip(self.receptors, self.ligands, self.not_ligands):
            self.training_data.append([receptor, ligand, np.eye(2)[0]])
            self.training_data.append([receptor, not_ligand, np.eye(2)[1]])
            
        np.random.shuffle(self.training_data) # Not nessecary, can be commented.
        print("Number of train data: ", len(self.training_data))
        np.save("D:/Projects/Molecular docking (colab with Borodin)/DATA/training_data.npy", self.training_data)
        
#         print("DATA ", self.training_data)


DataPreparation().make_data()
print("DONE")
