#!usr/bin/python

import re
import numpy as np

class Molecule():

	def __init__(self, block):
		""" Class for a molecule read from a mol2 file. """

		# Class block
		self.block = block
		mol_name, num_atoms, num_bonds = self.get_molecule_info()

		# Variable attributes
		self._mol_name = mol_name
		self._num_atoms = num_atoms
		self._num_bonds = num_bonds

		# Constants for class
		self.delimiter_bond = '@<TRIPOS>BOND'
		self.delimiter_atom = '@<TRIPOS>ATOM'
		self.delimiter_molecule = '@<TRIPOS>MOLECULE'
		self.bond_dict = {'nc': 0, '1': 1, '2': 2, '3': 3, 'am': 4, 'ar': 5, 'du': 6, 'un': 7}

		self.verified = self.verify_sequence()
		if not self.verified:
			raise ValueError('Sequence of Record Type Indicators is not correct.')

		self._feature_matrix = self.get_feature_matrix()
		self._adj_matrix = self.get_adj_matrix()

	# Properties

	@property
	def mol_name(self):
		return self._mol_name

	@property
	def num_atoms(self):
		return self._num_atoms
	
	@property
	def num_bonds(self):
		return self._num_bonds
	
	@property
	def feature_matrix(self):
		return self._feature_matrix
	
	@property
	def adj_matrix(self):
		return self._adj_matrix
	
	# Utility functions

	def parse_entry(self, entry):
		""" Utility function to parse an entry in any record. """

		sp = entry.strip()
		sp = re.sub(' +', ' ', sp).split(' ')
		return sp

	def parse_atoms(self, atom):
		""" Utility function to parse an entry in the atom block. """
		# Parse the entry
		atom = self.parse_entry(atom)

		# Add specific parts of the entry
		created = []
		created.append(int(atom[0]))
		created.append(atom[1])
		created.append(float(atom[2]))
		created.append(float(atom[3]))
		created.append(float(atom[4]))
		created.append(float(atom[8]))
		
		return created

	# Methods

	def get_molecule_info(self):
		""" Get basic info about the molecule. """
		
		block = self.block
		info = self.parse_entry(block[2])
		
		mol_name = block[1].strip()
		
		try:
			num_atoms = info[0]
		except ValueError:
			raise ValueError('num_atoms could not be found.')
			
		try:
			num_bonds = info[1]
		except ValueError:
			raise ValueError('num_bonds could not be found')
			
		return mol_name, int(num_atoms), int(num_bonds)

	def verify_sequence(self):
		""" Function to verify if sequence of RTIs is correct in the block. """
		verified_sequence = ['@<TRIPOS>MOLECULE', '@<TRIPOS>ATOM', '@<TRIPOS>BOND']

		molecule_sequence = []
		for line in self.block:
			if line.startswith(self.delimiter_molecule):
				molecule_sequence.append(self.delimiter_molecule)
			elif line.startswith(self.delimiter_bond):
				molecule_sequence.append(self.delimiter_bond)
			elif line.startswith(self.delimiter_atom):
				molecule_sequence.append(self.delimiter_atom)

		return set(molecule_sequence) == set(verified_sequence)

	def get_bonds(self):
		""" Get the integer list form of the bond block. """

		block = self.block
		start = -1

		for i in range(len(block)):
			if block[i].startswith(self.delimiter_bond):
				start = i
				break
				
		inds = list(range(start + 1, len(block)))
		bonds = []
		
		for i in inds:
			bonds.append(self.parse_entry(block[i])) 
			
		for i in range(len(bonds)):
			bonds[i][3] = self.bond_dict[bonds[i][3]]
			bonds[i] = [int(x) for x in bonds[i]]

		return bonds

	def get_feature_matrix(self):
		""" Function to get the feature matrix from the molecule. """

		block = self.block
		start, end = -1, -1

		# Get the start and end indices of the block
		for i in range(len(block)):
			if block[i].startswith(self.delimiter_atom):
				start = i
			elif block[i].startswith(self.delimiter_bond):
				end = i
				break

		inds = list(range(start + 1, end))
		
		# Parse and add each entry to the atoms list
		atoms = []
		for i in inds:
			atoms.append(self.parse_atoms(block[i]))
		
		return np.array(atoms)

	def get_adj_matrix(self):
		""" Function to get the adjacency matrix from the molecule. """

		block = self.block
		adj_matrix = np.zeros(shape=(self.num_atoms, self.num_atoms), dtype='int')
		
		bonds = self.get_bonds()
		for bond in bonds:
			src = bond[1]
			dst = bond[2]
			b_type = bond[3]
			adj_matrix[src-1][dst-1] = b_type
		
		return adj_matrix