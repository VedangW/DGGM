#!usr/bin/python

import os
import click
import pickle
import multiprocessing

from Molecule import Molecule

delimiter = '@<TRIPOS>MOLECULE'

def create_Molecule(block):
	try:
		mol = Molecule(block)
	except ValueError as e:
		print (str(e))
		return None

	return mol

def train_test_split(molecule_graphs):
	test = molecule_graphs[:32000]
	train = molecule_graphs[32000:]

	print ("Saving train and test files...")

	f = open('train.pkl')
	pickle.dump(train, f)
	f.close()

	f = open('test.pkl')
	pickle.dump(test, f)
	f.close()

def preprocess():
	molecule_graphs = []
	for fname in os.listdir('graph_data'):
		# Open file and read lines.
		print ("Opening " + fname + "...")
		f = open('graph_data/' + fname)
		lines = f.readlines()

		# Collect indices of all lines starting with delimiter.
		print ("Collecting start indices...")
		start = []
		for i in range(len(lines)):
			if lines[i].startswith(delimiter):
				start.append(i)

		# Read all lines between consecutive indices as a block.
		print ("Collecting blocks...")
		blocks = []
		for i in range(len(start)):
			if i != len(start) - 1:
				block = lines[start[i]:start[i + 1]]
			else:
				block = lines[start[i]:len(lines)]
			blocks.append(block)

		# Use each block to create a molecule object and 
		# add the feature and adjacency matrix of each to 
		# a list of molecules.

		print ("Creating Molecules...")
		molecules = []
		with click.progressbar(blocks) as bar:
			# pool = multiprocessing.Pool(2)
			# molecules = pool.map(create_Molecule, bar)
			for block in bar:
				molecules.append(create_Molecule(block))

		molecule_graphs += molecules
		print ("")

	print ("Splitting dataset into train and test...")
	molecule_graphs = np.array([x for x in molecule_graphs if x is not None])
	train_test_split(molecule_graphs)
	print ("Done.")

def main():
	preprocess()

if __name__ == "__main__":
	main()