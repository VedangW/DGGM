#!usr/bin/python

""" Module to preprocess the data from several batches
	of data files. 

	The path to the directory containing data files is to
	be passed as a command-line argument.
"""

import os
import sys
import click
import multiprocessing
import concurrent.futures

from time import time
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys

def retrieve_smiles(l):
	""" Retrieves the smiles representation from 
		a line in file.

		Parameters
		----------
		l: str
			Line in file

		Returns
		-------
		entry_in_dataset: list
			A list with 1st element as the index 
			and second as the smiles string.
	"""
	
	l = str(l)
	l = l.split("\\t")
	entry_in_dataset = [l[0].split("'")[1], l[1].split("\\n")[0]] 
	# print (entry_in_dataset)
	return entry_in_dataset

def convert_to_fingerprint(s):
	""" Convert a smiles string to a MACCS fingerprint

		Parameters
		----------
		s: str
			The smiles string for a molecule
	
		Returns
		-------
		macccs_key: str or None
			Binary string if generation is possible
			and None otherwise
	"""

	try:
		# Convert SMILES to Molecule object
		molecule = Chem.MolFromSmiles(s)
		# Get MACCS Key from Molecule object
		maccs_key = MACCSkeys.GenMACCSKeys(molecule)
		return maccs_key.ToBitString()
	except:
		return None

def remove_NoneTypes(maccs):
	""" Function to remove NoneType objects from
		the list 'maccs'.

		Parameters
		----------
		maccs: list
			Contains the list of maccs from file

		Returns
		-------
		maccs: list
			Maccs without NoneType objects
	"""

	# Filter None from maccs
	maccs = list(filter(None, maccs))

	# Check if all fingerprints in maccs are strings
	for fp in maccs:
		if not (type(fp) == str):
			# If not, raise an error
			raise ValueError('Something except a string in maccs.')

	return maccs

def check_maccs(maccs):
	""" Check number of features and homogeneity of
		the dataset.

		Parameters
		----------
		maccs: list
			Contains the list of maccs from file

		Returns
		-------
		ret_check: boolean
			False if data is homogeneous and True otherwise
	"""

	# print number of features
	print ("Number of features =", len(maccs[0]))

	# Check if size of all fingerprints is 167
	count = 0
	for fp in maccs:
		if len(fp) != 167:
			count += 1

	if count == 0:
		print ("All instances have length 167.")
	else:
		print ("Data not uniform. Check lengths for instances.")
		return False

	return True

def get_smiles_from_file(f):
	""" Gets SMILES from a file 'f'. 

		Parameters
		----------
		f: file
			File containing smiles.

		Returns
		-------
		smiles: list
			A list of smiles taken from file 'f'.
	"""

	lines = f.readlines()

	# Create SMILES list
	smiles = []
	for i in range(len(lines)):
		smiles.append(retrieve_smiles(lines[i]))

	# Remove all empty strings
	smiles = list(filter(None, [smiles[i][1] for i in range(len(smiles))]))
	
	return smiles

def perform_filecheck():
	""" Checks current number of training and testing
		samples in the dataset.
	"""

	# Open files
	train = open('train_aae_final', 'r')
	test = open('test_aae_final', 'r')


	# Check number of training and testing samples
	print ("")
	print ("Number of training samples =", len(train.readlines()))
	print ("Number of testing samples =", len(test.readlines()))
	print ("")

	train.close()
	test.close()

def save_data(maccs, split_ratio):
	""" Saves the MACCS fps into txt files.

		Parameters
		----------
		maccs: list
			A list of MACCS fps
		split_ratio: float
			The ratio in which the train and test
			set is to be divided.
	"""
	
	# Split dataset according to the split ratio
	train_set = maccs[:int(len(maccs)*split_ratio)]
	test_set = maccs[int(len(maccs)*split_ratio):]

	train = open('train_aae_final', 'a')
	test = open('test_aae_final', 'a')

	# Save the train set
	print ("Saving train set...")
	with click.progressbar(list(range(len(train_set)))) as bar:
		for i in bar:
			train.write(str(i) + '\t' + train_set[i] + '\n')    
	print ("")

	# Save the test set
	print ("Saving test set...")
	with click.progressbar(list(range(len(test_set)))) as bar:
		for i in bar:
			test.write(str(i) + '\t' + test_set[i] + '\n')
	print ("")    
	
	# Close files
	train.close()
	test.close()

	print ("Done.")

def preprocess(directory):
	""" A function to retrieve SMILES strings from
		a set of files, convert them into MACCS fps
		and store them as txt files elsewhere
		according to a train-test split_ratio.

		Process pools are used for calling 
		convert_to_fingerprint to use all cores for
		computation and speed it up.

		Parameters
		----------
		directory: str
			Path to directory where files are stored.
	""" 
	print ("Preprocessing data from " + directory + " =>")
	
	smiles = []
	files = os.listdir(directory)

	# Read the data from the files 
	print ("Reading data..." )
	with click.progressbar(list(range(len(files)))) as bar:
		for i in bar:
			f = open(directory + "/" + files[i], 'rb')
			smiles += get_smiles_from_file(f)
			f.close()

	# print (smiles)

	t0 = time()
	# Convert the SMILES strings to MACCS fingerprints
	print ("Converting SMILES to MACCS fingerprints...")
	with concurrent.futures.ProcessPoolExecutor() as executor:
		with click.progressbar(smiles) as bar:
			# Convert all SMILES to MACCS Keys
			print ("")
			maccs = []
			# Create a thread pool with size = no_of_cores
			pool = multiprocessing.Pool(multiprocessing.cpu_count())
			maccs = pool.map(convert_to_fingerprint, bar)
			print ("")

	print ("")
	t1 = time() - t0
	print ("Time taken =", t1, "s")

	# Remove all NoneType objects from maccs
	maccs = remove_NoneTypes(maccs)    
	
	# Check different statistics for maccs
	ret_check = check_maccs(maccs)

	# Save data
	print ("Saving files =>")
	print ("")
	save_data(maccs, 0.999)
	
	# Get statistics for current number of train
	# and test instances
	perform_filecheck()

def main():
	multiprocessing.freeze_support()
	preprocess(sys.argv[1])
	
if __name__ == "__main__":
	main()