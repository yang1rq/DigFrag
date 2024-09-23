import os
import csv
import time
import threading
from functools import wraps
from rdkit import Chem
from rdkit.Chem import BRICS, Recap, AllChem
import argparse


def read_molecules(file_path):
    file_type = file_path.split('.')[-1]
    if file_type == 'sdf':
        suppl = Chem.SDMolSupplier(file_path)
        mols = [x for x in suppl if x is not None]
    elif file_type == 'smi':
        suppl = Chem.SmilesMolSupplier(file_path)
        mols = [x for x in suppl if x is not None]
    elif file_type == 'mol':
        suppl = Chem.MolSupplier(file_path)
        mols = [x for x in suppl if x is not None]
    elif file_type == 'mol2':
        suppl = Chem.Mol2MolSupplier(file_path)
        mols = [x for x in suppl if x is not None]
    elif file_type == 'pdb':
        mols = [AllChem.MolFromPDBFile(file_path)]
    else:
        raise ValueError(f'Unsupported file type: {file_type}')

    if len(mols) == 0:
        raise ValueError('No molecules found in file!')

    return mols


# def timeout_decorator(timeout):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             def target_func():
#                 return func(*args, **kwargs)
#
#             thread = threading.Thread(target=target_func)
#             thread.start()
#             thread.join(timeout)
#
#             if thread.is_alive():
#                 print("Function execution timed out.")
#                 return None
#             else:
#                 return target_func()
#
#         return wrapper
#
#     return decorator


# @timeout_decorator(timeout=5)
def Brics_frag(mol):
    brics_fragments = BRICS.BRICSDecompose(mol)

    return brics_fragments


# @timeout_decorator(timeout=10)
def Recap_frag(mol):
    recap_tree = Recap.RecapDecompose(mol)
    recap_fragments = list(recap_tree.children.keys())

    return recap_fragments


# def MacFrac(smi_path, output):
#     func = './MacFrag.py'
#     print(output)
#     os.system(f'python {func} -i {smi_path} -o {output} -maxBlocks 6 -maxSR 8 -asMols False -minFragAtoms 1')


def main(smi_path, output, methods):
    # if ('Brics' in methods) or ('Recap' in methods):
    #     start = time.time()
    mols = read_molecules(smi_path)
    #     end = time.time()
    #     runtime = runtime = end - start
    #     print(f'read mols runtime: {runtime}')

    if 'Brics' in methods:
        # start = time.time()
        # with open('./number.csv', mode='w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        with open(os.path.join(output), 'w') as f:
            for mol in mols:
                brics_fragments = Brics_frag(mol)
                # writer.writerow([len(brics_fragments)])
                f.write('\n'.join(brics_fragments) + '\n')
            # end = time.time()
            # runtime = end - start
            # print(f'brics runtime: {runtime}')

    if 'Recap' in methods:
        # start = time.time()
        # with open('./number.csv', mode='w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        with open(os.path.join(output), 'w') as f:
            for mol in mols:
                recap_fragments = Recap_frag(mol)
                # writer.writerow([len(recap_fragments)])
                f.write('\n'.join(recap_fragments) + '\n')
            # end = time.time()
            # runtime = end - start
            # print(f'recap runtime: {runtime}')

    # if 'Macfrag' in methods:
        # start = time.time()
        # MacFrac(smi_path, output)
        # end = time.time()
        # runtime = end - start
        # print(f'macfrag runtime: {runtime}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make fragments for smiles file')
    parser.add_argument('-i', '--smi_path', required=True, type=str)
    parser.add_argument('-o', '--output', required=True, type=str)
    parser.add_argument('-m', '--method', required=True, choices=['Brics', 'Recap'], type=str)
    args = parser.parse_args()

    smi_path = args.smi_path
    output = args.output
    methods = args.method

    main(smi_path, output, methods)
