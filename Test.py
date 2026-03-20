import os
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print("Checking torch...")
import torch
print("Torch imported!")

print("Checking umap...")
import umap
print("UMAP imported!")

print("Checking transformers...")
from transformers import AutoTokenizer
print("Transformers imported!")