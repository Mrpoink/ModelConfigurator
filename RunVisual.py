import os
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print("Starting imports....")
from ModelBackEnd.LoadModel import Model
from MapBackEnd.LoadMap import Map
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Starting program....")

model = Model()
map = Map()

while True:
    
    userin = input("Enter prompt (or q to quit): ")
    
    if userin.lower() == "q":
        exit(0)
    
    model_out, attention, input_tensors =  model.inference(userin)
    
    map.setup(attention)
    
    map.plot()