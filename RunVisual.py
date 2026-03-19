from MapBackEnd.LoadMap import Map
from ModelBackEnd.LoadModel import Model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Starting....")

model = Model()
map = Map()

while True:
    
    userin = input("Enter prompt (or q to quit): ")
    
    if userin.lower() == "q":
        exit(0)
    
    model_out, attention, input_tensors =  model.inference(userin)
    
    Map.setup(attention)
    
    Map.plot()