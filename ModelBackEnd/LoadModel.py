from transformers import BertTokenizer, BertModel
import torch
from torch import nn

class Model(nn.Module):
    
    def __init__(self):
        
        print("Beginning Tokenizer Load")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("Tokenizer Loaded")
        self.model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        print("Model Loaded")
        
    def inference(self, input):
        r"""
        Params: Text input
        Returns: Model output tensors, Attention weights, Input Tensors
        """
        inputs = self.tokenizer(input, return_tensors="pt")
        
        outputs = self.model(**inputs)
        
        attentions = torch.stack(outputs.attentions).squeeze(1)
        
        num_layers, num_heads, seq_len, _ = attentions.shape
        
        flattened_attentions = attentions.view(num_layers * num_heads, -1).detach().numpy()
        
        return outputs, flattened_attentions, inputs