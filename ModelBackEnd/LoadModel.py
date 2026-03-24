print("Importing transformers....")
from transformers import AutoTokenizer, AutoModelForCausalLM
print("Importing torch....")
import torch
import numpy as np
from torch import nn

class Model(nn.Module):
    
    def __init__(self, model_id="HuggingFaceTB/SmolLM-135M-Instruct"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Beginning Tokenizer Load")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        print("Tokenizer Loaded")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True).to(self.device)
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.total_heads = self.num_layers * self.num_heads
        self.hidden_dim = self.model.config.hidden_size
        self.head_dim = self.hidden_dim // self.num_heads
        self.steering_scalars = torch.ones(self.total_heads)
        self.hooks = []
        self.messages = []
        print("Model Loaded")
        
    def _get_steering_pre_hook(self, layer_idx):
        def hook(module, args):
            # pre_hooks take (module, input_tuple) and return a modified input_tuple
            hidden_states = args[0]
            
            original_shape = hidden_states.shape
            if len(original_shape) == 2:
                hidden_states = hidden_states.unsqueeze(0)
            
            batch, seq, dim = hidden_states.shape
            num_heads = self.num_heads
            head_dim = dim // num_heads
            
            # 1. Reshape to separate heads BEFORE o_proj mixes them
            hidden_states = hidden_states.reshape(batch, seq, num_heads, head_dim).contiguous()
            
            # 2. Extract specific scalars for this layer
            start_idx = layer_idx * num_heads
            end_idx = (layer_idx + 1) * num_heads
            
            layer_scalars = self.steering_scalars[start_idx:end_idx].to(
                device=hidden_states.device, 
                dtype=hidden_states.dtype
            ).view(1, 1, num_heads, 1)
            
            # 3. Apply Intervention
            hidden_states = hidden_states * layer_scalars
            
            # 4. Flatten back to (Batch, Seq, Dim)
            new_input = hidden_states.reshape(original_shape)
            
            return (new_input,) + args[1:]
            
        return hook
    
    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def apply_steering(self, cluster_assignments=None, cluster_magnitudes=None, layer_magnitudes=None):
        
        # Reset scalars to baseline 1.0
        self.steering_scalars = torch.ones(self.total_heads, device=self.device)

        # 1. Apply Cluster Magnitudes
        if cluster_assignments is not None and cluster_magnitudes is not None:
            for head_idx, cluster_id in enumerate(cluster_assignments):
                if head_idx < self.total_heads:
                    self.steering_scalars[head_idx] = cluster_magnitudes.get(cluster_id, 1.0)

        # 2. Apply Layer Magnitudes (Multiplies on top of cluster steering)
        if layer_magnitudes is not None:
            for layer_idx, mag in layer_magnitudes.items():
                start_idx = layer_idx * self.num_heads
                end_idx = start_idx + self.num_heads
                # Multiply so layer and cluster weights combine
                self.steering_scalars[start_idx:end_idx] *= mag

        # Register PRE-hooks
        layer_count = 0
        for name, module in self.model.named_modules():
            if name.endswith("self_attn.o_proj"):
                if layer_count < self.num_layers:
                    handle = module.register_forward_pre_hook(self._get_steering_pre_hook(layer_count))
                    self.hooks.append(handle)
                    layer_count += 1
                    
    def inference(self, input_text, cluster_assignments=None, cluster_magnitudes=None, layer_magnitudes=None):
        self.apply_steering(cluster_assignments, cluster_magnitudes, layer_magnitudes)
        
        # --- 1. FORMAT & TOKENIZE ---
        messages = [
            {"role": "user", "content": input_text}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = self.model(**inputs, output_attentions=True)
            gen_tokens = self.model.generate(
                **inputs, 
                max_new_tokens=1028,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
            decoded_text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        
        # --- 2. EXTRACT RAW ATTENTION ---
        attentions = torch.stack(outputs.attentions).squeeze(1) 
        
        # --- 3. FLATTEN THE QUERIES ---
        P = attentions.mean(dim=-2) + 1e-9 
        
        # --- 4. CALCULATE STATISTICAL FEATURES ---
        # FEATURE 1: ENTROPY
        entropy = -(P * torch.log(P)).sum(dim=-1)
        
        # FEATURE 2: MAX WEIGHT
        max_weight = P.max(dim=-1).values
        
        # FEATURE 3: VARIANCE
        variance = P.var(dim=-1)
        
        # --- 5. REASSEMBLE FOR PCA ---
        head_features = torch.stack([entropy, max_weight, variance], dim=-1)
        
        flattened_features = head_features.view(self.total_heads, 3).to(torch.float32).cpu().numpy()
        
        split_output = decoded_text.split('assistant')
        text_output = split_output[len(split_output) - 1]
        
        return text_output, flattened_features
    
    
    