class StateManager:
    def __init__(self):
        self.history = []
        self.idx = -1

    def save(self, prompt, text_out, c_mags, l_mags, labels, emb, prev_emb, features):
        if self.idx < len(self.history) - 1:
            self.history = self.history[:self.idx + 1]
            
        self.history.append({
            'prompt': prompt, 'text_out': text_out, 
            'cluster_mags': c_mags.copy(), 'layer_mags': l_mags.copy(),
            'labels': labels.copy() if labels is not None else None,
            'emb': emb.copy() if emb is not None else None,
            'prev_emb': prev_emb.copy() if prev_emb is not None else None,
            'features': features.copy() if features is not None else None
        })
        self.idx += 1

    def get_current(self):
        if 0 <= self.idx < len(self.history):
            return self.history[self.idx]
        return None
    
    def reset(self):
        self.history = []
