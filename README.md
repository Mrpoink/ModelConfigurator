# Model Configurator

A real-time Mechanistic Interpretability and Activation Steering dashboard built for large language models. 

This dashboard allows researchers to peer inside the "brain" of a causal language model during forward inference, map the behavioral functions of its individual attention heads, and dynamically steer the model's output by artificially amplifying or ablating specific cognitive circuits in real-time.

---


## Core Features

### 1. Dual-Mode Activation Steering
Intercept and modify attention tensors *before* they are mixed by the output projection layer (`o_proj`) using PyTorch `register_forward_pre_hook` injection.
* **Cluster-Based Steering:** Control emergent behavioral circuits (e.g., "Helpfulness" or "Syntax Sniping") by scaling groups of heads that share similar routing strategies.
* **Layer-Based Steering:** Target the physical architecture directly by scaling the attention outputs of any of the specific hidden layers.

### 2. Sequence-Independent Latent Mapping (UMAP)
Projects all attention heads into a 2D interactive space to visualize how their behavior shifts depending on the prompt.
* **Statistical Feature Extraction:** Instead of using raw token positions (which crash dimensionality reduction when sequence lengths change), the pipeline extracts **Entropy, Maximum Attention Weight, and Variance**. This stabilizes the PCA/UMAP pipeline, allowing 5-token and 500-token prompts to be mapped on the exact same coordinate system.
* **Color Mapping:** Instantly toggle the scatter plot colors to map heads by their **Behavioral K-Means Cluster** or their **Architectural Layer**.

### 3. State Management & "Time Travel"
The dashboard maintains a complete history of your interventions and prompt states.
* **Current/Previous:** Use `< PREV STATE` and `NEXT STATE >` to instantly revert the UI sliders, text outputs, and graph visualizations to a previous experimental state.
* **Ghosting Reference:** Overlay trace lines on the UMAP graph to visualize exactly how the attention heads physically moved through the latent space from the pristine **Baseline** or the **Previous** state.

### 4. Live Attention Variance Heatmap
A dynamic `plotly` heatmap that tracks the actual neurological "spikiness" of the model. Instead of just displaying UI slider values, this plots the real-time calculated Variance of the attention distributions, scaled by your active interventions. Watch different layers light up as you steer the attention outputs or switch from coding tasks to creative writing.

---

## 🧠 The Math Under the Hood

### Feature Extraction for Stability
To completely decouple the mathematical pipeline from the physical length of the user's prompt, the system evaluates *routing strategy* rather than *positional values*:
1. **Entropy ($H$):** Measures the spread of attention. High entropy indicates broad context-gathering; low entropy indicates sharp, specific token sniping.
2. **Maximum Weight:** Captures the highest single attention spike to identify highly reactive trigger-word heads.
3. **Variance ($\sigma^2$):** Measures the deviation from the mean, distinguishing smooth distributions from rapidly oscillating ones.

### The Interception Point
The steering mechanism uses `register_forward_pre_hook` on the `self_attn.o_proj` module. When the hook catches the concatenated $(B, S, D_{hidden})$ tensor, it reshapes it to isolate the heads, performs element-wise multiplication with the custom slider magnitudes, and reassembles the tensor before allowing the model to continue its forward pass.

---

## 🛠️ Tech Stack
* **Backend Framework:** PyTorch, HuggingFace `transformers`
* **Frontend UI:** Plotly Dash (`dash`, `dcc`, `html`)
* **Dimensionality Reduction:** `scikit-learn` (PCA, K-Means), `umap-learn`
* **Data Manipulation:** NumPy

## ⚙️ Installation & Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Dashboard:**
   ```bash
   python App.py
   ```
   *Note: The model weights will be downloaded from HuggingFace on the first run. Ensure you have internet access.*
3. **Explore:** Open `http://127.0.0.1:8050/` in your browser. Enter a system query, hit Execute, and begin steering the latent space.
