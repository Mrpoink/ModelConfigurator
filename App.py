import dash
from dash import dcc, html, Input, Output, State, ALL
import plotly.graph_objects as go
import plotly.express as px
from ModelBackEnd.LoadModel import Model
from MapBackEnd.LoadMap import Map
from DashBoardBackend.StateManager import StateManager
import numpy as np

# --- 1. Init & State Management ---
num_clusters = 6
model_engine = Model()
visualizer = Map(num_clusters=num_clusters)
app = dash.Dash(__name__)

state_manager = StateManager()

# --- 2. THE "RESEARCH TERMINAL" LAYOUT ---
app.layout = html.Div([
    html.H1("SmolLM LATENT STEERING TERMINAL", 
            style={'textAlign': 'center', 'color': '#00FF41', 'fontFamily': 'Courier New', 'padding': '20px'}),
    
    html.Div([
        # LEFT COLUMN: System Inputs
        html.Div([
            html.Div([
                dcc.Input(id='prompt-input', type='text', placeholder='Enter System Query...', 
                          style={'width': '100%', 'padding': '5px', 'backgroundColor': '#000', 'color': '#00FF41', 'border': '1px solid #00FF41'}),
                html.Button('EXECUTE', id='run-button', n_clicks=0, 
                            style={'backgroundColor': '#00FF41', 'color': 'black', 'fontWeight': 'bold', 'padding': '12px', 'border': 'none', 'cursor': 'pointer'}),
                html.Button('RESET', id='reset-button', n_clicks=0, 
                            style={'backgroundColor': '#441111', 'color': 'white', 'marginLeft': '5px', 'border': 'none', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'marginBottom': '10px'}),
            
            html.Div([
                html.Button('< PREV STATE', id='prev-button', n_clicks=0, 
                            style={'backgroundColor': '#222', 'color': '#00FF41', 'border': '1px solid #00FF41', 'cursor': 'pointer', 'flex': 1}),
                html.Button('NEXT STATE >', id='next-button', n_clicks=0, 
                            style={'backgroundColor': '#222', 'color': '#00FF41', 'border': '1px solid #00FF41', 'cursor': 'pointer', 'flex': 1, 'marginLeft': '5px'}),
            ], style={'display': 'flex', 'marginBottom': '20px'}),

            html.Label("GHOSTING REFERENCE:", style={'color': '#888', 'fontSize': '12px'}),
            dcc.RadioItems(
                id='ghost-mode',
                options=[{'label': ' Baseline (Pure Model)', 'value': 'base'}, {'label': ' Previous State', 'value': 'prev'}],
                value='prev',
                style={'color': 'white', 'marginBottom': '20px', 'display': 'flex', 'gap': '20px'}
            ),

            html.Label("MODEL RESPONSE:", style={'color': '#00FF41', 'fontSize': '12px'}),
            html.Div(id='output-text', style={
                'padding': '15px', 'border': '1px solid #333', 'backgroundColor': '#050505',
                'color': '#00FF41', 'fontFamily': 'Courier New', 'minHeight': '150px', 'fontSize': '14px', 'marginBottom': '20px'
            }),

            # Dynamic Slider Container (Will scroll for layers)
            html.Div(id='slider-container', style={'maxHeight': '500px', 'overflowY': 'auto', 'paddingRight': '10px'})
        ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

        # RIGHT COLUMN: Visual Analytics
        html.Div([
            dcc.Tabs(id='viz-tabs', value='umap-tab', children=[
                dcc.Tab(label='LATENT SPACE (UMAP)', value='umap-tab', 
                        style={'backgroundColor': '#111', 'color': '#888'}, selected_style={'backgroundColor': '#00FF41', 'color': 'black'}),
                dcc.Tab(label='LAYER INTENSITY (HEATMAP)', value='heat-tab', 
                        style={'backgroundColor': '#111', 'color': '#888'}, selected_style={'backgroundColor': '#00FF41', 'color': 'black'}),
            ]),
            html.Div(id='tabs-content')
        ], style={'width': '60%', 'display': 'inline-block', 'padding': '20px'})
    ], style={'display': 'flex'})
], style={'backgroundColor': '#0a0a0a', 'minHeight': '100vh', 'fontFamily': 'sans-serif'})

# --- 3. CALLBACKS ---
@app.callback(
    [Output('output-text', 'children'),
     Output('tabs-content', 'children'),
     Output('slider-container', 'children'),
     Output('prompt-input', 'value')],
    [Input('run-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks'),
     Input('viz-tabs', 'value')],
    [State('prompt-input', 'value'),
     State({'type': 'cluster-slider', 'index': ALL}, 'value'),
     State({'type': 'cluster-slider', 'index': ALL}, 'id'),
     State({'type': 'layer-slider', 'index': ALL}, 'value'),
     State({'type': 'layer-slider', 'index': ALL}, 'id'),
     State('ghost-mode', 'value')]
)
def update_dashboard(run_clicks, reset_clicks, prev_clicks, next_clicks, active_tab, 
                     prompt, c_values, c_ids, l_values, l_ids, ghost_mode):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Default magnitude baselines
    c_mags = {i: 1.0 for i in range(num_clusters)}
    l_mags = {i: 1.0 for i in range(model_engine.num_layers)}

    # Pull latest from StateManager if exists
    curr_state = state_manager.get_current()
    if curr_state:
        c_mags = curr_state['cluster_mags'].copy()
        l_mags = curr_state['layer_mags'].copy()

    # Update with UI values if the user just moved a slider
    if c_values and c_ids:
        for val, id_dict in zip(c_values, c_ids):
            c_mags[id_dict['index']] = val
    if l_values and l_ids:
        for val, id_dict in zip(l_values, l_ids):
            l_mags[id_dict['index']] = val

    # 1. HANDLE SYSTEM RESET
    if trigger == 'reset-button':
        model_engine.clear_hooks()
        visualizer.labels = None
        visualizer.base_embedding = None
        state_manager.reset()
        return "System state cleared. All hooks detached.", html.Div("Awaiting execution..."), [], ""

    # 2. HANDLE HISTORY NAVIGATION
    if trigger in ['prev-button', 'next-button', 'viz-tabs'] and state_manager.get_current() is not None:
        if trigger == 'prev-button':
            state_manager.idx = max(0, state_manager.idx - 1)
        elif trigger == 'next-button':
            state_manager.idx = min(len(state_manager.history) - 1, state_manager.idx + 1)
            
        state = state_manager.get_current()
        prompt = state['prompt']
        text_out = state['text_out']
        c_mags = state['cluster_mags']
        l_mags = state['layer_mags']
        current_labels = state['labels']
        current_emb = state['emb']
        prev_emb = state['prev_emb']
        
    # 3. HANDLE NEW EXECUTION
    elif trigger == 'run-button' and prompt:
        if visualizer.labels is None: 
            model_engine.clear_hooks()
            _, init_weights = model_engine.inference(prompt)
            visualizer.setup(init_weights)

        current_labels = visualizer.labels
        
        # Pass BOTH cluster and layer magnitudes to the updated inference engine
        text_out, weights = model_engine.inference(
            prompt, 
            cluster_assignments=current_labels, 
            cluster_magnitudes=c_mags, 
            layer_magnitudes=l_mags
        )
        visualizer.setup(weights)
        
        current_emb = visualizer.embedding
        prev_emb = visualizer.prev_embedding
        
        # Save to history
        state_manager.save(prompt, text_out, c_mags, l_mags, current_labels, current_emb, prev_emb)
        
    else:
        return "System Idle. Awaiting Prompt.", html.Div("Awaiting execution..."), [], prompt

    # 4. BUILD VISUALS BASED ON ACTIVE TAB
    if active_tab == 'umap-tab':
        fig = go.Figure()
        ref = visualizer.base_embedding if ghost_mode == 'base' else prev_emb
        
        for idx in range(len(current_emb)):
            fig.add_scatter(
                x=[ref[idx, 0], current_emb[idx, 0]], y=[ref[idx, 1], current_emb[idx, 1]],
                mode='lines', line=dict(color='rgba(0, 255, 65, 0.15)', width=1),
                showlegend=False, hoverinfo='skip'
            )

        h_labels = [f"L{j//model_engine.num_heads} H{j%model_engine.num_heads}" for j in range(len(current_emb))]
        for i in range(num_clusters):
            mask = current_labels == i
            fig.add_scatter(
                x=current_emb[mask, 0], y=current_emb[mask, 1],
                mode='markers', name=f"Cluster {i}",
                text=[h_labels[j] for j, m in enumerate(mask) if m],
                hoverinfo='text', marker=dict(size=10, line=dict(width=1, color='#000'))
            )
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=600)
        viz_output = dcc.Graph(figure=fig)

        # Generate CLUSTER Sliders
        sliders = [html.Div([
            html.Label(f"CLUSTER {i} INTENSITY", style={'fontSize': '11px', 'color': '#00FF41'}),
            dcc.Slider(id={'type': 'cluster-slider', 'index': i}, min=0, max=2, step=0.1, value=c_mags.get(i, 1.0),
                       marks={0: '0', 1: '1', 2: '2'})
        ], style={'padding': '10px', 'backgroundColor': '#111', 'marginBottom': '5px', 'borderLeft': '3px solid #00FF41'}) for i in range(num_clusters)]

    else:
        # Layer Heatmap
        heat_data = np.zeros((model_engine.num_layers, num_clusters))
        for head_idx, cluster_id in enumerate(current_labels):
            layer = head_idx // model_engine.num_heads
            # Show the combined effect of cluster * layer magnitude
            heat_data[layer, cluster_id] += (c_mags.get(cluster_id, 1.0) * l_mags.get(layer, 1.0))
        
        fig = px.imshow(heat_data.T, 
                        labels=dict(x="Model Layer", y="Intervention Cluster", color="Magnitude"),
                        x=[f"Layer {i}" for i in range(model_engine.num_layers)],
                        y=[f"Cluster {i}" for i in range(num_clusters)],
                        color_continuous_scale='Viridis')
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=600)
        viz_output = dcc.Graph(figure=fig)

        # Generate LAYER Sliders (Scrollable)
        sliders = [html.Div([
            html.Label(f"LAYER {i} INTENSITY", style={'fontSize': '11px', 'color': '#FF00FF'}),
            dcc.Slider(id={'type': 'layer-slider', 'index': i}, min=0, max=2, step=0.1, value=l_mags.get(i, 1.0),
                       marks={0: '0', 1: '1', 2: '2'})
        ], style={'padding': '10px', 'backgroundColor': '#111', 'marginBottom': '5px', 'borderLeft': '3px solid #FF00FF'}) for i in range(model_engine.num_layers)]

    return text_out, viz_output, sliders, prompt

if __name__ == '__main__':
    app.run(debug=True, port=8050)