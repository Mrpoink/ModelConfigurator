import dash
from dash import dcc, html, Input, Output, State, ALL
import plotly.graph_objects as go
import plotly.express as px
from ModelBackEnd.LoadModel import Model
from MapBackEnd.LoadMap import Map
from DashBoardBackend.StateManager import StateManager
import numpy as np

# --- 1. Init ---
num_clusters = 6
model_engine = Model()
visualizer = Map(num_clusters=num_clusters)
app = dash.Dash(__name__)

state_manager = StateManager()

# --- 2. LAYOUT ---
app.layout = html.Div([
    html.H1("Model Weight Dashboard", 
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

            # UI CONTROLS
            html.Div([
                html.Div([
                    html.Label("GHOSTING REFERENCE:", style={'color': '#888', 'fontSize': '12px', 'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='ghost-mode',
                        options=[{'label': ' Baseline', 'value': 'base'}, {'label': ' Previous', 'value': 'prev'}],
                        value='prev',
                        style={'color': '#00FF41', 'display': 'flex', 'gap': '15px', 'marginTop': '5px'},
                        inputStyle={'marginRight': '5px', 'cursor': 'pointer'}
                    )
                ], style={'flex': 1}),
                html.Div([
                    html.Label("COLOR MAP BY:", style={'color': '#888', 'fontSize': '12px', 'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='color-mode',
                        options=[{'label': ' Behaviors', 'value': 'cluster'}, {'label': ' Architecture', 'value': 'layer'}],
                        value='cluster',
                        style={'color': '#00FF41', 'display': 'flex', 'gap': '15px', 'marginTop': '5px'},
                        inputStyle={'marginRight': '5px', 'cursor': 'pointer'}
                    )
                ], style={'flex': 1})
            ], style={'display': 'flex', 'marginBottom': '20px', 'backgroundColor': '#111', 'padding': '10px', 'border': '1px solid #333'}),

            html.Label("MODEL RESPONSE:", style={'color': '#00FF41', 'fontSize': '12px'}),
            html.Div(id='output-text', style={
                'padding': '15px', 'border': '1px solid #333', 'backgroundColor': '#050505',
                'color': '#00FF41', 'fontFamily': 'Courier New', 'minHeight': '150px', 'fontSize': '14px', 'marginBottom': '20px'
            }),

            html.Div(id='slider-container')
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
     Input('viz-tabs', 'value'),
     Input('color-mode', 'value'),
     Input('ghost-mode', 'value')], 
    [State('prompt-input', 'value'),
     State({'type': 'cluster-slider', 'index': ALL}, 'value'),
     State({'type': 'cluster-slider', 'index': ALL}, 'id'),
     State({'type': 'layer-slider', 'index': ALL}, 'value'),
     State({'type': 'layer-slider', 'index': ALL}, 'id')]
)
def update_dashboard(run_clicks, reset_clicks, prev_clicks, next_clicks, active_tab, color_mode, ghost_mode,
                     prompt, c_values, c_ids, l_values, l_ids):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    c_mags = {i: 1.0 for i in range(num_clusters)}
    l_mags = {i: 1.0 for i in range(model_engine.num_layers)}

    curr_state = state_manager.get_current()
    if curr_state:
        c_mags = curr_state['cluster_mags'].copy()
        l_mags = curr_state['layer_mags'].copy()

    if c_values and c_ids:
        for val, id_dict in zip(c_values, c_ids):
            c_mags[id_dict['index']] = val
    if l_values and l_ids:
        for val, id_dict in zip(l_values, l_ids):
            l_mags[id_dict['index']] = val

    if trigger == 'reset-button':
        model_engine.clear_hooks()
        visualizer.labels = None
        visualizer.base_embedding = None
        state_manager.reset()
        return "System state cleared. All hooks detached.", html.Div("Awaiting execution..."), [], ""

    if trigger in ['prev-button', 'next-button', 'viz-tabs', 'color-mode', 'ghost-mode'] and state_manager.get_current() is not None:
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
        current_features = state['features']
        
    elif trigger == 'run-button' and prompt:
        if visualizer.labels is None: 
            model_engine.clear_hooks()
            _, init_weights = model_engine.inference(prompt)
            visualizer.setup(init_weights)

        current_labels = visualizer.labels
        
        text_out, weights = model_engine.inference(
            prompt, 
            cluster_assignments=current_labels, 
            cluster_magnitudes=c_mags, 
            layer_magnitudes=l_mags
        )
        visualizer.setup(weights)
        
        current_emb = visualizer.embedding
        prev_emb = visualizer.prev_embedding
        current_features = weights
        
        state_manager.save(prompt, text_out, c_mags, l_mags, current_labels, current_emb, prev_emb, current_features)
        
    else:
        return "System Idle. Awaiting Prompt.", html.Div("Awaiting execution..."), [], prompt

    # --- BUILD BOTH GRAPHS ---
    
    # 1. UMAP Figure
    umap_fig = go.Figure()
    
    # Safe fallback if prev_emb is None on the very first run
    ref = visualizer.base_embedding if ghost_mode == 'base' else (prev_emb if prev_emb is not None else current_emb)
    
    # Draw ghosting lines
    for idx in range(len(current_emb)):
        umap_fig.add_scatter(
            x=[ref[idx, 0], current_emb[idx, 0]], y=[ref[idx, 1], current_emb[idx, 1]],
            mode='lines', line=dict(color='rgba(0, 255, 65, 0.15)', width=1),
            showlegend=False, hoverinfo='skip'
        )

    h_labels = [f"L{j//model_engine.num_heads} H{j%model_engine.num_heads}" for j in range(len(current_emb))]
    
    if color_mode == 'cluster':
        for i in range(num_clusters):
            mask = current_labels == i
            umap_fig.add_scatter(
                x=current_emb[mask, 0], y=current_emb[mask, 1],
                mode='markers', name=f"Cluster {i}",
                text=[h_labels[j] for j, m in enumerate(mask) if m],
                hoverinfo='text', marker=dict(size=10, line=dict(width=1, color='#000'))
            )
    else:
        colors = px.colors.sample_colorscale("Turbo", [n/(model_engine.num_layers-1) for n in range(model_engine.num_layers)])
        for i in range(model_engine.num_layers):
            mask = np.array([j // model_engine.num_heads == i for j in range(len(current_emb))])
            umap_fig.add_scatter(
                x=current_emb[mask, 0], y=current_emb[mask, 1],
                mode='markers', name=f"Layer {i}",
                text=[h_labels[j] for j, m in enumerate(mask) if m],
                hoverinfo='text', marker=dict(size=10, color=colors[i], line=dict(width=1, color='#000'))
            )
        
    umap_fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=600,
        xaxis=dict(showgrid=True, gridcolor='#222', zeroline=True, zerolinecolor='#444'),
        yaxis=dict(showgrid=True, gridcolor='#222', zeroline=True, zerolinecolor='#444'),
        legend=dict(
            font=dict(color="white", size=13),
            bgcolor="rgba(20, 20, 20, 0.9)",
            bordercolor="#00FF41",
            borderwidth=1,
            itemsizing="constant"
        ),
        updatemenus=[
            dict(
                type="buttons", direction="left", pad={"r": 10, "t": 10}, showactive=True,
                x=1.0, xanchor="right", y=1.1, yanchor="bottom", bgcolor="#222", font=dict(color="#00FF41", size=11),
                buttons=[
                    dict(label="Hide All", method="restyle", args=[{"visible": "legendonly"}]),
                    dict(label="Show All", method="restyle", args=[{"visible": True}])
                ]
            )
        ]
    )

    # 2. Heatmap Figure
    heat_data = np.zeros((model_engine.num_layers, num_clusters))
    counts = np.zeros((model_engine.num_layers, num_clusters))
    
    for head_idx, cluster_id in enumerate(current_labels):
        layer = head_idx // model_engine.num_heads
        raw_variance = current_features[head_idx, 2] 
        steered_activity = raw_variance * c_mags.get(cluster_id, 1.0) * l_mags.get(layer, 1.0)
        
        heat_data[layer, cluster_id] += steered_activity
        counts[layer, cluster_id] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        heat_data = np.true_divide(heat_data, counts)
        heat_data[np.isnan(heat_data)] = 0
    
    heat_fig = px.imshow(
        heat_data.T, 
        labels=dict(x="Model Layer", y="Intervention Cluster", color="Attention Variance"),
        x=[f"Layer {i}" for i in range(model_engine.num_layers)],
        y=[f"Cluster {i}" for i in range(num_clusters)],
        color_continuous_scale='Magma'
    )
    heat_fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=600)

    # Toggle Visibility based on Tab
    umap_style = {'display': 'block'} if active_tab == 'umap-tab' else {'display': 'none'}
    heat_style = {'display': 'block'} if active_tab == 'heat-tab' else {'display': 'none'}

    viz_output = html.Div([
        dcc.Graph(figure=umap_fig, style=umap_style),
        dcc.Graph(figure=heat_fig, style=heat_style)
    ])

    # --- BUILD BOTH SLIDER SETS ---
    cluster_ui = [html.Div([
        html.Label(f"CLUSTER {i} INTENSITY", style={'fontSize': '11px', 'color': '#00FF41'}),
        dcc.Slider(id={'type': 'cluster-slider', 'index': i}, min=0, max=2, step=0.1, value=c_mags.get(i, 1.0), marks={0: '0', 1: '1', 2: '2'})
    ], style={'padding': '10px', 'backgroundColor': '#111', 'marginBottom': '5px', 'borderLeft': '3px solid #00FF41'}) for i in range(num_clusters)]

    layer_ui = [html.Div([
        html.Label(f"LAYER {i} INTENSITY", style={'fontSize': '11px', 'color': '#FF00FF'}),
        dcc.Slider(id={'type': 'layer-slider', 'index': i}, min=0, max=2, step=0.1, value=l_mags.get(i, 1.0), marks={0: '0', 1: '1', 2: '2'})
    ], style={'padding': '10px', 'backgroundColor': '#111', 'marginBottom': '5px', 'borderLeft': '3px solid #FF00FF'}) for i in range(model_engine.num_layers)]

    c_container_style = {'display': 'block'} if active_tab == 'umap-tab' else {'display': 'none'}
    l_container_style = {'display': 'block' if active_tab == 'heat-tab' else 'none', 'maxHeight': '500px', 'overflowY': 'auto', 'paddingRight': '10px'}

    sliders_output = html.Div([
        html.Div(cluster_ui, style=c_container_style),
        html.Div(layer_ui, style=l_container_style)
    ])

    return text_out, viz_output, sliders_output, prompt

if __name__ == '__main__':
    app.run(debug=True, port=8050)