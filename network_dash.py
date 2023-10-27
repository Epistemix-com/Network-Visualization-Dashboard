import itertools
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import plotly.tools as tls
import pandas as pd
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
from ipywidgets import interact
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from callbacks import update_graph, update_influential_graph, update_within_graph

def read_csv(filename):
    if filename:
        return(pd.read_csv(filename))
    if not filename:
        print("Something went wrong with feeding in a new data set, you can modify the script manually at line 27 with the new dataset name")

# Create your Dash app instance
app = Dash(__name__)

background_color = '#252e59'
text_color = '#6187eb'

flu = read_csv(sys.argv[1])

bins = list(range(0, 201, 10))  # Creates bins from 0 to 200 with a step of 10
labels = ['{}-{}'.format(i, i+9) for i in range(0, 200, 10)]  # Creates labels for these bins

flu['age_group'] = pd.cut(flu['age'], bins=bins, labels=labels, right=False)

quantiles = flu['income'].quantile(q=np.linspace(0, 1, 11))
bins = [quantiles.iloc[i] for i in range(11)]
labels = ['{}-{}'.format(int(bins[i]), int(bins[i+1])) for i in range(10)]

# Create 'income_group' column
flu['income_group'] = pd.cut(flu['income'], bins=bins, labels=labels, right=False, include_lowest=True)


G = nx.from_pandas_edgelist(flu[['source', 'id']], source='source', target='id', create_using=nx.DiGraph)
for _, row in flu.iterrows():
    node_id = row['id']
    # Check if the node exists in the graph before adding attributes
    if node_id in G.nodes:
        G.nodes[node_id]['race'] = row['race_string']
        G.nodes[node_id]['age'] = row['age']
        G.nodes[node_id]['income'] = row['income']
        G.nodes[node_id]['expdate'] = datetime.strptime(str(int(row['expdate'])), '%Y%m%d').strftime('%m/%d/%Y')
        G.nodes[node_id]['exp_place'] = row['exp_place']
        
G.nodes[0]['race'] = 0
G.nodes[0]['age'] = 0
G.nodes[0]['income'] = 0
G.nodes[0]['expdate'] = datetime.strptime(str(int(min(flu['expdate']))), '%Y%m%d').strftime('%m/%d/%Y')
G.nodes[0]['exp_place'] = 'Other'


G2 = nx.DiGraph()
G3 = nx.Graph()
G_di_race = nx.DiGraph()
G_race = nx.Graph()
G_di_age = nx.DiGraph()
G_age = nx.Graph()
# G_di_sex = nx.DiGraph()
# G_sex = nx.Graph()
G_di_income = nx.DiGraph()
G_income = nx.Graph()

def expand(*args):
    return list(itertools.product(*args))

def add_edges(graph, edges):
    for edge in edges:
        if graph.has_edge(*edge):
            # increment weight by 1 if the edge already exists
            graph[edge[0]][edge[1]]['weight'] += 1
        else:
            # create a new edge with weight 1
            graph.add_edge(edge[0], edge[1], weight=1)
    return graph

for s in set(flu['source']):
    from_exp_places = flu[flu['id'] == s]['exp_place']
    to_exp_place = flu[flu['source'] == s]['exp_place']
    from_races = flu[flu['id'] == s]['race_string']
    to_race = flu[flu['source'] == s]['race_string']
    from_ages = flu[flu['id'] == s]['age_group']
    to_age = flu[flu['source'] == s]['age_group']    
    # from_sex = flu[flu['id'] == s]['sex']
    # to_sex = flu[flu['source'] == s]['sex']
    from_income = flu[flu['id'] == s]['income_group']
    to_income = flu[flu['source'] == s]['income_group']
    # add edges to the graph
    G2 = add_edges(G2, expand(from_exp_places, to_exp_place))
    G3 = add_edges(G3, expand(from_exp_places, to_exp_place))
    G_di_race = add_edges(G_di_race, expand(from_races, to_race))
    G_race = add_edges(G_race, expand(from_races, to_race))    
    G_di_age = add_edges(G_di_age, expand(from_ages, to_age))
    G_age = add_edges(G_age, expand(from_ages, to_age))
    # G_di_sex = add_edges(G_di_sex, expand(from_sex, to_sex))
    # G_sex = add_edges(G_sex, expand(from_sex, to_sex))
    G_di_income = add_edges(G_di_income, expand(from_income, to_income))
    G_income = add_edges(G_income, expand(from_income, to_income))  

descendants_counts = [len(nx.descendants(G, node)) for node in G.nodes()]
df_descendants = pd.DataFrame({'agent_id': [node for node in G.nodes()], 'descendants': descendants_counts})
df_descendants['log_descendants'] = np.log(df_descendants['descendants'] + 1) # + 1 smoothing for log
top_100_nodes = df_descendants.nlargest(100, 'descendants')

descendants_counts_gen1 = [len(nx.descendants_at_distance(G, node, 1)) for node in G.nodes()]
df_descendants_gen1 = pd.DataFrame({'agent_id': [node for node in G.nodes()], 'descendants': descendants_counts_gen1})
df_descendants_gen1['log_descendants'] = np.log(df_descendants_gen1['descendants'] + 1) # + 1 smoothing for log
top_100_nodes = df_descendants_gen1.nlargest(100, 'descendants')


ancestors_counts = [len(nx.ancestors(G, node)) for node in G.nodes()]
df_ancestors = pd.DataFrame(ancestors_counts, columns=['ancestors'])
df_ancestors['log_ancestors'] = np.log(df_ancestors['ancestors'] + 1)

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
df_degree = pd.DataFrame(degree_sequence, columns=['degree'])
df_degree['log_degree'] = np.log(df_degree['degree']  + 1)

# Assuming `df` is your DataFrame and num_cols is your list of column names
scaler = StandardScaler()
flu_norm = flu.copy()
num_cols = ['age', 'income']
flu_norm[num_cols] = scaler.fit_transform(flu_norm[num_cols])

age_range = (min(flu['age']), max(flu['age']))
race_strings = list(set(flu['race_string']))
incomes = (min(flu['income']), max(flu['income']))
latitudes = (min(flu['house_latitude']), max(flu['house_latitude']))
longitudes = (min(flu['house_longitude']), max(flu['house_longitude']))
exp_places = list(set(flu['exp_place']))
layouts=['Multipartite', 'Planar', 'Shell', 'Spiral', "Kamada-Kawai", 'Random']
by_dates = ['Plot by Exposure Date', 'No Date']
workplaces = list(set(flu[flu['exp_place'] == "Workplace"]['exp_location']))
schools = list(set(flu[flu['exp_place'] == "School"]['exp_location']))

background_color = '#252e59'
text_color = '#6187eb'
# Define your app layout and components
app.layout = html.Div([
    dcc.Markdown('# FRED Network Dashboard',style={'color': text_color, 'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Egocentric Network Visualization', children=[
            html.Div([
                # Top Half - Controllers in Two Columns
                html.Div(children=[
                    html.Div([
                        html.Br(),
                        html.Label("Race", style={'color': text_color}),
                        dcc.Dropdown(race_strings, 'White', id='race-selection'),
                        
                        html.Br(),
                        html.Label("Exposure Place", style={'color': text_color}),
                        dcc.Dropdown(exp_places, 'Household', id='exp-place-selection'),
                        
                        html.Br(),
                        html.Label("Network Layout Options", style={'color': text_color}),
                        dcc.Dropdown(by_dates, 'No Date', id='by-date-selection'),
                        dcc.Dropdown(layouts, 'Multipartite', id='layout-selection'),
                    ], style={'flex': 1, 'padding': 10}),
                    html.Div([
                        html.Br(),
                        html.Label("Age", style={'color': text_color}),
                        dcc.Slider(min=age_range[0], max=age_range[1], value=age_range[0], id="age-slider"),
                        
                        html.Br(),
                        html.Label("Income", style={'color': text_color}),
                        dcc.Slider(id="income-slider", min=incomes[0], max=incomes[1], value=incomes[0]),
                        html.Br(),
                        
                        html.Label("Use Specific Ego? (Click on Node to copy ID)", style={'color': text_color}),
                        dcc.RadioItems(
                            id='specific-ego-check',
                            options=[
                                {'label': 'Yes', 'value': True},
                                {'label': 'No', 'value': False}
                            ],
                            value=False),
                        dcc.Input(id='specific-ego-id', type='text'),  # Add this line
                        html.Br(),
                        ], style={'flex': 1}),
                ], style={'display': 'flex', 'flex-direction': 'row'}),
                
                # Bottom Half - Graph
                html.Div([
                    html.Label("Radius", style={'color': text_color}),
                    dcc.Slider(id="radius-slider", min=1, max=10, step=1, value=1),
                    dcc.Graph(id='graph-content'),
                    html.Div(id='dummy-output')
                    # dcc.Clipboard(id='my-clipboard')
                ], style={'flex': 1, 'padding': 10}),
            ], style={'display': 'flex', 'flex-direction': 'column'})
                    ]),
        dcc.Tab(label='Network Summary Plots', children=[          
            # Add your bar chart components here
            html.Div([
                html.Label("Log-log", style={'color': text_color}),
                html.Div([
                    dcc.RadioItems(
                        id='log-scale',
                        options=[
                            {'label': 'True', 'value': True},
                            {'label': 'False', 'value': False}
                        ],
                        value=True,
                        style={'display': 'inline-block', 'margin-right': '30px'}
                    ),
                ], style={'margin-bottom': '20px'}),
                html.Label("Top 100 Ancestors", style={'color': text_color}),

                html.Div([
                    dcc.RadioItems(
                        id='top-100',
                        options=[
                            {'label': 'True', 'value': True},
                            {'label': 'False', 'value': False}
                        ],
                        value=False,
                        style={'display': 'inline-block'}
                    ),
                ], style={'margin-bottom': '20px'}),
            ]),

            dcc.Graph(id='graph-stat1'),
            dcc.Graph(id='graph-stat2'),
            dcc.Graph(id='graph-stat3'),
        ]),        
        dcc.Tab(label='Influential Agents', children=[
            html.Button("Regenerate Graph", id = 'regen-button', n_clicks=0),
            html.Label("Network Layout Options", style={'color': text_color}),
            dcc.Dropdown(by_dates, 'No Date', id='top-by-date-selection'),
            dcc.Dropdown(layouts, 'Multipartite', id='top-layout-selection'),
            html.Label("Radius", style={'color': text_color}),
            dcc.Slider(id="top-radius-slider", min=1, max=10, step=1, value=1),
            dcc.Graph(id='influential-graph'),
        ]),        
        dcc.Tab(label='Group-level Transmission Network', children=[
            html.Div([
                html.Div([
                    html.Label('Edge Type', style={'color': text_color}),
                    dcc.RadioItems(
                        id='directed-button',
                        options=[
                            {'label': 'Directed', 'value': True},
                            {'label': 'Undirected', 'value': False}
                        ],
                        value=True
                    ),            
                    html.Label('Transition Probability or Weight (Count) Matrix', style={'color': text_color}),
                    dcc.RadioItems(
                        id='prob-count-button',
                        options=[
                            {'label': 'Transition Probability', 'value': True},
                            {'label': 'Counts', 'value': False}
                        ],
                        value=True
                    ),                   
                    html.Label('Group, Age, or Race Transitions', style={'color': text_color}),
                    dcc.RadioItems(
                        id='aggregate-type-button',
                        options=[
                            {'label': 'Mixing Group', 'value': 'group'},
                            {'label': 'Age', 'value': 'age'},
                            {'label': 'Race', 'value': 'race'},
                            # {'label': 'Sex', 'value': 'sex'},
                            {'label': 'Income', 'value': 'income'}
                        ],
                        value='group'
                    ),
                    html.Div([
                        dcc.Graph(id='group-transition-heatmap'),
                    ])
                ], className="six columns"),

                html.Div([
                    dcc.Graph(id='group-transition-network'),
                ], className="six columns"),
            ], className="row")]),
        dcc.Tab(label='Within-Place Transmissions', children=[
            html.Label("Network Layout Options", style={'color': text_color}),
            dcc.RadioItems(
                id='work-school-button',
                options=[
                    {'label': 'School', 'value': False},
                    {'label': 'Work', 'value': True}
                ],
                value=True
            ),
            dcc.Dropdown(workplaces, 5142694400000, id='within-workplace-selection'),
            dcc.Dropdown(schools, 4501391770000, id='within-school-selection'),
            dcc.Dropdown(by_dates, 'No Date', id='within-by-date-selection'),
            dcc.Dropdown(layouts, 'Multipartite', id='within-layout-selection'),
            dcc.Graph(id='within-place-transmission'),
            ]),
    ]),
])

@app.callback(
    Output('graph-content', 'figure'),
    Input('race-selection', 'value'),
    Input('age-slider', 'value'),
    Input('income-slider', 'value'),
    Input('exp-place-selection', 'value'),
    Input('specific-ego-check', 'value'),
    Input('specific-ego-id', 'value'),
    Input('by-date-selection', 'value'),
    Input('layout-selection', 'value'),
    Input('radius-slider', 'value')
)

def update_graph_wrapper(race,age,income,exp_place,ego_check,specific_ego_id,by_date,layout,radius):
    return update_graph(flu_norm,G,scaler,race,age,income,exp_place,ego_check,specific_ego_id,by_date,layout,radius)

# @app.callback(
#     Output('my-clipboard', 'content'),
#     Input('graph-content', 'clickData'),
#     prevent_initial_call=True
# )

# def update_clipboard(clickData):
#     if clickData is not None:
#         # Extract the node label you want to copy
#         node_label = clickData['points'][0]['text']
        
#         # Extract the node number from the node label
#         node_number = node_label.split('Node: ')[1].split('<br>')[0]
        
#         return node_number
app.clientside_callback(
    """
    function(clickData) {
        if(clickData !== undefined) {
            var node_label = clickData.points[0].text;
            var node_number = node_label.split('Node: ')[1].split('<br>')[0];
            return node_number;
        }
    }
    """,
    Output('specific-ego-id', 'value'),  # Change this line
    Input('graph-content', 'clickData')
)


@app.callback(
    Output('graph-stat1', 'figure'),
    Output('graph-stat2', 'figure'),
    Output('graph-stat3', 'figure'),
    Input('log-scale', 'value'),
    Input('top-100', 'value')
)

def update_graph_stats(log_scale, top_100):
    if log_scale:
        if top_100:
            fig1 = px.histogram(df_descendants[df_descendants['agent_id'].isin(top_100_nodes['agent_id'])], 
                                x='log_descendants', log_y=True, labels={"x": "Log(Descendants)", 'y': 'Count'})
        else:
            fig1 = px.histogram(df_descendants, x='log_descendants', log_y=True, labels={"x": "Log(Descendants)", 'y': 'Count'})
        fig2 = px.histogram(df_ancestors, x='log_ancestors', log_y=True, labels={'x': "Log(Ancestors)", 'y': 'Count'})
        fig3 = px.histogram(df_degree, x='log_degree', log_y=True, labels={'x': "Log(Degree)", 'y': 'Count'})
    else:
        if top_100:
            fig1 = px.histogram(df_descendants[df_descendants['agent_id'].isin(top_100_nodes['agent_id'])], 
                                x='descendants', labels={"x": "Descendants", 'y': 'Count'})
        else:
            fig1 = px.histogram(df_descendants, x='descendants', log_y=True, labels={"x": "Descendants", 'y': 'Count'})
        fig2 = px.histogram(df_ancestors, x='ancestors', labels={'x': "Ancestors", 'y': 'Count'})
        fig3 = px.histogram(df_degree, x='degree', labels={'x': "Degree", 'y': 'Count'})
        
    return fig1, fig2, fig3

@app.callback(
    Output('influential-graph', 'figure'),
    Input('top-by-date-selection', 'value'),
    Input('top-layout-selection', 'value'),
    Input('top-radius-slider', 'value'),
    Input('regen-button', 'n_clicks')
)

def update_influential_graph_wrapper(by_date, layout, radius, n_clicks, top_100=top_100_nodes, G=G):
    return update_influential_graph(top_100, G, by_date, layout, radius)


@app.callback(
    Output('group-transition-heatmap', 'figure'),
    Output('group-transition-network', 'figure'),
    Input('directed-button', 'value'),
    Input('prob-count-button', 'value'),
    Input('aggregate-type-button', 'value')
)

def plot_group_transitions(directed, prob, aggregate_type):
    #TODO: fix the transition probability for the external group and the undirected transition probability matrix
    if aggregate_type == 'group':
        if directed:
            tmp_G = G2
        else:
            tmp_G = G3
    elif aggregate_type == 'age':
        if directed:
            tmp_G = G_di_age
        else:
            tmp_G = G_age
    elif aggregate_type == 'race':
        if directed:
            tmp_G = G_di_race
        else:
            tmp_G = G_race
    # elif aggregate_type == 'sex':
    #     if directed:
    #         tmp_G = G_di_sex
    #     else:
    #         tmp_G = G_sex
    elif aggregate_type == 'income':
        if directed:
            tmp_G = G_di_income
        else:
            tmp_G = G_income

    # Calculate the sum of edge weights for each node
    weight_sum = {node: sum(weight for _, _, weight in tmp_G.edges(node, data='weight')) for node in tmp_G.nodes()}

    # Get the adjacency matrix as a DataFrame with (transition) 'probability' or 'weight' (count) as the edge attribute
    if prob:
        # Calculate transition probabilities
        for u, _, data in tmp_G.edges(data=True):
            if weight_sum[u] == 0:
                data['probability'] = 0
            else:
                data['probability'] = data['weight'] / weight_sum[u]
        df = nx.to_pandas_adjacency(tmp_G, weight='probability')
    else:
        df = nx.to_pandas_adjacency(tmp_G, weight='weight')

    # Sort the DataFrame by index and columns
    df = df.sort_index().sort_index(axis=1)

    fig2 = px.imshow(df)
    
    # Get positions for the nodes in G
    pos = nx.shell_layout(tmp_G)

    # Create edge traces
    edge_traces = []
    weights = nx.get_edge_attributes(tmp_G, 'weight')
    edge_colors = [weights.get(edge) for edge in tmp_G.edges()]
    
    # Create a color mapper
    norm = colors.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
    mapper = cm.ScalarMappable(norm=norm, cmap='viridis')

    for i, edge in enumerate(tmp_G.edges()):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = weights.get(edge)
        color = colors.rgb2hex(mapper.to_rgba(weight))
        edge_trace = go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=1, color=color),
            hoverinfo='none',
            mode='lines')
        edge_traces.append(edge_trace)

    # Create node traces
    node_x = []
    node_y = []
    for node in tmp_G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='rgba(255, 0, 0, 0.5)',
            size=10,
            line_width=2))

    # Create a Plotly figure and add the traces
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title='<br>Network graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig2, fig

@app.callback(
    Output('within-place-transmission', 'figure'),
    Input('within-layout-selection', 'value'),
    Input('within-by-date-selection', 'value'),
    Input('work-school-button', 'value'),
    Input('within-workplace-selection', 'value'),
    Input('within-school-selection', 'value'),
)

def update_within_graph_wrapper(layout, by_date, work_or_school, workplace, school, G=G):
    return update_within_graph(layout, by_date, work_or_school, workplace, school, G, flu)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
