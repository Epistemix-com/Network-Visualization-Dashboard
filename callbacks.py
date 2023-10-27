from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import random
from ipywidgets import interact
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import networkx as nx
import plotly.graph_objects as go
from dash import Input, Output
from mini_dash import app  # Import your Dash app instance
import plotly.express as px  # Import Plotly Express if needed

def update_graph(df, G, scale, race,age,income,exp_place,ego_check,specific_ego_id,by_date,layout,radius):
        # Only call find_closest_match if radius has not changed
    if ego_check:
        ego = int(specific_ego_id)
    else:
        def find_closest_match(df, race_string, age, income, exp_place,scale):
            # Create a DataFrame for the inputs
            input_data = pd.DataFrame({'age': [age], 'income': [income]})
            
            filt = df[(df['race_string'] == race_string) & (df['exp_place'] == exp_place)].copy()
            if filt.shape[0] == 0:
                print(f"There is no data matching race = {race_string} and exp_place = {exp_place}, using random agent")
                return(df.loc[0])
            num_cols = ['age', 'income']

            input_data[num_cols] = scale.transform(input_data[num_cols])
            # Calculate absolute difference for each column
            for col in num_cols:
                filt[col] = abs(filt[col] - input_data.loc[0, col])

            # Create 'distance' column as sum of differences
            filt['distance'] = filt[num_cols].sum(axis=1)

            # Return row with lowest 'distance'
            min_distance_row = filt[filt['distance'] == filt['distance'].min()].copy()
            return(min_distance_row)

        closest_match = find_closest_match(df, race_string=race, age=age, income=income, exp_place=exp_place, scale = scale)
    
        ego = closest_match['id'].values[0]
    
    rad = radius

    # G.remove_node(0)

    # subgraph = nx.ego_graph(G, ego, radius=rad, center=True, undirected=True)
    # Find all nodes that have a direct path to or from the ego node
    # nodes_with_direct_path = [node for node in G.nodes if nx.has_path(G, node, ego) or nx.has_path(G, ego, node)]

    # Find all nodes that have a direct path to or from the ego node within a particular radius
    nodes_with_direct_path = [node for node in G.nodes if (nx.has_path(G, node, ego) and nx.shortest_path_length(G, node, ego) <= rad) or (nx.has_path(G, ego, node) and nx.shortest_path_length(G, ego, node) <= rad)]

    # Create a subgraph that only includes nodes with a direct path to or from the ego node
    subgraph = nx.subgraph(G, nodes_with_direct_path)


    # Compute the multipartite_layout using the "layer" node attribute
    if layout == 'Multipartite':
        for layer, nodes in enumerate(nx.topological_generations(subgraph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
            for node in nodes:
                subgraph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(subgraph, subset_key="layer")
    elif layout == 'Planar':
        pos = nx.planar_layout(subgraph)
    elif layout == 'Shell':
        nlist = [list(nodes) for nodes in nx.topological_generations(subgraph)]
        pos = nx.shell_layout(subgraph, nlist = nlist)  
    elif layout == 'Spiral':
        pos = nx.spiral_layout(subgraph)
    elif layout == 'Kamada-Kawai':
        for layer, nodes in enumerate(nx.topological_generations(subgraph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
            for node in nodes:
                subgraph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(subgraph, subset_key="layer")
        pos = nx.kamada_kawai_layout(subgraph, pos = pos, scale = 5)    
    elif layout == 'Random':
        pos = nx.random_layout(subgraph)
    # Convert 'expdate' strings to datetime objects
    for node in subgraph.nodes:
        if isinstance(subgraph.nodes[node]['expdate'], str):
            subgraph.nodes[node]['expdate'] = datetime.strptime(subgraph.nodes[node]['expdate'], '%m/%d/%Y').date()

    # Now you can sort the nodes by 'expdate'
    sorted_nodes = sorted(pos.keys(), key=lambda node: subgraph.nodes[node]['expdate'])

    layout = pos

    # Convert NetworkX positions to Plotly format
    pos = {node: (layout[node][0], layout[node][1]) for node in layout}
    
    # Create a list of edges
    edges = list(subgraph.edges)
    if by_date == 'Plot by Exposure Date':
        # Create nodes and edges traces
        node_trace = go.Scatter(x=[subgraph.nodes[node]['expdate'] for node in sorted_nodes],
                                y=[pos[node][1] for node in sorted_nodes],
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(showscale=False, color = 'black'))
        node_trace_agent = go.Scatter(x=[subgraph.nodes[ego]['expdate']],
                                y=[pos[ego][1]],
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(showscale=False, color='red', size=10))  # Set color to black and adjust size
        node_text = []

        for node in sorted_nodes:
            node_attrs = subgraph.nodes[node]
            node_label = (
                f"Node: {node}<br>"
                f"Race: {node_attrs['race']}<br>"
                f"Age: {node_attrs['age']}<br>"
                f"Income: {node_attrs['income']}<br>"
                f"Exposure Date: {node_attrs['expdate']}<br>"
                f"Exposure Location: {node_attrs['exp_place']}"
            )
            node_text.append(node_label)

        node_trace.text = node_text

        # Create custom arrow annotations for each edge
        annotations = []
        # Create edge traces
        edge_traces = []

        for edge in edges:
            x0, y0 = subgraph.nodes[edge[0]]['expdate'], pos[edge[0]][1]
            x1, y1 = subgraph.nodes[edge[1]]['expdate'], pos[edge[1]][1]

            # Create a trace for the line
            line_trace = go.Scatter(x=[x0, x1], y=[y0, y1],
                                    line=dict(width=2, color='#636363'),
                                    hoverinfo='none',
                                    mode='lines')
            edge_traces.append(line_trace)

            annotations.append(dict(
                ax=x0,
                ay=y0,
                axref='x',
                ayref='y',
                x=x1,
                y=y1,
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#636363'
            ))

        # Set the x-axis range based on 'expdate' values
        x_min = min(subgraph.nodes[node]['expdate'] for node in sorted_nodes) - timedelta(days=3)  # Adjust as needed
        x_max = max(subgraph.nodes[node]['expdate'] for node in sorted_nodes) + timedelta(days=3)

        # Create the plotly figure
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=0, l=0, r=0, t=0),
                                        xaxis=dict(range=[x_min, x_max]),  # Set x-axis range
                                        yaxis=dict(showticklabels=False),
                                        annotations=annotations))  # Add the annotations to the figure
        fig.add_trace(node_trace_agent)
        # Show the plot
        return fig
    elif by_date == 'No Date':
        # Create nodes and edges traces
        node_trace = go.Scatter(x=[pos[node][0] for node in sorted_nodes],
                                y=[pos[node][1] for node in sorted_nodes],
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(showscale=False, color = 'black'))
        node_trace_agent = go.Scatter(x=[pos[ego][0]],
                                y=[pos[ego][1]],
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(showscale=False, color='red', size=10))  # Set color to black and adjust size
        node_text = []

        for node in sorted_nodes:
            node_attrs = subgraph.nodes[node]
            node_label = (
                f"Node: {node}<br>"
                f"Race: {node_attrs['race']}<br>"
                f"Age: {node_attrs['age']}<br>"
                f"Income: {node_attrs['income']}<br>"
                f"Exposure Date: {node_attrs['expdate']}<br>"
                f"Exposure Location: {node_attrs['exp_place']}"
            )
            node_text.append(node_label)

        node_trace.text = node_text

        # Create custom arrow annotations for each edge
        annotations = []
        # Create edge traces
        edge_traces = []

        for edge in edges:
            x0, y0 = pos[edge[0]][0], pos[edge[0]][1]
            x1, y1 = pos[edge[1]][0], pos[edge[1]][1]

            # Create a trace for the line
            line_trace = go.Scatter(x=[x0, x1], y=[y0, y1],
                                    line=dict(width=2, color='#636363'),
                                    hoverinfo='none',
                                    mode='lines')
            edge_traces.append(line_trace)

            annotations.append(dict(
                ax=x0,
                ay=y0,
                axref='x',
                ayref='y',
                x=x1,
                y=y1,
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#636363'
            ))

        # Set the x-axis range based on 'expdate' values
        x_min = min(pos[node][0] for node in sorted_nodes) - 1  # Adjust as needed
        x_max = max(pos[node][0] for node in sorted_nodes) + 1

        # Create the plotly figure
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=0, l=0, r=0, t=0),
                                        xaxis=dict(range=[x_min, x_max]),  # Set x-axis range
                                        yaxis=dict(showticklabels=False),
                                        annotations=annotations))  # Add the annotations to the figure
        fig.add_trace(node_trace_agent)
        return fig

def update_graph_statistics(G, logarithm):
    descendants_counts = [len(nx.descendants(G, node)) for node in G.nodes()]
    df_descendants = pd.DataFrame(descendants_counts, columns=['Descendants'])
    fig = px.histogram(df_descendants, x='Descendants')
    fig.show()

    ancestors_counts = [len(nx.ancestors(G, node)) for node in G.nodes()]
    df_ancestors = pd.DataFrame(ancestors_counts, columns=['Ancestors'])
    fig = px.histogram(df_ancestors, x='Ancestors')
    fig.show()

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    df_degree = pd.DataFrame(degree_sequence, columns=['Degree'])
    fig = px.histogram(df_degree, x='Degree')
    fig.show()

def update_influential_graph(top_100, G, by_date, layout, radius):
    ego = top_100.sort_values('descendants')['agent_id'].iloc[:10].sample(1).values[0]
    rad = radius

    # Find all nodes that have a direct path to or from the ego node within a particular radius
    nodes_with_direct_path = [node for node in G.nodes if (nx.has_path(G, node, ego) and nx.shortest_path_length(G, node, ego) <= rad) or (nx.has_path(G, ego, node) and nx.shortest_path_length(G, ego, node) <= rad)]

    # Create a subgraph that only includes nodes with a direct path to or from the ego node
    subgraph = nx.subgraph(G, nodes_with_direct_path)

    # Compute the multipartite_layout using the "layer" node attribute
    if layout == 'Multipartite':
        for layer, nodes in enumerate(nx.topological_generations(subgraph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
            for node in nodes:
                subgraph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(subgraph, subset_key="layer")
    elif layout == 'Planar':
        pos = nx.planar_layout(subgraph)
    elif layout == 'Shell':
        nlist = [list(nodes) for nodes in nx.topological_generations(subgraph)]
        pos = nx.shell_layout(subgraph, nlist = nlist)  
    elif layout == 'Spiral':
        pos = nx.spiral_layout(subgraph)
    elif layout == 'Kamada-Kawai':
        for layer, nodes in enumerate(nx.topological_generations(subgraph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
            for node in nodes:
                subgraph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(subgraph, subset_key="layer")
        pos = nx.kamada_kawai_layout(subgraph, pos = pos, scale = 5)    
    elif layout == 'Random':
        pos = nx.random_layout(subgraph)
    # Convert 'expdate' strings to datetime objects
    for node in subgraph.nodes:
        if isinstance(subgraph.nodes[node]['expdate'], str):
            subgraph.nodes[node]['expdate'] = datetime.strptime(subgraph.nodes[node]['expdate'], '%m/%d/%Y').date()

    # Now you can sort the nodes by 'expdate'
    sorted_nodes = sorted(pos.keys(), key=lambda node: subgraph.nodes[node]['expdate'])

    layout = pos

    # Convert NetworkX positions to Plotly format
    pos = {node: (layout[node][0], layout[node][1]) for node in layout}
       
    # Create a list of edges
    edges = list(subgraph.edges)
    if by_date == 'Plot by Exposure Date':
        # Create nodes and edges traces
        node_trace = go.Scatter(x=[subgraph.nodes[node]['expdate'] for node in sorted_nodes],
                                y=[pos[node][1] for node in sorted_nodes],
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(showscale=False, color = 'black'))
        node_trace_agent = go.Scatter(x=[subgraph.nodes[ego]['expdate']],
                                y=[pos[ego][1]],
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(showscale=False, color='red', size=10))  # Set color to black and adjust size
        node_text = []

        for node in sorted_nodes:
            node_attrs = subgraph.nodes[node]
            node_label = (
                f"Node: {node}<br>"
                f"Race: {node_attrs['race']}<br>"
                f"Age: {node_attrs['age']}<br>"
                f"Income: {node_attrs['income']}<br>"
                f"Exposure Date: {node_attrs['expdate']}<br>"
                f"Exposure Location: {node_attrs['exp_place']}"
            )
            node_text.append(node_label)

        node_trace.text = node_text

        # Create custom arrow annotations for each edge
        annotations = []
        # Create edge traces
        edge_traces = []

        for edge in edges:
            x0, y0 = subgraph.nodes[edge[0]]['expdate'], pos[edge[0]][1]
            x1, y1 = subgraph.nodes[edge[1]]['expdate'], pos[edge[1]][1]

            # Create a trace for the line
            line_trace = go.Scatter(x=[x0, x1], y=[y0, y1],
                                    line=dict(width=2, color='#636363'),
                                    hoverinfo='none',
                                    mode='lines')
            edge_traces.append(line_trace)

            annotations.append(dict(
                ax=x0,
                ay=y0,
                axref='x',
                ayref='y',
                x=x1,
                y=y1,
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#636363'
            ))

        # Set the x-axis range based on 'expdate' values
        x_min = min(subgraph.nodes[node]['expdate'] for node in sorted_nodes) - timedelta(days=3)  # Adjust as needed
        x_max = max(subgraph.nodes[node]['expdate'] for node in sorted_nodes) + timedelta(days=3)

        # Create the plotly figure
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=0, l=0, r=0, t=0),
                                        xaxis=dict(range=[x_min, x_max]),  # Set x-axis range
                                        yaxis=dict(showticklabels=False),
                                        annotations=annotations))  # Add the annotations to the figure
        fig.add_trace(node_trace_agent)
        # Show the plot
        return fig
    elif by_date == 'No Date':
        # Create nodes and edges traces
        node_trace = go.Scatter(x=[pos[node][0] for node in sorted_nodes],
                                y=[pos[node][1] for node in sorted_nodes],
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(showscale=False, color = 'black'))
        node_trace_agent = go.Scatter(x=[pos[ego][0]],
                                y=[pos[ego][1]],
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(showscale=False, color='red', size=10))  # Set color to black and adjust size
        node_text = []

        for node in sorted_nodes:
            node_attrs = subgraph.nodes[node]
            node_label = (
                f"Node: {node}<br>"
                f"Race: {node_attrs['race']}<br>"
                f"Age: {node_attrs['age']}<br>"
                f"Income: {node_attrs['income']}<br>"
                f"Exposure Date: {node_attrs['expdate']}<br>"
                f"Exposure Location: {node_attrs['exp_place']}"
            )
            node_text.append(node_label)

        node_trace.text = node_text

        # Create custom arrow annotations for each edge
        annotations = []
        # Create edge traces
        edge_traces = []

        for edge in edges:
            x0, y0 = pos[edge[0]][0], pos[edge[0]][1]
            x1, y1 = pos[edge[1]][0], pos[edge[1]][1]

            # Create a trace for the line
            line_trace = go.Scatter(x=[x0, x1], y=[y0, y1],
                                    line=dict(width=2, color='#636363'),
                                    hoverinfo='none',
                                    mode='lines')
            edge_traces.append(line_trace)

            annotations.append(dict(
                ax=x0,
                ay=y0,
                axref='x',
                ayref='y',
                x=x1,
                y=y1,
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#636363'
            ))

        # Set the x-axis range based on 'expdate' values
        x_min = min(pos[node][0] for node in sorted_nodes) - 1  # Adjust as needed
        x_max = max(pos[node][0] for node in sorted_nodes) + 1

        # Create the plotly figure
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=0, l=0, r=0, t=0),
                                        xaxis=dict(range=[x_min, x_max]),  # Set x-axis range
                                        yaxis=dict(showticklabels=False),
                                        annotations=annotations))  # Add the annotations to the figure
        fig.add_trace(node_trace_agent)
        return fig

def update_within_graph(layout, by_date, work_or_school, workplace, school, G, df):
    if work_or_school:
        workplace_nodes = list(set(df[df['exp_location']==workplace]['id'].values))
        subgraph = nx.subgraph(G, workplace_nodes)    
    else:
        school_nodes = list(set(df[df['exp_location']==school]['id'].values))
        subgraph = nx.subgraph(G, school_nodes)

    # Compute the multipartite_layout using the "layer" node attribute
    if layout == 'Multipartite':
        for layer, nodes in enumerate(nx.topological_generations(subgraph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
            for node in nodes:
                subgraph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(subgraph, subset_key="layer")
    elif layout == 'Planar':
        pos = nx.planar_layout(subgraph)
    elif layout == 'Shell':
        nlist = [list(nodes) for nodes in nx.topological_generations(subgraph)]
        pos = nx.shell_layout(subgraph, nlist = nlist)  
    elif layout == 'Spiral':
        pos = nx.spiral_layout(subgraph)
    elif layout == 'Kamada-Kawai':
        for layer, nodes in enumerate(nx.topological_generations(subgraph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
            for node in nodes:
                subgraph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(subgraph, subset_key="layer")
        pos = nx.kamada_kawai_layout(subgraph, pos = pos, scale = 5)    
    elif layout == 'Random':
        pos = nx.random_layout(subgraph)
    # Convert 'expdate' strings to datetime objects
    for node in subgraph.nodes:
        if isinstance(subgraph.nodes[node]['expdate'], str):
            subgraph.nodes[node]['expdate'] = datetime.strptime(subgraph.nodes[node]['expdate'], '%m/%d/%Y').date()

    # Now you can sort the nodes by 'expdate'
    sorted_nodes = sorted(pos.keys(), key=lambda node: subgraph.nodes[node]['expdate'])

    layout = pos

    # Convert NetworkX positions to Plotly format
    pos = {node: (layout[node][0], layout[node][1]) for node in layout}
       
    # Create a list of edges
    edges = list(subgraph.edges)
    if by_date == 'Plot by Exposure Date':
        # Create nodes and edges traces
        node_trace = go.Scatter(x=[subgraph.nodes[node]['expdate'] for node in sorted_nodes],
                                y=[pos[node][1] for node in sorted_nodes],
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(showscale=False, color = 'black'))

        node_text = []

        for node in sorted_nodes:
            node_attrs = subgraph.nodes[node]
            node_label = (
                f"Node: {node}<br>"
                f"Race: {node_attrs['race']}<br>"
                f"Age: {node_attrs['age']}<br>"
                f"Income: {node_attrs['income']}<br>"
                f"Exposure Date: {node_attrs['expdate']}<br>"
                f"Exposure Location: {node_attrs['exp_place']}"
            )
            node_text.append(node_label)

        node_trace.text = node_text

        # Create custom arrow annotations for each edge
        annotations = []
        # Create edge traces
        edge_traces = []

        for edge in edges:
            x0, y0 = subgraph.nodes[edge[0]]['expdate'], pos[edge[0]][1]
            x1, y1 = subgraph.nodes[edge[1]]['expdate'], pos[edge[1]][1]

            # Create a trace for the line
            line_trace = go.Scatter(x=[x0, x1], y=[y0, y1],
                                    line=dict(width=2, color='#636363'),
                                    hoverinfo='none',
                                    mode='lines')
            edge_traces.append(line_trace)

            annotations.append(dict(
                ax=x0,
                ay=y0,
                axref='x',
                ayref='y',
                x=x1,
                y=y1,
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#636363'
            ))

        # Set the x-axis range based on 'expdate' values
        x_min = min(subgraph.nodes[node]['expdate'] for node in sorted_nodes) - timedelta(days=3)  # Adjust as needed
        x_max = max(subgraph.nodes[node]['expdate'] for node in sorted_nodes) + timedelta(days=3)

        # Create the plotly figure
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=0, l=0, r=0, t=0),
                                        xaxis=dict(range=[x_min, x_max]),  # Set x-axis range
                                        yaxis=dict(showticklabels=False),
                                        annotations=annotations))  # Add the annotations to the figure
        # Show the plot
        return fig
    elif by_date == 'No Date':
        # Create nodes and edges traces
        node_trace = go.Scatter(x=[pos[node][0] for node in sorted_nodes],
                                y=[pos[node][1] for node in sorted_nodes],
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(showscale=False, color = 'black'))

        node_text = []

        for node in sorted_nodes:
            node_attrs = subgraph.nodes[node]
            node_label = (
                f"Node: {node}<br>"
                f"Race: {node_attrs['race']}<br>"
                f"Age: {node_attrs['age']}<br>"
                f"Income: {node_attrs['income']}<br>"
                f"Exposure Date: {node_attrs['expdate']}<br>"
                f"Exposure Location: {node_attrs['exp_place']}"
            )
            node_text.append(node_label)

        node_trace.text = node_text

        # Create custom arrow annotations for each edge
        annotations = []
        # Create edge traces
        edge_traces = []

        for edge in edges:
            x0, y0 = pos[edge[0]][0], pos[edge[0]][1]
            x1, y1 = pos[edge[1]][0], pos[edge[1]][1]

            # Create a trace for the line
            line_trace = go.Scatter(x=[x0, x1], y=[y0, y1],
                                    line=dict(width=2, color='#636363'),
                                    hoverinfo='none',
                                    mode='lines')
            edge_traces.append(line_trace)

            annotations.append(dict(
                ax=x0,
                ay=y0,
                axref='x',
                ayref='y',
                x=x1,
                y=y1,
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#636363'
            ))

        # Set the x-axis range based on 'expdate' values
        x_min = min(pos[node][0] for node in sorted_nodes) - 1  # Adjust as needed
        x_max = max(pos[node][0] for node in sorted_nodes) + 1

        # Create the plotly figure
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=0, l=0, r=0, t=0),
                                        xaxis=dict(range=[x_min, x_max]),  # Set x-axis range
                                        yaxis=dict(showticklabels=False),
                                        annotations=annotations))  # Add the annotations to the figure
        return fig
