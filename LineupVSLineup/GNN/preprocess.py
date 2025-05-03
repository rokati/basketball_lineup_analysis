import pandas as pd
import numpy as np
import torch
import ast
from torch_geometric.data import Data

def create_lineup_graph(df, pre_df, pre_player_stats_df):
    pre_df.fillna(0, inplace=True)
    df.dropna(subset=['net_score'], inplace=True)
    df = df[~df['net_score'].isin([np.inf, -np.inf])]

    all_lineups = pd.concat([df['home_lineup'], df['away_lineup']]).unique()
    pre_df_lineups = pd.concat([pre_df['home_lineup'], pre_df['away_lineup']]).unique()
    filtered_lineups = [lineup for lineup in all_lineups if lineup in pre_df_lineups]
    all_lineups = np.array(filtered_lineups)

    df = df[df['home_lineup'].isin(all_lineups) & df['away_lineup'].isin(all_lineups)]
    pre_df = pre_df[pre_df['home_lineup'].isin(all_lineups) & pre_df['away_lineup'].isin(all_lineups)]

    lineups_df = pd.DataFrame(all_lineups, columns=['lineup'])
    lineups_df["3pt_made"] = 0
    lineups_df["points"] = 0
    lineups_df["assists"] = 0
    lineups_df["def_rebounds"] = 0
    lineups_df["off_rebounds"] = 0
    lineups_df["fouls"] = 0
    lineups_df["2pt_made"] = 0
    lineups_df["turnovers"] = 0
    lineups_df["ft_made"] = 0
    lineups_df["steals"] = 0
    lineups_df["blocks"] = 0

    cols = ['3pt_made', 'points', 'assists', 'def_rebounds', 'off_rebounds', 'fouls', '2pt_made', 'turnovers', 'ft_made', 'steals', 'blocks']
    for index, row in pre_df.iterrows():
        home_lineup = row['home_lineup']
        away_lineup = row['away_lineup']
        
        for col in cols:
            home_col = 'home_' + col
            away_col = 'away_' + col
            if home_col in row and away_col in row:
                lineups_df.loc[lineups_df['lineup'] == home_lineup, col] += row[home_col]
                lineups_df.loc[lineups_df['lineup'] == away_lineup, col] += row[away_col]
            else:
                print(f'Column {home_col} or {away_col} not found in row')

    # Parse the lineup strings into tuples
    lineups_df['lineup_tuple'] = lineups_df['lineup'].apply(lambda x: ast.literal_eval(x))

    # Create a dictionary mapping player names to their stats
    player_stats_dict = pre_player_stats_df.set_index('PLAYER_NAME').to_dict('index')

    # Initialize columns for aggregated stats
    stat_columns = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV']
    for col in stat_columns:
        lineups_df[col] = 0

    num_lineups_to_not_aggregate = 0

    # Aggregate stats for each lineup
    for index, row in lineups_df.iterrows():
        lineup = row['lineup_tuple']
        for player in lineup:
            if player in player_stats_dict:
                player_stats = player_stats_dict[player]
                for col in stat_columns:
                    lineups_df.at[index, col] += player_stats.get(col, 0)
            else:
                # Delete the lineup if any player is not found in player stats
                lineups_df.drop(index=index, inplace=True)
                num_lineups_to_not_aggregate += 1
                break
    lineups_df.reset_index(drop=True, inplace=True)
    df = df[df['home_lineup'].isin(lineups_df['lineup']) & df['away_lineup'].isin(lineups_df['lineup'])]
    df.reset_index(drop=True, inplace=True)

    # Normalize the specified columns
    for col in stat_columns:
        max_value = lineups_df[col].max()
        if max_value > 0:  # Avoid division by zero
            lineups_df[col] = lineups_df[col] / max_value

    lineups_df.drop(columns=['lineup_tuple'], inplace=True)
    
    # Create unique lineup nodes
    lineup2idx = {lineup: idx for idx, lineup in enumerate(lineups_df['lineup'])}
    
    # Create node features based on lineups_df
    x = torch.tensor(lineups_df.iloc[:, 1:].values, dtype=torch.float)
    
    # Create directed edges with score-based direction
    edge_index = []
    edge_attr = []
    
    for _, row in df.iterrows():
        home = row['home_lineup']
        away = row['away_lineup']
        home_idx = lineup2idx[home]
        away_idx = lineup2idx[away]
        
        # Determine edge direction based on normalized scores
        if row['normalized_home_score'] > row['normalized_away_score']:
            src, dst = home_idx, away_idx
        else:
            src, dst = away_idx, home_idx
            
        edge_index.append([src, dst])
        edge_attr.append(abs(row['net_score']))
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(lineups_df))

# Function to add reversed edges as negatives to a Data object
def add_reversed_negatives(data):
    if data.edge_index.size(1) == 0:
        return data  # Skip if no edges
    original_edges = data.edge_index
    reversed_edges = torch.flip(original_edges, dims=[0])
    
    # Combine original and reversed edges
    edge_label_index = torch.cat([original_edges, reversed_edges], dim=1)
    edge_label = torch.cat([
        torch.ones(original_edges.size(1)),
        torch.zeros(reversed_edges.size(1))
    ])
    
    # Use original edge_attr for both directions
    edge_label_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
    
    # Shuffle the combined data
    perm = torch.randperm(edge_label_index.size(1))
    edge_label_index = edge_label_index[:, perm]
    edge_label_attr = edge_label_attr[perm]
    edge_label = edge_label[perm]
    
    data.edge_label_index = edge_label_index
    data.edge_label_attr = edge_label_attr
    data.edge_label = edge_label
    return data