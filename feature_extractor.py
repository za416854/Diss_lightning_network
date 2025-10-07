import json
import networkx as nx
from pathlib import Path

# file paths
nodes_path = Path("nodes_200.json")
edges_path = Path("edges_generated.json")
transactions_path = Path("transactions_simulated.json")
dynamic_features_path = Path("node_features.json")
output_path = Path("node_feature_vectors.json")

# loading JSON data
with nodes_path.open('r', encoding='utf-8') as f:
    raw_nodes = json.load(f)
nodes_data = raw_nodes["nodes"] if isinstance(raw_nodes, dict) and "nodes" in raw_nodes else raw_nodes

with edges_path.open('r', encoding='utf-8') as f:
    raw_edges = json.load(f)
edges_data = raw_edges["edges"] if isinstance(raw_edges, dict) and "edges" in raw_edges else raw_edges

with transactions_path.open('r', encoding='utf-8') as f:
    transactions = json.load(f)

with dynamic_features_path.open('r', encoding='utf-8') as f:
    dynamic_features = json.load(f)

# build graph
G = nx.Graph()
for node in nodes_data:
    G.add_node(node["pub_key"], alias=node.get("alias", "unknown"), features=node.get("features", {}))

for edge in edges_data:
    try:
        G.add_edge(edge["node1_pub"], edge["node2_pub"], capacity=int(edge["capacity"]))
    except Exception:
        continue

# use the NetworkX library to calculate degree_centrality, clustering, and betweenness_centrality for the entire static network graph G
degree_centrality = nx.degree_centrality(G)
clustering = nx.clustering(G)
betweenness_centrality = nx.betweenness_centrality(G, normalized=True)

# merge features to form feature vectors
node_vectors = {}

for node in G.nodes:
    dyn_feat = dynamic_features.get(node, {})
    # feature_* extraction: read the features dictionary directly from the nodes_200.json file, which declares which specifications each node supports.
    static_feat = G.nodes[node].get("features", {})
    alias = G.nodes[node].get("alias", "")

    vector = {
        "as_source": dyn_feat.get("as_source", 0),
        "as_target": dyn_feat.get("as_target", 0),
        "forwarding_events_count": dyn_feat.get("forwarding_events_count", 0),
        "success_rate": dyn_feat.get("success_rate", 0.0),
        
        # jamming features
        "jamming_attempts": dyn_feat.get("jamming_attempts", 0),
        "jamming_ratio": dyn_feat.get("jamming_ratio", 0.0),
        
        "avg_slot_utilization": dyn_feat.get("avg_slot_utilization", 0.0),
        
        
        "degree_centrality": degree_centrality.get(node, 0.0),
        "betweenness_centrality": betweenness_centrality.get(node, 0.0),
        "clustering": clustering.get(node, 0.0),
        "alias": alias
    }

    # logic for handling Static feature flags
    all_feature_ids = {
        '0', '5', '7', '8', '9', '11', '12', '14', '17', '19', 
        '23', '25', '27', '31', '35', '39', '45', '47', '51', 
        '55', '181', '261', '2023'
    }

    # for each node, first initialise all features to False
    for fid in all_feature_ids:
        vector[f"feature_{fid}"] = False

    # if the node has a feature, set it to True
    static_feat = G.nodes[node].get("features", {})
    for fid, fdata in static_feat.items():
        if fid in all_feature_ids: 
            vector[f"feature_{fid}"] = fdata.get("is_known", False)
    # extract average channel policy features from edges
    base_fees = []
    fee_rates = []
    max_htlcs = []

    for edge in edges_data:
        if edge.get("node1_pub") == node:
            policy = edge.get("node1_policy", {})
            try:
                # traverse all channels belonging to a node in edges_generated.json, extract the publicly disclosed policy values, and calculate the average.
                base_fees.append(int(policy.get("fee_base_msat", 0)))
                fee_rates.append(int(policy.get("fee_rate_milli_msat", 0)))
                max_htlcs.append(int(policy.get("max_htlc_msat", 0)))
            except ValueError:
                continue  # skip malformed entries

    if base_fees:
        vector["avg_base_fee_msat"] = sum(base_fees) / len(base_fees)
    else:
        vector["avg_base_fee_msat"] = 0.0

    if fee_rates:
        vector["avg_fee_rate_milli_msat"] = sum(fee_rates) / len(fee_rates)
    else:
        vector["avg_fee_rate_milli_msat"] = 0.0

    if max_htlcs:
        vector["avg_max_htlc_msat"] = sum(max_htlcs) / len(max_htlcs)
    else:
        vector["avg_max_htlc_msat"] = 0.0
    # extract capacity_centralit: first traversing all channels and calculate the total_network_capacity, then calculating the node_capacity for each node and divide the two.
    node_capacity = sum(
        int(edge.get("capacity", 0))
        for edge in edges_data
        if edge.get("node1_pub") == node or edge.get("node2_pub") == node
    )
    # calculate total network capital
    total_network_capacity = sum(
        int(edge.get("capacity", 0)) for edge in edges_data
    )
    # calculate capacity centrality
    vector["capacity_centrality"] = (
        node_capacity / total_network_capacity if total_network_capacity > 0 else 0.0
    )
    node_vectors[node] = vector

# save node_feature_vectors.json file
with output_path.open("w") as f:
    json.dump(node_vectors, f, indent=2)
    print(f" node_feature_vectors.json has been generated.")
