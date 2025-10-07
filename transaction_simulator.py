import json
import random
import networkx as nx
from datetime import datetime
from collections import defaultdict

# configurable parameters
NODES_PATH = "nodes_200.json"
EDGES_PATH = "edges_generated.json"
OUTPUT_TRANSACTIONS = "transactions_simulated.json"
OUTPUT_FEATURES = "node_features.json"

NUM_TRANSACTIONS = 3000
BAD_NODES_NUM = 40
MAX_AMOUNT = 300000 # maximum amount per simulated transaction is 300,000 satoshis
MIN_AMOUNT = 1000   # maximum amount is 1000 satoshis
BAD_NODE_FAIL_RATE = 0.8 

# load graph (convert to a directed graph)
def load_graph(nodes_path, edges_path):
    with open(nodes_path, "r", encoding="utf-8") as f: 
        nodes_data = json.load(f)["nodes"]
    with open(edges_path, "r", encoding="utf-8") as f:
        edges_data = json.load(f)["edges"]

    # convert to a directed graph
    G = nx.DiGraph()
    for node in nodes_data:
        G.add_node(node["pub_key"], alias=node.get("alias", ""))

    for edge in edges_data:
        u = edge["node1_pub"]
        v = edge["node2_pub"]

        # build up directed graph and keep channel policy
        G.add_edge(
            u, v,
            capacity=int(edge.get("capacity", 0)),
            fee_base_msat=int(edge.get("node1_policy", {}).get("fee_base_msat", 1000)),
            fee_rate_milli_msat=int(edge.get("node1_policy", {}).get("fee_rate_milli_msat", 1)),
            cltv_delta=int(edge.get("node1_policy", {}).get("cltv_delta", 1)),
            # initialize balance & slot status
            balance=int(edge.get("capacity", 0)),
            slots_used=0,
            max_slots=30
        )

        # if there is a policy in the other direction, creates additional edge.
        if edge.get("node2_policy") is not None:
            G.add_edge(
                v, u,
                capacity=int(edge.get("capacity", 0)),
                fee_base_msat=int(edge.get("node2_policy", {}).get("fee_base_msat", 1000)),
                fee_rate_milli_msat=int(edge.get("node2_policy", {}).get("fee_rate_milli_msat", 1)),
                cltv_delta=int(edge.get("node2_policy", {}).get("cltv_delta", 1)),
                # initialize balance & slot status for simulation
                balance=int(edge.get("capacity", 0)),
                slots_used=0,
                max_slots=30
            )

    return G

# select bad nodes
def select_bad_nodes(G, num_bad):
    return set(random.sample(list(G.nodes), num_bad))

# simulate transactions
def simulate_transactions(G, bad_nodes, num_transactions):
    transactions = []
    # static forwarding_events_count 
    forwarding_counts = defaultdict(int) 

    node_keys = list(G.nodes())

    for _ in range(num_transactions):
        source, target = random.sample(node_keys, 2)
        amount = random.randint(MIN_AMOUNT, MAX_AMOUNT)
        success = True
        fail_reason = None
        path = []

        try:
            # use Dijkstra to find the shortest path and the path with the lowest overall cost (cost + delay).
            # the weight calculated by the edge_weight function is base fee + rate * amount + time lock
            def edge_weight(u, v, edge_data):
                fee_base = edge_data.get("fee_base_msat", 1000)  # default 1000 msat
                fee_rate = edge_data.get("fee_rate_milli_msat", 1) / 1_000_000  # per msat
                cltv     = edge_data.get("cltv_delta", 1)  # hop delay
                return fee_base + (fee_rate * amount) + cltv  
            path = nx.dijkstra_path(G, source=source, target=target, weight=edge_weight)
            # forwarding_events_count: increment forwarding count for intermediate nodes
            if success and path and len(path) > 2: 
                for i in range(1, len(path) - 1): # remove the head and tail, only count the middle nodes
                    intermediate_node = path[i]
                    forwarding_counts[intermediate_node] += 1
            
            # simulates balance changes: diGraph + balance + jamming slot
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]

                if not G.has_edge(u, v):
                    success = False
                    fail_reason = "no_edge"
                    break

                edge = G[u][v]

                # initialize balance and slot for dynamic routing
                edge.setdefault("balance", edge.get("capacity", 0))
                edge.setdefault("slots_used", 0)
                # 483 HTLCs will cause the memory and computing time to explode.
                edge.setdefault("max_slots", 30) 

                # balance check
                available_balance = edge["balance"]
                if available_balance < amount:
                    success = False
                    fail_reason = "insufficient_balance"
                    break

                # slot check
                if edge["slots_used"] >= edge["max_slots"]:
                    success = False
                    fail_reason = "htlc_slots_full"
                    break

                # simulating jamming attack behavior
                if success and (u in bad_nodes or v in bad_nodes):
                    edge["slots_used"] += 1  # malicious occupation without release

            # if successful, deduct the balance(only for successful paths)
            if success:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge = G[u][v]
                    edge["balance"] -= amount

            # simulating a probing attack by proxy: to achieve the similar outcome as a probing attack
            if fail_reason is None and (source in bad_nodes or target in bad_nodes):
                if random.random() < BAD_NODE_FAIL_RATE:
                    success = False
                    fail_reason = "bad_node"

        except nx.NetworkXNoPath:
            path = [source, target]
            success = False
            fail_reason = "no_path"

        transactions.append({
            "timestamp": datetime.utcnow().isoformat(),
            "source": source,
            "target": target,
            "path": path,
            "success": success,
            "fail_reason": fail_reason,
            "hops": len(path) - 1 if len(path) > 1 else 0,
            "amount": amount
        })
    return transactions, forwarding_counts


# for each node, extract dynamic features
def extract_node_features(G, transactions, forwarding_counts):
    node_features = defaultdict(lambda: {
        "as_source": 0,
        "as_target": 0,
        "success_rate": 0.0,
        "forwarding_events_count": 0,
        "avg_slot_utilization": 0.0,
        "jamming_attempts": 0,
        "jamming_ratio": 0.0
    })

    success_count = defaultdict(int)
    total_count = defaultdict(int)
    # traverse 3,000 transaction records
    for tx in transactions:
        src = tx["source"]
        dst = tx["target"]
        node_features[src]["as_source"] += 1 # if the node is the source, as_source will add one
        node_features[dst]["as_target"] += 1 # if it is a target, as_target will add one
        total_count[src] += 1
        if tx["success"]:
            success_count[src] += 1

    for node, count in forwarding_counts.items():
        node_features[node]["forwarding_events_count"] = count

    for node in node_features:
        if total_count[node]:
            # the number of successful payments, made by the node as the starter, divided by the total number of payments
            node_features[node]["success_rate"] = success_count[node] / total_count[node] 
    # slot and jamming features
    for u, v, edge in G.edges(data=True):
        slots_used = edge.get("slots_used", 0) 
        max_slots  = edge.get("max_slots", 1)
        util = slots_used / max_slots if max_slots > 0 else 0.0
        node_features[u]["avg_slot_utilization"] += util 
        if slots_used > 0:
            node_features[u]["jamming_attempts"] += 1 

    for node in node_features:
        num_edges = G.out_degree(node)
        if num_edges > 0:
            node_features[node]["avg_slot_utilization"] /= num_edges
            # for example: a node has 5 channels, 3 of which are maliciously occupied → ratio = 3/5 = 0.6
            node_features[node]["jamming_ratio"] = node_features[node]["jamming_attempts"] / num_edges
    return node_features

# main method
if __name__ == "__main__":
    G = load_graph(NODES_PATH, EDGES_PATH)
    BAD_NODES = select_bad_nodes(G, BAD_NODES_NUM)
    print(f"Selected bad nodes: {BAD_NODES}")

    txs, forwarding_counts = simulate_transactions(G, BAD_NODES, NUM_TRANSACTIONS) # simulating transactions on the LN
    node_features = extract_node_features(G, txs, forwarding_counts) 

    with open(OUTPUT_TRANSACTIONS, "w") as f:
        json.dump(txs, f, indent=2)

    with open(OUTPUT_FEATURES, "w") as f:
        json.dump(node_features, f, indent=2)
        
    with open("bad_nodes.json", "w") as f:
        json.dump(list(BAD_NODES), f, indent=2)
    print(f"Simulated {NUM_TRANSACTIONS} transactions and extracted features.")
    print(f"node_features.json been created.")
    print(f"transactions_simulated.json been created.")
    print(f"bad_nodes.json been created.")
    