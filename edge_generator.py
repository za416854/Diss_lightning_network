import json
import random
import uuid
import time
import hashlib
from itertools import combinations
from typing import List

# read parameter statistics range from snapshot
def load_snapshot_edges(file_path: str):
    # load snapshot JSON and return edges list
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("edges", [])

# analyze the capacity and node policy range (min, max) in the snapshot
def analyze_snapshot(edges):
    stats = {
        "capacity": [],
        "time_lock_delta": [],
        "min_htlc": [],
        "fee_base_msat": [],
        "fee_rate_milli_msat": [],
        "max_htlc_msat": [],
        "inbound_fee_base_msat": [],
        "inbound_fee_rate_milli_msat": []
    }
    for e in edges:
        # capacity
        stats["capacity"].append(int(e["capacity"]))
        # node policies
        for side in ["node1_policy", "node2_policy"]:
            if e.get(side):
                p = e[side]
                stats["time_lock_delta"].append(int(p["time_lock_delta"]))
                stats["min_htlc"].append(int(p["min_htlc"]))
                stats["fee_base_msat"].append(int(p["fee_base_msat"]))
                stats["fee_rate_milli_msat"].append(int(p["fee_rate_milli_msat"]))
                stats["max_htlc_msat"].append(int(p["max_htlc_msat"]))
                stats["inbound_fee_base_msat"].append(int(p["inbound_fee_base_msat"]))
                stats["inbound_fee_rate_milli_msat"].append(int(p["inbound_fee_rate_milli_msat"]))
    # returns the (min, max) range
    return {k: (min(v), max(v)) for k, v in stats.items() if v}

allow_null_prob = 0
stats_from_snapshot = None  # use to store statistical range

# generates the policy for both ends. If null is allowed, it will randomly return None.
def generate_channel_policies(allow_null_prob=allow_null_prob):
    if random.random() < allow_null_prob:
        return None, None
    return generate_policy(), generate_policy()

# generate policy based on snapshot statistics range
def generate_policy(allow_null=True):
    if stats_from_snapshot:  
        return {
            "time_lock_delta": random.randint(*stats_from_snapshot["time_lock_delta"]),
            "min_htlc": str(random.randint(*stats_from_snapshot["min_htlc"])),
            "fee_base_msat": str(random.randint(*stats_from_snapshot["fee_base_msat"])),
            "fee_rate_milli_msat": str(random.randint(*stats_from_snapshot["fee_rate_milli_msat"])),
            "disabled": False,
            "max_htlc_msat": str(random.randint(*stats_from_snapshot["max_htlc_msat"])),
            "last_update": int(time.time()),
            "custom_records": {},
            "inbound_fee_base_msat": random.randint(*stats_from_snapshot["inbound_fee_base_msat"]),
            "inbound_fee_rate_milli_msat": random.randint(*stats_from_snapshot["inbound_fee_rate_milli_msat"])
        }

def load_nodes(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return [node['pub_key'] for node in data['nodes'] if 'pub_key' in node]

# first combine channel_{index} with a uuid4(), hash it with SHA-256 and get a 64-bit value to ensure a unique ID every time
def generate_channel_id_and_point(index: int) -> (str, str):
    base = f"channel_{index}_{uuid.uuid4()}"
    channel_id = int(hashlib.sha256(base.encode()).hexdigest(), 16) % (1 << 64)
    chan_point_txid = hashlib.sha256((base + "_tx").encode()).hexdigest()
    chan_point_index = random.randint(0, 1)
    return str(channel_id), f"{chan_point_txid}:{chan_point_index}"

def generate_edges(pub_keys: List[str], num_edges: int, allow_parallel: bool = False):
    channels = []
    # two node pairing methods, this case is to traverse all possible combinations (pairing two by two), and then randomly extract a fixed number
    all_possible_pairs = list(combinations(pub_keys, 2)) 
    if not allow_parallel:
        random.shuffle(all_possible_pairs)
        selected_pairs = all_possible_pairs[:min(num_edges, len(all_possible_pairs))]
    else:
        # two node pairing methods, this case is random sampling pairing for the same pair of nodes to be selected multiple times: simulating parallel channels
        selected_pairs = [tuple(random.sample(pub_keys, 2)) for _ in range(num_edges)] 

    for i, pair in enumerate(selected_pairs):
        node1, node2 = pair
        channel_id, chan_point = generate_channel_id_and_point(i)
        # change capacity to snapshot statistics range
        if stats_from_snapshot:
            capacity = str(random.randint(*stats_from_snapshot["capacity"]))
        else:
            capacity = str(random.randint(10000, 1000000))
        node1_policy, node2_policy = generate_channel_policies()
        channels.append({
            "channel_id": channel_id,
            "chan_point": chan_point,
            "last_update": 0,
            "node1_pub": node1,
            "node2_pub": node2,
            "capacity": capacity,
            "balance_A_to_B": int(capacity) // 2,
            "balance_B_to_A": int(capacity) // 2,
            "max_slots": 30,        
            "slots_used": 0,        
            "node1_policy": node1_policy,
            "node2_policy": node2_policy,
            "custom_records": {}
        })
    return channels

def save_edges(edges: List[dict], output_path: str):
    with open(output_path, 'w') as f:
        json.dump({"edges": edges}, f, indent=4)

if __name__ == "__main__":
    node_file_path = "nodes_200.json"
    snapshot_file_path = "larger_and_real_json_file_CHANNELS.json"  # actual channel snapshot, can be extracted to get the range
    output_file_path = "edges_generated.json"
    NUM_EDGES = 500
    ALLOW_PARALLEL_CHANNELS = False

    edges_snapshot = load_snapshot_edges(snapshot_file_path)
    stats_from_snapshot = analyze_snapshot(edges_snapshot)
    print("The statistical range obtained from the snapshot:", stats_from_snapshot)

    pub_keys = load_nodes(node_file_path)
    print(f"nodes number: {len(pub_keys) - 1}")
    edges = generate_edges(pub_keys, NUM_EDGES, allow_parallel=ALLOW_PARALLEL_CHANNELS)
    save_edges(edges, output_file_path)
    print(f"Generated {len(edges)} edges saved to {output_file_path}")