from pathlib import Path
import pandas as pd
import json
import re

DEVICE_MAP = {
    "7": 0,
    "a": 1,
    "47": 2,
    "4d": 3,
    "87": 4,
    "8d": 5,
    "c7": 6,
    "ca": 7,
}

def metadata_from_filename(path: Path):
    """ extract metadata (server, tp degree, interconnect) from file name """
    name = path.stem
    parts = name.split("_")

    meta = {
        "server": None,
        "tp": None,
        "timestamp": None,
        "interconnect": None,
        "file_type": None,
    }

    # p2p test output file
    if "p2p" in name:
        meta["file_type"] = "p2p"
        meta["server"] = parts[0]
        meta["timestamp"] = parts[1] + "_" + parts[2]
        meta["interconnect"] = "p2p"
        return meta

    # nccl test output file
    if "tp" in name:
        meta["file_type"] = "nccl"
        meta["server"] = parts[0]
        meta["tp"] = int(parts[1].replace("tp", ""))
        meta["timestamp"] = parts[2] + "_" + parts[3]
        interconnect = parts[4]
        if interconnect == "default":
            meta["interconnect"] = interconnect
        if interconnect == "shm":
            meta["interconnect"] = "p2p_disabled" 
        elif interconnect == "socket":
            meta["interconnect"] = "network_socket"

    return meta

def parse_p2ptest_output(path: Path):
    """ parse p2ptest bandwidth and latency matrices """
    content = path.read_text().splitlines()
    
    matrices = {
        "uni_disabled": None,
        "uni_enabled": None,
        "bi_disabled": None,
        "bi_enabled": None,
        "lat_disabled": None,
        "lat_enabled": None,
    }
    def parse_matrix(content, header_idx):
        matrix = []
        i = header_idx + 1
        while i < len(content):
            curr_line = content[i].strip()
            if "GPU" in curr_line or "D\\D" in curr_line:
                i += 1
                break
            i += 1

        while i < len(content):
            line = content[i].strip()

            if not line or "Matrix" in line or "CPU" in line:
                break

            parts = line.split()
            if parts and parts[0].isdigit():
                row_vals = [float(val) for val in parts[1:]]
                matrix.append(row_vals)
            i += 1
        
        return matrix, i

    i = 0
    while i < len(content):
        line = content[i]

        if "Unidirectional P2P=Disabled" in line:
            matrices["uni_disabled"], i = parse_matrix(content, i)
            continue
        if "Unidirectional P2P=Enabled" in line:
            matrices["uni_enabled"], i = parse_matrix(content, i)
            continue
        if "Bidirectional P2P=Disabled" in line:
            matrices["bi_disabled"], i = parse_matrix(content, i)
            continue
        if "Bidirectional P2P=Enabled" in line:
            matrices["bi_enabled"], i = parse_matrix(content, i)
            continue
        if "P2P=Disabled Latency" in line:
            matrices["lat_disabled"], i = parse_matrix(content, i)
            continue
        if "P2P=Enabled Latency" in line:
            matrices["lat_enabled"], i = parse_matrix(content, i)
            continue

        i +=1
    return matrices

def parse_nccltest_output(path: Path):
    content = path.read_text().splitlines()

    sizes = []
    algbw = []
    busbw = []
    latencies = []
    gpu_ids = []

    rank_pci_pattern = re.compile(
        r"Rank\s+\d+.*device\s+\d+\s+\[[0-9a-fA-F]+:([0-9a-fA-F]+):[0-9a-fA-F]+\]")
    nccl_data_pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+\S+\s+\S+\s+-?\d+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)"
    )

    for line in content:
        # parse metadata lines to detect gpu ids
        rank_match = rank_pci_pattern.search(line)
        if rank_match:
            pci_bus_id = rank_match.group(1).lower()
            if pci_bus_id in DEVICE_MAP:
                gpu_id = DEVICE_MAP[pci_bus_id]
                if gpu_id not in gpu_ids:
                    gpu_ids.append(gpu_id)
            continue

        # parse performance metrics rows
        match = nccl_data_pattern.match(line)
        if match:
            sizes.append(int(match.group(1)))         # size (bytes)
            latencies.append(float(match.group(3)))   # time (us)
            algbw.append(float(match.group(4)))       # algbw (GB/s)
            busbw.append(float(match.group(5)))       # busbw (GB/s)
    
    # sort detected gpu ids
    gpu_ids.sort()

    return {
        "gpu_ids": gpu_ids,
        "nccl_sizes": sizes,
        "nccl_latencies": latencies,
        "nccl_algbw": algbw,
        "nccl_busbw": busbw,
    }
def slice_matrix_on_gpus(matrix, gpu_ids):
    """ slices full topology down to only contain active gpu ids """
    if not matrix or not gpu_ids:
        return None
    sub_matrix = []
    for row_idx in gpu_ids:
        row_data = []
        for col_idx in gpu_ids:
            row_data.append(matrix[row_idx][col_idx])
        sub_matrix.append(row_data)
    return sub_matrix

def build_measurement_csv(results_dir: Path):
    all_files = list(results_dir.glob("*.txt"))
    
    # get meta data from filenames, parse p2p output
    p2p_topology_lookup = {}
    nccl_files = []

    for f in all_files:
        meta = metadata_from_filename(f)
        if meta["file_type"] == "p2p":
            p2p_key = (meta["server"], meta["timestamp"])
            p2p_topology_lookup[p2p_key] = parse_p2ptest_output(f)
        elif meta["file_type"] == "nccl":
            nccl_files.append((f, meta))

    # parse nccl outputs, all configs
    rows = []
    for f, meta in nccl_files:
        server = meta["server"]
        timestamp = meta["timestamp"]
        interconnect = meta["interconnect"]

        # match p2p matrix to run by timestamp
        p2p_key = (server, timestamp)
        p2p_data = p2p_topology_lookup.get(p2p_key, None)

        uni_matrix, bi_matrix, lat_matrix = None, None, None
        if p2p_data:
            if interconnect == "default":
                uni_matrix = p2p_data["uni_enabled"]
                bi_matrix = p2p_data["bi_enabled"]
                lat_matrix = p2p_data["lat_enabled"]
            elif interconnect == "shm":
                uni_matrix = p2p_data["uni_disabled"]
                bi_matrix = p2p_data["bi_disabled"]
                lat_matrix = p2p_data["lat_disabled"]

        # parse nccl measurements and gpu_ids
        nccl_data = parse_nccltest_output(f)
        active_gpus = nccl_data["gpu_ids"]

        active_uni_matrix = slice_matrix_on_gpus(uni_matrix, active_gpus)
        active_bi_matrix = slice_matrix_on_gpus(bi_matrix, active_gpus)
        active_lat_matrix = slice_matrix_on_gpus(lat_matrix, active_gpus)

        # compile config row
        experiment_row = {
            # configuration
            "server": server,
            "timestamp": timestamp,
            "tp": meta["tp"],
            "interconnect": interconnect,
            "gpu_ids": json.dumps(nccl_data["gpu_ids"]),
            # hardware: full p2p matrices
            # "p2p_uni_bw_matrix": json.dumps(uni_matrix) if uni_matrix else None,
            # "p2p_bi_bw_matrix": json.dumps(bi_matrix) if bi_matrix else None,
            # "p2p_lat_matrix": json.dumps(lat_matrix) if lat_matrix else None,
            # hardware: full p2p matrices
            "active_p2p_uni_bw_matrix": json.dumps(active_uni_matrix) if active_uni_matrix else None,
            "active_p2p_bi_bw_matrix": json.dumps(active_bi_matrix) if active_bi_matrix else None,
            "active_p2p_lat_matrix": json.dumps(active_lat_matrix) if active_lat_matrix else None,
            
            # nccl measurements
            "nccl_message_sizes": json.dumps(nccl_data["nccl_sizes"]),
            "nccl_algbw": json.dumps(nccl_data["nccl_algbw"]),
            "nccl_busbw": json.dumps(nccl_data["nccl_busbw"]),
            "nccl_latency": json.dumps(nccl_data["nccl_latencies"]),
        }
        rows.append(experiment_row)

    df = pd.DataFrame(rows)
    output_csv = results_dir / "interconnect_characterization_frames.csv"
    df.to_csv(output_csv, index=False)
    print(f"saved {output_csv} ({len(df)} distinct test iters)")
    return df
    
if __name__ == "__main__":
    results_dir = Path('./results/interconnect_characterization')

    # build measurement csv with measurements from all test outputs
    df = build_measurement_csv(results_dir)
