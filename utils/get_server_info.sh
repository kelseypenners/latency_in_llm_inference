#!/bin/bash

OUTDIR="server_info_$(hostname)_20260202"
mkdir -p "$OUTDIR"

echo "saving server info to $OUTDIR"

echo "---------- host ----------" > "$OUTDIR/host.txt"
hostname >> "$OUTDIR/host.txt"

# gpu inventory
echo "---------- nvidia-smi ----------" > "$OUTDIR/nvidia_smi.txt"
nvidia-smi >> "$OUTDIR/nvidia_smi.txt"

# gpu / nic topology
echo "---------- topology ----------" > "$OUTDIR/topology.txt"
nvidia-smi topo -m >> "$OUTDIR/topology.txt"

# network hardware
echo "---------- network ----------" > "$OUTDIR/network.txt"
lspci | grep -i ethernet >> "$OUTDIR/network.txt"

echo "done!"