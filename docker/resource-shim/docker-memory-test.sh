#!/bin/sh
# Optional end-to-end check for an already built image. No GPU is required.
set -eu

image=${1:?usage: docker-memory-test.sh IMAGE}
limit_bytes=1073741824

docker run --rm \
    --memory "$limit_bytes" \
    --entrypoint python3 \
    "$image" \
    -c '
import json
import os
import subprocess

configured = 1073741824
payload = json.loads(
    subprocess.check_output(["/usr/local/bin/pod-resources", "--json"], text=True)
)
memory = payload["memory"]
assert memory["limited"] is True, memory
assert 0 < memory["limit_bytes"] <= configured, memory

environment = dict(os.environ, LC_ALL="C")
output = subprocess.check_output(
    ["/usr/local/bin/free", "--bytes"], text=True, env=environment
)
mem_line = next(line for line in output.splitlines() if line.startswith("Mem:"))
displayed_total = int(mem_line.split()[1])
# /proc/meminfo is reported in KiB, so allow its normal downward rounding.
assert abs(displayed_total - memory["limit_bytes"]) < 1024, (output, memory)
print("1 GiB Docker memory-limit integration test passed")
'

