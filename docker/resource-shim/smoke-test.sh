#!/bin/sh
# Linux CI gate for compilation, fail-open behavior, wrapper argument/exit
# forwarding, and (when a finite cgroup exists) a basic displayed-total check.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
test_root=$(mktemp -d)
trap 'rm -rf "$test_root"' EXIT HUP INT TERM

CC=${CC:-gcc} PODPROC_PREFIX="$test_root/usr/local" "$script_dir/install.sh"
library="$test_root/usr/local/lib/libpodproc.so"
wrapper="$test_root/usr/local/libexec/podproc-wrapper"
receipt="$test_root/verified-memory-bytes-v1"
receipt_library="$test_root/libpodproc-receipt-test.so"
fixture="$test_root/fixture"
cgroup="$fixture/sys/fs/cgroup"
mkdir -p "$fixture/proc/self" "$cgroup"
printf '%s\n' \
    'MemTotal:        1048576 kB' \
    'MemFree:          524288 kB' \
    'MemAvailable:     786432 kB' \
    'Buffers:               0 kB' \
    'Cached:           131072 kB' \
    'SReclaimable:          0 kB' \
    'Shmem:                 0 kB' \
    'SwapTotal:        262144 kB' \
    'SwapFree:         262144 kB' > "$fixture/proc/meminfo"
printf '0::/\n' > "$fixture/proc/self/cgroup"
printf '29 23 0:26 / %s rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n' \
    "$cgroup" > "$fixture/proc/self/mountinfo"
printf 'max\n' > "$cgroup/memory.max"
printf '268435456\n' > "$cgroup/memory.current"
printf 'inactive_file 67108864\n' > "$cgroup/memory.stat"
printf 'max\n' > "$cgroup/memory.swap.max"
printf '0\n' > "$cgroup/memory.swap.current"

"${CC:-gcc}" \
    -std=c17 -O2 -fPIC -Wall -Wextra -Werror -shared \
    -Wl,-z,relro,-z,now \
    "-DPODPROC_RECEIPT_PATH=\"$receipt\"" \
    "-DPODPROC_RECEIPT_OWNER_UID=$(id -u)" \
    "-DPODPROC_RECEIPT_OWNER_GID=$(id -g)" \
    "-DPODPROC_HOST_MEMINFO_PATH=\"$fixture/proc/meminfo\"" \
    "-DPODPROC_SELF_CGROUP_PATH=\"$fixture/proc/self/cgroup\"" \
    "-DPODPROC_SELF_MOUNTINFO_PATH=\"$fixture/proc/self/mountinfo\"" \
    -o "$receipt_library" "$script_dir/libpodproc.c" -ldl

# PODPROC_DISABLE must expose the real host total. Other meminfo counters are
# intentionally not compared because they can change between consecutive reads.
host_total=$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
disabled_total=$(
    PODPROC_DISABLE=1 LD_PRELOAD="$library" \
        awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo
)
[ "$host_total" = "$disabled_total" ] || {
    echo "PODPROC_DISABLE changed MemTotal" >&2
    exit 1
}
PODPROC_DISABLE=1 LD_PRELOAD="$library" /usr/bin/free --bytes >/dev/null

# Invoke the shared wrapper through a supported basename without changing the
# machine's /usr/local. Arguments and the underlying command's status survive.
ln -s "$wrapper" "$test_root/free"
PODPROC_LIBRARY="$library" "$test_root/free" --bytes >/dev/null
if "$wrapper" >/dev/null 2>&1; then
    echo "direct wrapper invocation unexpectedly succeeded" >&2
    exit 1
else
    status=$?
    [ "$status" -eq 64 ] || {
        echo "direct wrapper returned $status, expected 64" >&2
        exit 1
    }
fi

# Normal preload is required to remain usable even on an unlimited host: the
# library deliberately falls through to the original meminfo in that case.
LD_PRELOAD="$library" /usr/bin/free --bytes >/dev/null

# A verified assignment may cap presentation when a private cgroup namespace
# exposes membership and mount root as `/` with memory.max=max. The production
# build accepts only the fixed root-owned receipt; this test build substitutes
# private proc/cgroup paths and the invoking test uid/gid.
receipt_total_kib=524288
receipt_bytes=536870912
printf '%s\n' "$receipt_bytes" > "$receipt"
chmod 0444 "$receipt"
displayed_total=$(LD_PRELOAD="$receipt_library" \
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_total" -eq "$receipt_total_kib" ] || {
    echo "verified receipt did not cap displayed MemTotal" >&2
    exit 1
}
wrapper_total=$(LC_ALL=C PODPROC_LIBRARY="$receipt_library" \
    "$test_root/free" --bytes | awk 'NR == 2 { print $2; exit }')
[ "$wrapper_total" -eq "$receipt_bytes" ] || {
    echo "scoped free wrapper did not display the verified receipt" >&2
    exit 1
}
displayed_swap=$(LD_PRELOAD="$receipt_library" \
    awk '/^SwapTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_swap" -eq 0 ] || {
    echo "assignment-only presentation exposed unverified swap" >&2
    exit 1
}

# The command-scoped value is another conservative ceiling; when both are
# present, the smaller value wins.
environment_total_kib=393216
environment_bytes=402653184
displayed_total=$(RUNPOD_ASSIGNED_MEMORY_BYTES="$environment_bytes" \
    LD_PRELOAD="$receipt_library" \
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_total" -eq "$environment_total_kib" ] || {
    echo "command-scoped assignment did not cap displayed MemTotal" >&2
    exit 1
}
displayed_total=$(RUNPOD_ASSIGNED_MEMORY_BYTES=not-a-number \
    LD_PRELOAD="$receipt_library" \
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_total" -eq "$host_total" ] || {
    echo "malformed command-scoped capacity did not fail open" >&2
    exit 1
}
displayed_total=$(RUNPOD_ASSIGNED_MEMORY_BYTES=0536870912 \
    LD_PRELOAD="$receipt_library" \
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_total" -eq "$host_total" ] || {
    echo "non-canonical command-scoped capacity did not fail open" >&2
    exit 1
}

# With the receipt absent, the command-scoped verified value is sufficient for
# descendants of the orchestrated build command.
chmod 0644 "$receipt"
rm "$receipt"
displayed_total=$(RUNPOD_ASSIGNED_MEMORY_BYTES="$receipt_bytes" \
    LD_PRELOAD="$receipt_library" \
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_total" -eq "$receipt_total_kib" ] || {
    echo "environment-only assignment did not cap displayed MemTotal" >&2
    exit 1
}

# A real finite cgroup remains the smaller, kernel-enforced ceiling.
printf '268435456\n' > "$cgroup/memory.max"
printf '134217728\n' > "$cgroup/memory.current"
printf '%s\n' "$receipt_bytes" > "$receipt"
chmod 0444 "$receipt"
displayed_total=$(LD_PRELOAD="$receipt_library" \
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_total" -eq 262144 ] || {
    echo "finite cgroup did not remain the smallest displayed capacity" >&2
    exit 1
}
printf 'max\n' > "$cgroup/memory.max"
printf '268435456\n' > "$cgroup/memory.current"

# Malformed/truncated receipts and a capacity below current usage must expose
# the real host view rather than synthesize untrustworthy numbers.
chmod 0644 "$receipt"
displayed_total=$(LD_PRELOAD="$receipt_library" \
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_total" -eq "$host_total" ] || {
    echo "writable receipt did not fail open to host MemTotal" >&2
    exit 1
}
printf '%s' "$receipt_bytes" > "$receipt"
chmod 0444 "$receipt"
displayed_total=$(LD_PRELOAD="$receipt_library" \
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_total" -eq "$host_total" ] || {
    echo "malformed receipt did not fail open to host MemTotal" >&2
    exit 1
}
chmod 0644 "$receipt"
mv "$receipt" "$receipt.target"
chmod 0444 "$receipt.target"
ln -s "$receipt.target" "$receipt"
displayed_total=$(LD_PRELOAD="$receipt_library" \
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_total" -eq "$host_total" ] || {
    echo "symlink receipt did not fail open to host MemTotal" >&2
    exit 1
}
rm "$receipt" "$receipt.target"
printf '%s\n' "$receipt_bytes" > "$receipt.target"
chmod 0444 "$receipt.target"
ln "$receipt.target" "$receipt"
displayed_total=$(LD_PRELOAD="$receipt_library" \
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_total" -eq "$host_total" ] || {
    echo "multiply-linked receipt did not fail open to host MemTotal" >&2
    exit 1
}
rm "$receipt" "$receipt.target"
printf '1\n' > "$receipt"
chmod 0444 "$receipt"
displayed_total=$(LD_PRELOAD="$receipt_library" \
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_total" -eq "$host_total" ] || {
    echo "capacity below current usage did not fail open to host MemTotal" >&2
    exit 1
}
chmod 0644 "$receipt"
printf '%s\n' "$receipt_bytes" > "$receipt"
chmod 0444 "$receipt"
rm "$cgroup/memory.current"
displayed_total=$(LD_PRELOAD="$receipt_library" \
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
[ "$displayed_total" -eq "$host_total" ] || {
    echo "missing current usage did not fail open to host MemTotal" >&2
    exit 1
}

echo "resource shim smoke test passed"
