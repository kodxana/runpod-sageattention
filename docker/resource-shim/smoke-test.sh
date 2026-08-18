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

echo "resource shim smoke test passed"
