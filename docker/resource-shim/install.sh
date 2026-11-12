#!/bin/sh
# Build and install the scoped /proc/meminfo compatibility shim.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
prefix=${PODPROC_PREFIX:-/usr/local}
cc=${CC:-cc}

build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT HUP INT TERM

"$cc" \
    -std=c17 \
    -O2 \
    -fPIC \
    -Wall \
    -Wextra \
    -Werror \
    -shared \
    -Wl,-z,relro,-z,now \
    -o "$build_dir/libpodproc.so" \
    "$script_dir/libpodproc.c" \
    -ldl

install -d "$prefix/lib" "$prefix/libexec" "$prefix/bin"
install -m 0755 "$build_dir/libpodproc.so" "$prefix/lib/libpodproc.so"
install -m 0755 "$script_dir/podproc-wrapper" \
    "$prefix/libexec/podproc-wrapper"

for tool in free htop top; do
    if [ -x "/usr/bin/$tool" ] || [ -x "/bin/$tool" ]; then
        ln -sfn ../libexec/podproc-wrapper "$prefix/bin/$tool"
    fi
done
