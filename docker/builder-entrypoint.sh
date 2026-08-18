#!/usr/bin/env bash
set -euo pipefail

install_authorized_keys() {
    local keys="${SSH_AUTHORIZED_KEYS:-${PUBLIC_KEY:-}}"
    if [[ -z "${keys}" ]]; then
        return
    fi

    install -d -m 0700 /root/.ssh
    : > /root/.ssh/authorized_keys
    while IFS= read -r key; do
        key="${key%$'\r'}"
        [[ -z "${key}" ]] && continue
        case "${key}" in
            ssh-ed25519\ *|ssh-rsa\ *|ecdsa-sha2-nistp256\ *|ecdsa-sha2-nistp384\ *|ecdsa-sha2-nistp521\ *|sk-ssh-ed25519@openssh.com\ *|sk-ecdsa-sha2-nistp256@openssh.com\ *)
                printf '%s\n' "${key}" >> /root/.ssh/authorized_keys
                ;;
            *)
                echo "Refusing malformed SSH public key" >&2
                exit 64
                ;;
        esac
    done <<< "${keys}"
    chmod 0600 /root/.ssh/authorized_keys
}

start_self_termination_watchdog() {
    local seconds="${RUNPOD_SELF_TERMINATE_SECONDS:-}"
    local pod_id="${RUNPOD_POD_ID:-}"
    if [[ -z "${seconds}" ]]; then
        return
    fi
    if [[ ! "${seconds}" =~ ^[0-9]+$ ]] || (( seconds < 600 || seconds > 21600 )); then
        echo "RUNPOD_SELF_TERMINATE_SECONDS must be an integer from 600 through 21600" >&2
        exit 64
    fi
    if [[ ! "${pod_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
        echo "RUNPOD_SELF_TERMINATE_SECONDS requires a valid RUNPOD_POD_ID" >&2
        exit 64
    fi
    if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
        echo "RUNPOD_SELF_TERMINATE_SECONDS requires the Pod-scoped RUNPOD_API_KEY" >&2
        exit 64
    fi
    if ! command -v runpodctl >/dev/null 2>&1; then
        echo "RUNPOD_SELF_TERMINATE_SECONDS requires runpodctl in PATH" >&2
        exit 69
    fi

    nohup bash -c '
        sleep "$1"
        for attempt in 1 2 3; do
            if runpodctl pod delete "$2"; then
                exit 0
            fi
            sleep 30
        done
        exit 1
    ' runpod-self-terminate "${seconds}" "${pod_id}" \
        >/var/log/runpod-self-terminate.log 2>&1 &
    echo "Runpod self-termination watchdog armed for ${seconds} seconds"
}

install_authorized_keys
ssh-keygen -A >/dev/null 2>&1
start_self_termination_watchdog

if [[ "${START_SSHD:-0}" == "1" && "${1:-}" != "/usr/sbin/sshd" ]]; then
    /usr/sbin/sshd
fi

exec "$@"
