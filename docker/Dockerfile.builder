# syntax=docker/dockerfile:1.7

FROM ubuntu:24.04

ARG CUDA_VERSION
ARG CUDA_VERSION_DASH
ARG TORCH_VERSION
ARG TORCH_CUDA_VERSION
ARG TORCH_INDEX_SUFFIX
ARG CUDA_KEYRING_SHA256=d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba
ARG RUNPODCTL_VERSION=v2.3.0
ARG RUNPODCTL_SHA256=908f2210571e8a26a1cba6fb45f09556b34dcad3e1b20dd502df2adf7a57c169

LABEL org.opencontainers.image.title="SageAttention wheel builder" \
      org.opencontainers.image.description="GPU-independent, SSH-ready SageAttention wheel builder" \
      org.opencontainers.image.source="https://github.com/kodxana/runpod-sageattention" \
      io.runpod.sageattention.cuda="${CUDA_VERSION}" \
      io.runpod.sageattention.torch="${TORCH_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/targets/x86_64-linux/lib/stubs:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib \
    BUILDER_CUDA_VERSION=${CUDA_VERSION} \
    BUILDER_TORCH_VERSION=${TORCH_VERSION} \
    BUILDER_TORCH_CUDA_VERSION=${TORCH_CUDA_VERSION} \
    BUILDER_TORCH_INDEX_SUFFIX=${TORCH_INDEX_SUFFIX}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN test -n "${CUDA_VERSION}" \
    && test -n "${CUDA_VERSION_DASH}" \
    && test -n "${TORCH_VERSION}" \
    && test -n "${TORCH_CUDA_VERSION}" \
    && test -n "${TORCH_INDEX_SUFFIX}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        binutils \
        build-essential \
        ca-certificates \
        curl \
        file \
        git \
        htop \
        jq \
        less \
        nano \
        ninja-build \
        openssh-client \
        openssh-server \
        patch \
        patchelf \
        procps \
        python3.12 \
        python3.12-dev \
        python3-pip \
        python3.12-venv \
        rsync \
        tar \
        tini \
        unzip \
        vim-tiny \
        wget \
        zip \
    && curl -fsSLo /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && printf '%s  %s\n' "${CUDA_KEYRING_SHA256}" /tmp/cuda-keyring.deb | sha256sum -c - \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm -f /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        "cuda-minimal-build-${CUDA_VERSION_DASH}" \
        "libcublas-dev-${CUDA_VERSION_DASH}" \
        "libcusolver-dev-${CUDA_VERSION_DASH}" \
        "libcusparse-dev-${CUDA_VERSION_DASH}" \
    && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/runpodctl.tar.gz \
        "https://github.com/runpod/runpodctl/releases/download/${RUNPODCTL_VERSION}/runpodctl-linux-amd64.tar.gz" \
    && printf '%s  %s\n' "${RUNPODCTL_SHA256}" /tmp/runpodctl.tar.gz | sha256sum -c - \
    && tar -xzf /tmp/runpodctl.tar.gz -C /usr/local/bin runpodctl \
    && chmod 0755 /usr/local/bin/runpodctl \
    && rm -f /tmp/runpodctl.tar.gz \
    && runpodctl version

ENV VIRTUAL_ENV=/opt/sageattention-builder-venv \
    PATH=/opt/sageattention-builder-venv/bin:${PATH}

RUN /usr/bin/python3.12 -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/python" -m pip install --upgrade "pip==26.1.2" \
    && "${VIRTUAL_ENV}/bin/python" -m pip install \
        "build==1.2.2.post1" \
        "packaging==25.0" \
        "setuptools==80.9.0" \
        "wheel==0.45.1" \
    && "${VIRTUAL_ENV}/bin/python" -m pip install \
        --index-url https://pypi.org/simple \
        --extra-index-url "https://download.pytorch.org/whl/${TORCH_INDEX_SUFFIX}" \
        "torch==${TORCH_VERSION}" \
    && EXPECTED_TORCH="${TORCH_VERSION}" EXPECTED_CUDA="${TORCH_CUDA_VERSION}" \
        "${VIRTUAL_ENV}/bin/python" -c 'import os, torch; expected_torch = os.environ["EXPECTED_TORCH"]; expected_cuda = os.environ["EXPECTED_CUDA"]; assert torch.__version__ == expected_torch, f"torch mismatch: expected {expected_torch}, got {torch.__version__}"; assert torch.version.cuda == expected_cuda, f"torch CUDA mismatch: expected {expected_cuda}, got {torch.version.cuda}"'

RUN install -d -m 0700 /root/.ssh \
    && install -d -m 0755 /run/sshd /workspace /work \
    && printf '%s\n' \
        'PasswordAuthentication no' \
        'KbdInteractiveAuthentication no' \
        'PermitRootLogin prohibit-password' \
        'PubkeyAuthentication yes' \
        'AllowTcpForwarding yes' \
        'X11Forwarding no' \
        'PrintMotd no' \
        > /etc/ssh/sshd_config.d/99-wheel-builder.conf

COPY docker/builder-entrypoint.sh /usr/local/bin/builder-entrypoint
COPY tools/pod_resources.py /usr/local/bin/pod-resources
COPY docker/resource-shim /tmp/resource-shim
RUN sh /tmp/resource-shim/smoke-test.sh \
    && sh /tmp/resource-shim/install.sh \
    && rm -rf /tmp/resource-shim \
    && chmod 0755 /usr/local/bin/builder-entrypoint /usr/local/bin/pod-resources

WORKDIR /workspace
EXPOSE 22

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/builder-entrypoint"]
CMD ["/usr/sbin/sshd", "-D", "-e"]
