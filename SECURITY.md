# Security

Please report security issues privately to the repository maintainers rather
than opening a public issue.

The build service treats wheel source, patch files, matrix values, container
images, and downloaded release assets as supply-chain inputs. Release jobs must
use pinned source commits, checksum-verified tools, immutable container image
digests where available, and least-privilege credentials.

Runpod API keys used by automation should be limited to Pod management. The SSH
key should be dedicated to ephemeral build/test Pods. Neither secret is written
into a wheel, build manifest, GitHub artifact, container layer, or log.

