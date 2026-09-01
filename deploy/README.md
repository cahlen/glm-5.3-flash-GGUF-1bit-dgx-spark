# deploy/

Host-level files that live **outside** the repo on the deployment machine, kept
here so they are version-controlled. The host installs from the original paths —
these copies are not the ones systemd or the installer read.

| file | installed to | why it must live there |
|---|---|---|
| `install-gpu-mem-guard.sh` | `~/install-gpu-mem-guard.sh` | `tests/test_mem_guard.py` reads the watchdog out of this file's `DAEMON` heredoc — that path is the source of truth. Apply with `sudo bash ~/install-gpu-mem-guard.sh`. |
| `glm53.service` | `~/.config/systemd/user/glm53.service` | systemd user units load from there. `systemctl --user daemon-reload` after copying. |
| `opencode-provider.json` | merge into `~/.config/opencode/opencode.json` on the **client** | OpenCode runs on the workstation, not on the Spark. |

`tests/test_deploy_sync.py` fails the moment a copy here drifts from the
installed original. If it fails, copy whichever file is newer over the other —
do not edit the test.

## Why the guard matters

`gpu-mem-guard.service` SIGKILLs the largest GPU process when `MemAvailable`
drops below a 20 GiB floor. A ~91 GiB resident model is *always* the largest, and
it legitimately parks memory near that floor, so the watchdog killed the server
**11 times in under three hours** on 2026-08-31 (`code=killed, status=9/KILL`,
no core, no llama.cpp fault). The patched guard skips units named in
`GPU_GUARD_EXEMPT_UNITS` (default `glm53.service`) and kills the next-biggest GPU
process instead; if every GPU process is exempt it logs and does nothing, because
at that point the remedy is headroom, not a kill.

**If the service is renamed, update the exemption or the kill loop returns.**

## Install from scratch

```bash
sudo bash deploy/install-gpu-mem-guard.sh       # watchdog, with glm53.service exempt
cp deploy/glm53.service ~/.config/systemd/user/
# point ExecStart at wherever you cloned this repo, then:
systemctl --user daemon-reload
systemctl --user enable --now glm53
loginctl enable-linger "$USER"                  # survive logout / reboot
./serving/glm53-health.sh
```
