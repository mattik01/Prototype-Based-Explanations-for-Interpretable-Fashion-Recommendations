# Termux SSH & Wake-on-LAN Setup

## Overview

The Samsung Galaxy S10 runs Termux as an always-on LAN relay. It serves two purposes:
1. **WoL relay** — sends Wake-on-LAN magic packets to the GPU machine (`DESKTOP-715IF4P`) from anywhere via Tailscale
2. **SSH gateway** — accessible from USB, WiFi, or the internet (via Tailscale)

---

## Device Info

| Property | Value |
|---|---|
| Phone | Samsung Galaxy S10 |
| ADB serial | `R58N82ATSVJ` |
| Termux user | `u0_a317` |
| SSH port | `8022` |
| WiFi IP | `192.168.68.51` (DHCP — may change) |
| Tailscale IP | `100.71.183.60` (stable) |

## GPU Machine Info

| Property | Value |
|---|---|
| Hostname | `DESKTOP-715IF4P` |
| LAN IP | `192.168.68.58` (DHCP reserved) |
| MAC address | `D8:43:AE:B3:D7:3D` |
| Tailscale IP | `100.85.86.56` |

---

## Quick Reference

### Wake the GPU machine (from anywhere)
```bash
ssh termux-tailscale "python ~/wol_gpu.py"
```

### SSH into the phone
```bash
ssh termux              # via USB (needs adb forward first)
ssh termux-wifi         # via local WiFi
ssh termux-tailscale    # from anywhere (via Tailscale)
```

### USB port forwarding (needed each time phone is reconnected via USB)
```bash
# Windows
C:\Users\glaes\platform-tools\adb.exe forward tcp:8022 tcp:8022

# Git Bash / WSL
/c/Users/glaes/platform-tools/adb.exe forward tcp:8022 tcp:8022
```

---

## SSH Config (`~/.ssh/config`)

```
Host termux
    HostName localhost
    Port 8022
    User u0_a317

Host termux-wifi
    HostName 192.168.68.51
    Port 8022
    User u0_a317

Host termux-tailscale
    HostName 100.71.183.60
    Port 8022
    User u0_a317
```

This config exists on the Windows GPU machine at `C:\Users\glaes\.ssh\config`.

**TODO:** Add same config on the laptop (`mattik01`) after copying its SSH key to the phone.

---

## Authorized SSH Keys

Currently only the Windows GPU machine key is authorized:
```
ssh-ed25519 AAAAC3... glaes@DESKTOP-715IF4P
```

### Adding the laptop key
From the laptop (`mattik01`), run:
```bash
# Option A: if phone is reachable via Tailscale
ssh-copy-id -p 8022 u0_a317@100.71.183.60

# Option B: if on same WiFi
ssh-copy-id -p 8022 u0_a317@192.168.68.51
```

---

## WoL Script (`~/wol_gpu.py` on phone)

Sends magic packets to both the direct IP and broadcast address:
```python
import socket

MAC = "D8:43:AE:B3:D7:3D"
TARGETS = ["192.168.68.58", "255.255.255.255"]
PORT = 9

def send_wol(mac, targets=TARGETS, port=PORT):
    mac_bytes = bytes.fromhex(mac.replace(":", "").replace("-", ""))
    magic = b"\xff" * 6 + mac_bytes * 16
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for target in targets:
            s.sendto(magic, (target, port))
            print(f"WoL magic packet sent to {mac} via {target}:{port}")

if __name__ == "__main__":
    send_wol(MAC)
```

**Note:** Broadcast to `192.168.68.255` and `192.168.71.255` did NOT work from Termux. Only direct IP (`192.168.68.58`) and `255.255.255.255` are received by the PC. Tested and confirmed 2026-03-11.

---

## Initial Setup (already done)

### On the phone (Termux)
```bash
pkg update && pkg install openssh python -y
passwd                    # set a password
sshd                      # start SSH server
termux-wake-lock          # prevent Android from killing Termux
mkdir -p ~/.ssh
# Copy PC's public key into ~/.ssh/authorized_keys
```

### On the PC
```bash
# Install ADB platform-tools (already at C:\Users\glaes\platform-tools\)
adb forward tcp:8022 tcp:8022
ssh-copy-id -p 8022 u0_a317@localhost   # or manually copy key
```

### Tailscale
- Installed as Android app (Play Store) on the phone
- Account: `glaeser.matteo@`
- All three devices on the same Tailscale network:
  - `galaxy-s10` — `100.71.183.60`
  - `desktop-715if4p` — `100.85.86.56`
  - `mattik01` — `100.117.90.113`

---

## After Phone Reboot

Termux services don't survive a reboot. Re-run manually:
```bash
sshd
termux-wake-lock
```

Or open Termux and run both commands. Tailscale (Android app) should reconnect automatically.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `adb devices` shows "unauthorized" | Tap "Allow USB debugging" on the phone screen |
| SSH connection refused | Run `sshd` in Termux |
| Port 8022 not forwarded (USB) | Run `adb forward tcp:8022 tcp:8022` |
| WoL not received by PC | Check that `wol_gpu.py` sends to `192.168.68.58` (direct IP), not just broadcast |
| Phone WiFi IP changed | Check with `ssh termux "ip addr show wlan0"` and update `~/.ssh/config` |
| Tailscale not connecting | Open Tailscale app on phone, ensure it's enabled |

---

## Why a Phone Relay?

TP-Link Deco mesh routers don't support static ARP entries. When the GPU PC is off, its ARP entry expires (~1-2 min), and the router can't deliver WoL packets forwarded from the internet. An always-on LAN device (the phone) sends the WoL broadcast locally, which always works regardless of ARP state. See `Master/temp/wol_test_resume.md` for the full root cause analysis.
