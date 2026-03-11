# WoL & Sleep Fix — Session Resume Notes

## What was done

### Sleep fix ✅
- Verified sleep type is S3 (traditional, no Modern Standby)
- Disabled sleep timeout (AC) on ALL three power plans:
  - Balanced (`381b4222-f694-41f0-9685-ff5bb260df2e`)
  - High Performance (`8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`)
  - Power Saver (`a1841308-3541-4fab-bc81-f71556f20b4a`)
- SSH service (`sshd`) confirmed start type = `Automatic` ✅
- Hibernate already off ✅

### Wake-on-LAN port forwarding ✅
- Deco app: NAT-Weiterleitung rule added — UDP port 9 → 192.168.68.58
- DHCP address reservation: MAC `D8-43-AE-B3-D7-3D` → IP `192.168.68.58`
  - Note: app shows stale name "iPad von Luca" — cosmetic only, MAC is correct

### WoL listener test — NOT YET CONFIRMED
- Wrote a Python UDP listener on port 9, but received nothing
- Likely cause: Windows Firewall blocking inbound UDP port 9 (couldn't check — needs admin)
- **This doesn't matter for real WoL** — when the PC is off, the NIC handles WoL at hardware level, bypassing the OS firewall

## Root Cause Analysis (2026-03-11)

### Why external WoL likely won't work with current setup

**The core problem: ARP table expiration on TP-Link Deco**

1. Port forwarding on the Deco routes UDP port 9 → `192.168.68.58`
2. But when the PC is **off**, it has no IP address — the Deco's ARP table entry for `192.168.68.58` expires (typically within 1-2 minutes after shutdown)
3. Without an ARP entry, the router doesn't know which MAC address to deliver the packet to → packet is dropped
4. **DHCP reservation ≠ static ARP** — reservation only assigns the IP when the PC boots and requests one via DHCP. It does NOT keep the ARP mapping alive while the PC is off.

**Confirmed from research:**
- TP-Link Deco routers do NOT support static ARP entries (limited consumer UI)
- The Deco app's "Address Reservation" is DHCP only, not ARP binding
- No hidden settings, SSH, or workarounds available on Deco firmware
- This is a known limitation discussed extensively on r/TpLink and r/HomeNetworking

### What would fix it (but Deco can't do)
- **Static ARP entry** on the router (binds MAC to IP permanently)
- **Port forward to broadcast address** `192.168.68.255` (Deco doesn't support this either)
- **ISP router** doing the forwarding instead (if accessible)

## Solution: Always-on LAN device as WoL relay

Since the Deco can't solve this at the router level, an always-on device on the LAN is needed to send the WoL magic packet locally (broadcast on LAN always works).

### Option A: Old Android phone (cheapest — free)
- Keep plugged in 24/7 on home WiFi
- Install **Termux** (from F-Droid, not Play Store)
  ```bash
  pkg install wakeonlan python openssh
  ```
- Run a simple HTTP server that triggers WoL:
  ```python
  from http.server import HTTPServer, BaseHTTPRequestHandler
  import subprocess
  class H(BaseHTTPRequestHandler):
      def do_GET(self):
          subprocess.run(['wakeonlan', 'D8-43-AE-B3-D7-3D'])
          self.send_response(200)
          self.end_headers()
          self.wfile.write(b'WoL sent')
  HTTPServer(('0.0.0.0', 8080), H).serve_forever()
  ```
- Access via Tailscale or port forward to phone's IP:8080
- **Caveat:** Disable battery optimization for Termux so Android doesn't kill it
- Alternative apps: Tasker (~€3.50) or MacroDroid (free) with HTTP trigger + WoL plugin

### Option B: Raspberry Pi Zero 2 W (~€30-35 total)
- Pi Zero 2 W: ~€15-20
- Micro SD card: ~€5
- USB-C power supply: ~€10
- Install Raspberry Pi OS Lite, set up SSH + `wakeonlan` + Tailscale
- Most reliable long-term solution

### Option C: Old PC/motherboard (likely too expensive)
- Would need CPU + RAM + PSU — buying these exceeds Raspberry Pi cost
- Only viable if all parts are already available

## What to do next

### 1. Choose and set up the always-on relay device
- **Recommended:** Old Android phone if available (zero cost)
- **Otherwise:** Raspberry Pi Zero 2 W (~€30-35)

### 2. Set up remote access to the relay
- Install Tailscale on relay device + personal phone
- Or: port forward one port to the relay device on the Deco

### 3. Test the full WoL chain
1. Shut down the PC
2. Wait >5 minutes (let ARP expire)
3. From outside the network: trigger the relay device
4. Relay sends WoL broadcast on LAN → PC wakes up
5. SSH in to confirm

### 4. Local WoL still worth testing
- From a device on the same LAN (phone on WiFi), sending WoL to broadcast `192.168.68.255` should work — this bypasses the ARP problem entirely
- Good sanity check that the PC's WoL hardware is working

## Key info
- **PC hostname:** DESKTOP-715IF4P
- **LAN IP:** 192.168.68.58
- **MAC:** D8-43-AE-B3-D7-3D
- **Public IP:** 95.91.238.61 (check with `curl ifconfig.me` — may be dynamic)
- **Router:** TP-Link Deco mesh, managed via TP-Link Deco app (father's account)
- **UDP port 9:** Confirmed listening on `0.0.0.0:9` when PC is on
