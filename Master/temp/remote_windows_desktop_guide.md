# Complete Guide: Remote Access Setup for a Windows Desktop PC

This guide walks you through turning a Windows desktop into a fully remote-accessible workstation — wake it up, SSH in, run GPU workloads, and shut it down — all from anywhere.

---

## Table of Contents

1. [Prerequisites and Planning](#1-prerequisites-and-planning)
2. [Wake-on-LAN (WoL)](#2-wake-on-lan-wol)
3. [SSH Server Setup on Windows](#3-ssh-server-setup-on-windows)
4. [Using Computational Resources Remotely](#4-using-computational-resources-remotely)
5. [Security Hardening](#5-security-hardening)
6. [Remote Shutdown](#6-remote-shutdown)
7. [Bonus: Tailscale for Simplified Networking](#7-bonus-tailscale-for-simplified-networking)
8. [Troubleshooting Common Issues](#8-troubleshooting-common-issues)

---

## 1. Prerequisites and Planning

**What you need:**
- A Windows 10/11 desktop PC (Pro or Home both work for SSH; Pro needed for RDP)
- Ethernet connection (WoL is unreliable or unsupported over Wi-Fi)
- Access to your router's admin panel
- A second device to connect from (laptop, phone, another PC)

**Network planning — do this first:**
- Log into your router (typically `192.168.1.1` or `192.168.0.1`)
- Note your PC's current local IP (e.g., `192.168.1.50`)
- Note your public IP (visit `https://ifconfig.me` from any device on your network)
- Decide: will you use port forwarding or a VPN like Tailscale? (Tailscale is simpler and more secure — see Section 7)

---

## 2. Wake-on-LAN (WoL)

Wake-on-LAN lets you power on your PC by sending a "magic packet" to its network adapter. The PC must be shut down (S5 state) or sleeping (S3) — not powered off at the PSU.

### 2.1 BIOS/UEFI Settings

Restart your PC and enter BIOS/UEFI (usually `DEL`, `F2`, or `F12` during boot).

Look for these settings (names vary by motherboard manufacturer):

| Manufacturer | Setting Name | Location |
|---|---|---|
| ASUS | "Wake on LAN" or "Power On By PCI-E" | Advanced > APM Configuration |
| MSI | "Wake Up Event Setup" > "Resume By PCI-E Device" | Settings > Advanced > Wake Up Event Setup |
| Gigabyte | "Wake on LAN" or "PME Event Wake Up" | Settings > Platform Power (or Power Management) |
| ASRock | "PCIE Devices Power On" | Advanced > ACPI Configuration |

**Enable these:**
- Wake on LAN / Wake on PCI-E: **Enabled**
- ERP Ready / ErP / Deep Sleep: **Disabled** (this cuts power to the NIC and kills WoL)
- If present, "Wake on Magic Packet": **Enabled**

Save and exit BIOS.

### 2.2 Windows Network Adapter Settings

**Open Device Manager:**
```
Win + X > Device Manager > Network Adapters > right-click your Ethernet adapter > Properties
```

**Power Management tab:**
- [x] Allow the computer to turn off this device to save power
- [x] Allow this device to wake the computer
- [x] Only allow a magic packet to wake the computer

**Advanced tab** — find and set these properties (not all adapters have all of them):
- "Wake on Magic Packet": **Enabled**
- "Wake on Pattern Match": **Disabled** (optional, reduces false wakes)
- "Energy Efficient Ethernet": **Disabled** (can interfere with WoL)
- "Green Ethernet": **Disabled**
- "Shutdown Wake-On-LAN": **Enabled** (critical — allows wake from full shutdown, not just sleep)

### 2.3 Disable Fast Startup

Fast Startup (hybrid shutdown) puts the PC in a hibernation-like state that can break WoL on many systems.

```powershell
# Run PowerShell as Administrator
powercfg /h off
```

Or manually: **Control Panel > Power Options > Choose what the power buttons do > Change settings that are currently unavailable** > uncheck "Turn on fast startup".

### 2.4 Find Your MAC Address

```powershell
# In PowerShell or CMD
ipconfig /all
```

Look for your Ethernet adapter's **Physical Address**, e.g., `A8-B8-E0-12-34-56`. Write this down — you need it to send WoL packets.

Alternatively:
```powershell
Get-NetAdapter | Select-Object Name, MacAddress, Status
```

### 2.5 Set a Static IP or DHCP Reservation

Your PC needs a predictable local IP address. Two options:

**Option A: DHCP Reservation (recommended)** — configure in your router admin panel. Find the DHCP settings and reserve an IP for your PC's MAC address. This way the router always assigns the same IP.

**Option B: Static IP on Windows:**
```powershell
# PowerShell as Administrator
# Replace values with your network's settings
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.1.50 -PrefixLength 24 -DefaultGateway 192.168.1.1
Set-DnsClientServerAddress -InterfaceAlias "Ethernet" -ServerAddresses 1.1.1.1, 8.8.8.8
```

### 2.6 Sending WoL Packets

**From Linux (same network):**
```bash
# Install wakeonlan
sudo apt install wakeonlan

# Send magic packet
wakeonlan A8:B8:E0:12:34:56
```

**From Linux (with specific broadcast address):**
```bash
wakeonlan -i 192.168.1.255 A8:B8:E0:12:34:56
```

**From another Windows machine:**
```powershell
# Install the module
Install-Module -Name WakeOnLan -Force

# Send packet
Send-WolPacket -MacAddress A8:B8:E0:12:34:56 -BroadcastAddress 192.168.1.255
```

**From a phone:**
- Android: "Wake On Lan" app by Mike Webb (free on Play Store)
- iOS: "Mocha WOL" or "RemoteBoot WOL"

### 2.7 WoL Over the Internet (Outside Your Network)

For WoL from outside your local network, you need to forward the magic packet through your router.

**Router port forwarding:**
- External port: `9` (or any port you choose)
- Internal IP: `192.168.1.255` (the broadcast address of your subnet — NOT your PC's IP)
- Internal port: `9`
- Protocol: **UDP**

Some routers do not allow forwarding to a broadcast address. Workarounds:
- Forward to the PC's specific IP and set a static ARP entry on the router
- Use a device that is always on (Raspberry Pi, NAS) as a WoL relay on the LAN
- Use Tailscale (Section 7) which avoids this problem entirely

**Sending WoL over the internet:**
```bash
wakeonlan -i YOUR_PUBLIC_IP -p 9 A8:B8:E0:12:34:56
```

**Important:** Your public IP may change. Use a Dynamic DNS service (No-IP, DuckDNS) to get a stable hostname like `mypc.duckdns.org`.

Setting up DuckDNS (free):
1. Go to `https://www.duckdns.org`, sign in, create a subdomain
2. On your always-on device or router, set up periodic updates:
```bash
# Cron job on a Linux machine or Raspberry Pi on the same network
echo url="https://www.duckdns.org/update?domains=YOURSUBDOMAIN&token=YOUR_TOKEN&ip=" | curl -k -o /dev/null -s -K -
```

---

## 3. SSH Server Setup on Windows

### 3.1 Install OpenSSH Server

**Windows 10 (1809+) and Windows 11:**

```powershell
# PowerShell as Administrator

# Check if already installed
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH*'

# Install client and server
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
```

**Alternative via Settings:**
Settings > Apps > Optional Features > Add a feature > search "OpenSSH Server" > Install

### 3.2 Start and Enable the SSH Service

```powershell
# Start the service
Start-Service sshd

# Set to start automatically on boot
Set-Service -Name sshd -StartupType 'Automatic'

# Verify it's running
Get-Service sshd
```

### 3.2.1 What Happens on Boot (No Login Required)

When the PC boots via WoL, it lands on the **Windows lock screen** — no user is logged in. This is fine:

- `sshd` runs as a **Windows service**, which starts automatically at boot regardless of whether anyone is logged in at the desktop
- You authenticate via your SSH key and get a full PowerShell session as your user
- No need to "unlock" the desktop or type a Windows password at the physical screen
- GPU, file system, and conda environments are all accessible from the SSH session

**The boot-to-SSH flow:**
1. WoL wakes the PC
2. Windows boots to the lock screen (no interactive user session)
3. `sshd` service starts automatically
4. You SSH in with your key — full access

**Optional: Auto-login (only if needed)**

Some programs or environment variables are only available when a user is "interactively" logged in. If you run into this, you can enable auto-login so your user is always signed in after boot:

```powershell
# Run on the Windows PC (as admin)
netplwiz
```

Uncheck "Users must enter a user name and password to use this computer" and enter your credentials. The PC will then boot straight to the desktop without prompting for a password.

**Note:** Auto-login is a minor security trade-off — anyone with physical access to the PC can use it. For a home desktop this is usually acceptable.

### 3.2.2 Display Behavior After WoL Boot

WoL boots the PC normally, so the **monitor will turn on** (or wake from standby) and show the lock screen. This doesn't affect SSH at all, but if you want the screen to stay dark:

**Easiest: Set a short display timeout (recommended)**

The monitor will turn on briefly during boot, then go to sleep on its own:
```powershell
# Turn off display after 1 minute on AC power
powercfg /change monitor-timeout-ac 1
```
Or via Settings > System > Power & battery > Screen and sleep > "Turn off my screen after" → 1 minute.

**Force display off via SSH after connecting:**
```powershell
(Add-Type '[DllImport("user32.dll")] public static extern int SendMessage(int hWnd, int hMsg, int wParam, int lParam);' -Name a -Pas)::SendMessage(-1, 0x0112, 0xF170, 2)
```
The display will wake again on local mouse/keyboard input, but SSH sessions are unaffected.

**Other options:**
- Turn the monitor off physically (power button) — has no effect on SSH or GPU
- Disconnect the display cable — Windows still works headless, GPU remains accessible for CUDA workloads

### 3.3 Windows Firewall Rule

The installer usually creates this automatically, but verify:

```powershell
# Check existing rule
Get-NetFirewallRule -Name *ssh*

# If no rule exists, create one
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

### 3.4 Configure sshd_config

The config file is at `C:\ProgramData\ssh\sshd_config`.

```powershell
# Open in notepad (as admin)
notepad C:\ProgramData\ssh\sshd_config
```

Key settings to review and modify:

```
# Change port if desired (see Security section)
#Port 22

# Enable public key authentication
PubkeyAuthentication yes

# Disable password auth AFTER setting up keys (see below)
#PasswordAuthentication no

# Ensure these are set for Windows
StrictModes no

# Subsystem for SFTP
Subsystem sftp sftp-server.exe
```

**Critical for admin users:** At the bottom of `sshd_config`, there are two lines that override authorized_keys for users in the Administrators group. Comment them out so your keys work:

```
# Comment out these two lines (add # in front):
#Match Group administrators
#       AuthorizedKeysFile __PROGRAMDATA__/ssh/administrators_authorized_keys
```

After editing, restart the service:
```powershell
Restart-Service sshd
```

### 3.5 Set Up SSH Keys

**On your client machine (Linux/Mac):**
```bash
# Generate a key pair if you don't have one
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy your public key to the Windows machine
# Option 1: manually (since ssh-copy-id doesn't work well with Windows)
cat ~/.ssh/id_ed25519.pub
```

**On the Windows machine**, create the authorized_keys file:
```powershell
# Create .ssh directory in your user profile
mkdir $env:USERPROFILE\.ssh -Force

# Create or append to authorized_keys (paste your public key)
notepad $env:USERPROFILE\.ssh\authorized_keys
```

Paste your public key (the content of `id_ed25519.pub`) into the file. Save.

**Fix permissions on authorized_keys** (Windows is strict about this):
```powershell
# Remove inheritance and set correct permissions
icacls $env:USERPROFILE\.ssh\authorized_keys /inheritance:r /grant "$($env:USERNAME):F" /grant "SYSTEM:F"
```

**Test the connection from your client:**
```bash
ssh your_windows_username@192.168.1.50
```

If key auth works, you can then disable password auth in `sshd_config` (set `PasswordAuthentication no`) and restart sshd.

### 3.6 Port Forwarding for External SSH Access

In your router admin panel:
- External port: `22` (or a non-standard port like `2222` — see Security section)
- Internal IP: `192.168.1.50` (your PC's static IP)
- Internal port: `22`
- Protocol: **TCP**

**Connect from outside your network:**
```bash
ssh -p 22 your_windows_username@mypc.duckdns.org
```

### 3.7 SSH Config File for Convenience

On your client machine, create `~/.ssh/config`:

```
Host mypc
    HostName mypc.duckdns.org  # or your public IP
    Port 22
    User your_windows_username
    IdentityFile ~/.ssh/id_ed25519
```

Now you can just type:
```bash
ssh mypc
```

---

## 4. Using Computational Resources Remotely

### 4.1 Running GPU Workloads via SSH

Once connected via SSH, you land in a PowerShell (or CMD) session. CUDA and GPU access work normally.

```powershell
# Check GPU status
nvidia-smi

# Run a Python training script
python train.py --epochs 100 --batch-size 64

# Check GPU utilization during training
nvidia-smi -l 5  # refresh every 5 seconds
```

**Setting up your Python environment:**
```powershell
# If using conda
conda activate myenv
python train.py

# If using venv
.\myenv\Scripts\Activate.ps1
python train.py
```

### 4.2 Running Long Tasks (Keeping Them Alive After Disconnect)

SSH sessions terminate running processes when you disconnect. Solutions:

**Option A: PowerShell background jobs (native Windows):**
```powershell
# Start a job that survives you closing the SSH session
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "train.py --epochs 100" -RedirectStandardOutput "output.log" -RedirectStandardError "error.log"
```

**Option B: `nohup` equivalent with `Start-Process`:**
```powershell
Start-Process python -ArgumentList "train.py" -WindowStyle Hidden -RedirectStandardOutput "C:\Users\you\train_out.log" -RedirectStandardError "C:\Users\you\train_err.log"
```

**Option C: Use WSL2 with tmux (best option for long-running tasks):**
```powershell
# Enter WSL from your SSH session
wsl

# Inside WSL, use tmux
tmux new -s training
python train.py --epochs 100

# Detach: Ctrl+B, then D
# Reconnect later:
tmux attach -t training
```

**Option D: Windows Terminal + screen in WSL:**
```bash
# In WSL
screen -S training
python train.py
# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

### 4.3 Using WSL2 Through SSH

You can enter WSL directly from your Windows SSH session:

```powershell
# List installed distros
wsl --list --verbose

# Enter default distro
wsl

# Enter specific distro
wsl -d Ubuntu-22.04

# Run a single command in WSL
wsl -e bash -c "cd /mnt/c/Users/you/project && python train.py"
```

**GPU access in WSL2:** If you have NVIDIA drivers installed on Windows, CUDA works inside WSL2 automatically (with Windows 11 or Windows 10 21H2+). Verify:
```bash
# Inside WSL
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### 4.4 File Transfer with SCP/SFTP

```bash
# Copy file to Windows PC
scp ./model.py mypc:C:/Users/you/project/

# Copy file from Windows PC
scp mypc:C:/Users/you/project/results.csv ./

# Copy entire directory
scp -r ./data/ mypc:C:/Users/you/project/data/

# Interactive SFTP session
sftp mypc
> put localfile.txt
> get remotefile.txt
> ls
> cd project
> exit
```

**For large transfers, use rsync over SSH (from WSL or Linux):**
```bash
rsync -avz --progress ./data/ mypc:/cygdrive/c/Users/you/project/data/
```

### 4.5 Monitoring System Resources

```powershell
# GPU monitoring
nvidia-smi
nvidia-smi dmon -s u  # continuous GPU utilization monitoring

# CPU, memory overview
systeminfo | findstr /C:"Total Physical Memory" /C:"Available Physical Memory"

# Process list (like top)
Get-Process | Sort-Object CPU -Descending | Select-Object -First 20 Name, CPU, WorkingSet

# Disk usage
Get-PSDrive -PSProvider FileSystem

# Task manager equivalent
tasklist /FI "MEMUSAGE gt 100000"
```

**In WSL, use familiar Linux tools:**
```bash
htop
watch -n 2 nvidia-smi
free -h
df -h
```

### 4.6 Remote GUI Access (RDP)

If you need GUI access (e.g., for debugging visualizations), use Remote Desktop Protocol. Requires Windows Pro.

```powershell
# Enable RDP via PowerShell (as admin)
Set-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server' -Name "fDenyTSConnections" -Value 0
Enable-NetFirewallRule -DisplayGroup "Remote Desktop"
```

**Connect from Linux:**
```bash
# Install an RDP client
sudo apt install remmina

# Or use xfreerdp
xfreerdp /u:your_username /v:192.168.1.50 /dynamic-resolution
```

**Connect from another Windows machine:** use the built-in "Remote Desktop Connection" app.

**Port forward for external RDP:** forward TCP `3389` on your router (or use Tailscale for secure access without port forwarding).

---

## 5. Security Hardening

### 5.1 SSH Key-Only Authentication

After confirming key auth works:

Edit `C:\ProgramData\ssh\sshd_config`:
```
PasswordAuthentication no
```

```powershell
Restart-Service sshd
```

### 5.2 Change Default SSH Port

Edit `C:\ProgramData\ssh\sshd_config`:
```
Port 2222
```

Update the firewall rule:
```powershell
# Remove old rule and add new one
Remove-NetFirewallRule -Name sshd
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 2222
Restart-Service sshd
```

Update your router port forwarding accordingly. Connect with:
```bash
ssh -p 2222 user@mypc.duckdns.org
```

### 5.3 Brute-Force Protection

Windows doesn't have `fail2ban`, but you have alternatives:

**Option A: Windows Firewall + Task Scheduler (built-in)**

Create a PowerShell script that monitors failed login attempts and blocks IPs:

```powershell
# Save as C:\Scripts\block-ssh-bruteforce.ps1
$threshold = 5
$events = Get-WinEvent -FilterHashtable @{LogName='OpenSSH/Operational'; Id=4} -ErrorAction SilentlyContinue |
    Where-Object { $_.TimeCreated -gt (Get-Date).AddHours(-1) }

$badIPs = $events |
    ForEach-Object { if ($_.Message -match 'from\s+(\d+\.\d+\.\d+\.\d+)') { $Matches[1] } } |
    Group-Object |
    Where-Object { $_.Count -ge $threshold }

foreach ($ip in $badIPs) {
    $ruleName = "Block-SSH-$($ip.Name)"
    if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Block -RemoteAddress $ip.Name
        Write-Output "Blocked $($ip.Name) after $($ip.Count) failed attempts"
    }
}
```

Schedule this to run every 15 minutes via Task Scheduler.

**Option B: Use `sshguard` in WSL** if your SSH connections route through WSL.

**Option C: Use Tailscale** (Section 7) and don't expose SSH to the internet at all.

### 5.4 Additional SSH Hardening

Add to `sshd_config`:
```
# Only allow specific users
AllowUsers your_username

# Disable root login (less relevant on Windows, but good practice)
PermitRootLogin no

# Limit authentication attempts
MaxAuthTries 3

# Disable empty passwords
PermitEmptyPasswords no

# Set idle timeout (disconnect after 10 min idle)
ClientAliveInterval 300
ClientAliveCountMax 2
```

### 5.5 Keep Everything Updated

```powershell
# Check Windows Update
Install-Module PSWindowsUpdate -Force
Get-WindowsUpdate
Install-WindowsUpdate -AcceptAll

# Check OpenSSH version
ssh -V
```

---

## 6. Remote Shutdown

### 6.1 Shutdown via SSH

```powershell
# Immediate shutdown
Stop-Computer -Force

# Shutdown with delay (seconds)
shutdown /s /t 60 /c "Remote shutdown in 60 seconds"

# Cancel a pending shutdown
shutdown /a

# Restart instead
Restart-Computer -Force
# or
shutdown /r /t 0

# Hibernate (saves state, uses less power than sleep, WoL still works on most systems)
shutdown /h
```

### 6.2 Scheduled Auto-Shutdown (Safety Net)

Prevent forgetting to shut down (wasting electricity / running up heat):

```powershell
# Create a scheduled task that shuts down at 2 AM daily if no user is active
$action = New-ScheduledTaskAction -Execute "shutdown.exe" -Argument "/s /t 300 /c `"Auto-shutdown in 5 minutes. Run 'shutdown /a' to cancel.`""
$trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName "AutoShutdown" -Action $action -Trigger $trigger -Settings $settings -User "SYSTEM"
```

**Smarter approach — only shutdown if GPU is idle:**

```powershell
# Save as C:\Scripts\smart-shutdown.ps1
$gpuUtil = (nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits).Trim()
if ([int]$gpuUtil -lt 5) {
    shutdown /s /t 300 /c "GPU idle - auto shutdown in 5 minutes. Run 'shutdown /a' to cancel."
}
```

### 6.3 Hibernate vs Shutdown

| Aspect | Shutdown | Hibernate |
|---|---|---|
| Power consumption | Zero | Zero |
| WoL works | Yes | Yes (usually) |
| Resume speed | Full boot (30-60s) | Fast (10-20s) |
| State preserved | No | Yes (RAM saved to disk) |
| Reliability for WoL | Best | Good (test first) |

**Recommendation:** Use full shutdown for reliability. Use hibernate only if you need to preserve session state and have tested that WoL works from hibernate on your specific hardware.

---

## 7. Bonus: Tailscale for Simplified Networking

Tailscale creates a peer-to-peer VPN mesh. It eliminates the need for port forwarding, dynamic DNS, and most firewall configuration. **This is the recommended approach for most users.**

### 7.1 Why Tailscale

- No port forwarding needed
- Works behind any NAT, including carrier-grade NAT (CGNAT)
- End-to-end encrypted (WireGuard under the hood)
- Free for personal use (up to 100 devices)
- Gives each device a stable IP (e.g., `100.x.y.z`)

### 7.2 Setup

**On the Windows PC:**
1. Download Tailscale from `https://tailscale.com/download`
2. Install and sign in (GitHub, Google, or Microsoft account)
3. Note the Tailscale IP assigned (e.g., `100.64.1.2`)

**On your Linux laptop/client:**
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

**On your phone:** Install the Tailscale app from your app store.

### 7.3 SSH Over Tailscale

Once both devices are on Tailscale, just SSH to the Tailscale IP:
```bash
ssh your_username@100.64.1.2
```

No port forwarding. No dynamic DNS. Works from anywhere in the world.

Update `~/.ssh/config`:
```
Host mypc
    HostName 100.64.1.2
    User your_windows_username
    IdentityFile ~/.ssh/id_ed25519
```

### 7.4 WoL via Tailscale Subnet Router

Tailscale can also send WoL packets if you set up a subnet router (a device that is always on, like a Raspberry Pi or your router if it supports Tailscale):

1. On the always-on device, enable subnet routing:
```bash
sudo tailscale up --advertise-routes=192.168.1.0/24
```
2. Approve the route in the Tailscale admin console
3. Send WoL from anywhere through the Tailscale network:
```bash
wakeonlan -i 192.168.1.255 A8:B8:E0:12:34:56
```

### 7.5 ZeroTier (Alternative)

ZeroTier is similar to Tailscale. Key differences:
- More configuration options (virtual network switches)
- Also free for personal use (up to 25 devices)
- Slightly more complex setup

```bash
# Install on Linux
curl -s https://install.zerotier.com | sudo bash
sudo zerotier-cli join <network-id>
```

For most users, Tailscale is simpler and recommended.

---

## 8. Troubleshooting Common Issues

### WoL Not Working

**Problem:** PC doesn't wake up when magic packet is sent.

**Checklist:**
1. **Fast Startup disabled?** Run `powercfg /h off` and verify in Power Options.
2. **BIOS settings correct?** ERP/ErP/Deep Sleep must be **disabled**. Wake on LAN/PCI-E must be **enabled**.
3. **Network adapter settings?** "Shutdown Wake-On-LAN" must be **enabled** in adapter Advanced properties.
4. **Using Ethernet?** WoL over Wi-Fi is unreliable on most consumer hardware. Use a wired connection.
5. **Correct MAC address?** Double-check with `ipconfig /all`.
6. **Is the NIC powered in shutdown?** Check if the Ethernet port's LED stays on after shutdown. If not, this is a BIOS power setting issue.
7. **Testing locally first?** Before trying over the internet, test WoL from another device on the same LAN.
8. **Power strip or UPS?** If the PC is on a smart power strip that cuts power completely, WoL cannot work.

**Diagnostic:**
```powershell
# Check if the adapter supports WoL
powercfg /devicequery wake_armed
# Your Ethernet adapter should appear in this list
```

### SSH Connection Refused

**Checklist:**
1. **Is sshd running?**
```powershell
Get-Service sshd
# If stopped:
Start-Service sshd
```
2. **Firewall rule exists?**
```powershell
Get-NetFirewallRule -Name *ssh* | Format-Table Name, Enabled, Action
```
3. **Listening on correct port?**
```powershell
netstat -an | findstr :22
# Should show LISTENING
```
4. **Port forwarding correct?** Verify in router. Test with:
```bash
# From external network
nc -zv mypc.duckdns.org 22
```
5. **Windows username correct?** Windows usernames are case-insensitive but SSH may behave differently. Use the exact username from `whoami`.
6. **Key permissions?** The `authorized_keys` file and `.ssh` directory must have correct permissions (see Section 3.5).

### SSH Connects But Key Auth Fails

```powershell
# Check the sshd log for details
Get-Content C:\ProgramData\ssh\logs\sshd.log -Tail 30

# If no log file, enable logging in sshd_config:
# SyslogFacility LOCAL0
# LogLevel DEBUG3
# Then restart sshd and retry
```

Common cause: the `administrators_authorized_keys` override at the bottom of `sshd_config`. Make sure those lines are commented out (see Section 3.4).

### GPU Not Accessible via SSH

**Problem:** `nvidia-smi` returns an error or CUDA is not available.

**Checklist:**
1. **NVIDIA drivers installed?** Run `nvidia-smi` in a local session first.
2. **PATH correct?** The SSH session may have a different PATH:
```powershell
$env:PATH += ";C:\Program Files\NVIDIA Corporation\NVSMI"
```
3. **Session 0 isolation:** GPU should be accessible from any session. If not, check that you're running the correct NVIDIA driver version (not the "DCH" vs "Standard" mismatch).
4. **WSL2 GPU:** If using GPU in WSL2 via SSH, ensure you have the WSL GPU driver (automatically included in recent NVIDIA Game Ready/Studio drivers).

### PowerShell vs CMD Default Shell

By default, OpenSSH on Windows drops you into CMD. To change to PowerShell:

```powershell
# Set PowerShell as default SSH shell
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force

# For PowerShell 7 (if installed)
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Program Files\PowerShell\7\pwsh.exe" -PropertyType String -Force
```

### Connection Drops During Long Sessions

Add to your client `~/.ssh/config`:
```
Host mypc
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

This sends a keepalive every 60 seconds.

---

## Quick Reference: The Full Workflow

Once everything is set up, your daily workflow looks like this:

```bash
# 1. Wake up the PC (from phone app or terminal)
wakeonlan A8:B8:E0:12:34:56

# 2. Wait ~30-60 seconds for boot

# 3. SSH in
ssh mypc

# 4. Start a tmux session (via WSL) for long tasks
wsl
tmux new -s work
conda activate myenv
python train.py --epochs 200
# Ctrl+B, D to detach

# 5. Check on it later
ssh mypc
wsl
tmux attach -t work

# 6. When done, shut down
Stop-Computer -Force
```

**With Tailscale**, steps 1-3 become even simpler since you don't need to worry about port forwarding or public IPs, and WoL can be routed through a subnet router.

---

## Summary Checklist

- [ ] BIOS: WoL enabled, ErP/Deep Sleep disabled
- [ ] Windows: Network adapter WoL settings configured
- [ ] Windows: Fast Startup disabled (`powercfg /h off`)
- [ ] Network: Static IP or DHCP reservation set
- [ ] OpenSSH Server installed and running
- [ ] SSH keys set up and tested
- [ ] `sshd_config`: admin authorized_keys override commented out
- [ ] Firewall: port 22 (or custom port) allowed
- [ ] Router: port forwarding configured (or Tailscale installed)
- [ ] Default shell set to PowerShell
- [ ] WoL tested from local network
- [ ] SSH tested from external network
- [ ] GPU access verified via SSH (`nvidia-smi`)
- [ ] Long-running task strategy decided (tmux via WSL recommended)
- [ ] Auto-shutdown safety net scheduled
- [ ] Password authentication disabled (key-only)
