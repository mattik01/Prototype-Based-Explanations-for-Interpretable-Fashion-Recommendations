# Remote Machine Setup — Desktop (mattik01)

## Static IP Configuration
- **IP:** `192.168.68.58` (static, DHCP disabled)
- **Subnet:** `/24` (255.255.255.0)
- **Gateway:** `192.168.68.1` (Deco mesh router)
- **DNS:** `1.1.1.1`, `8.8.8.8`
- **Interface:** Ethernet

### Commands Used
```powershell
Set-NetIPInterface -InterfaceAlias "Ethernet" -Dhcp Disabled
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.68.58 -PrefixLength 24 -DefaultGateway 192.168.68.1
Set-DnsClientServerAddress -InterfaceAlias "Ethernet" -ServerAddresses 1.1.1.1,8.8.8.8
```

### Verify
```powershell
Get-NetIPAddress -InterfaceAlias "Ethernet" -AddressFamily IPv4 | Select-Object IPAddress, PrefixLength, AddressState
Get-NetIPInterface -InterfaceAlias "Ethernet" -AddressFamily IPv4 | Select-Object Dhcp
Get-DnsClientServerAddress -InterfaceAlias "Ethernet" -AddressFamily IPv4 | Select-Object ServerAddresses
```

## Tailscale
- **Tailscale IP:** `100.85.86.56`
- Allows SSH from anywhere (not just local network)
- Works through double NAT (Deco mesh behind Vodafone router)
- Should run as service on boot (no login required)

### SSH via Tailscale (from any machine on same Tailscale account)
```bash
ssh glaes@100.85.86.56
```

### SSH via LAN (local network only)
```bash
ssh glaes@192.168.68.58
```

## BIOS Settings (MSI PRO B650-S WIFI)
Access BIOS with `Del` at boot.

- **Settings → Advanced → Wake Up Event Setup**
  - Resume by PCI-E Device → `Enabled` (WOL)
- **Settings → Advanced → Power Management Setup**
  - Restore after AC Power Loss → `Power On`
  - ErP Ready → `Disabled`

## Windows Power Settings
```powershell
powercfg /hibernate off
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

## Post-Reboot Verification
```powershell
# Check SSH server
Get-Service sshd | Select-Object Status, StartType

# Check Tailscale
Get-Service Tailscale | Select-Object Status, StartType

# Check Tailscale IP
tailscale ip -4

# Test connectivity
Test-NetConnection -ComputerName google.com
```

## Motherboard
- **Manufacturer:** Micro-Star International Co., Ltd.
- **Model:** PRO B650-S WIFI (MS-7E26)
