#!/bin/bash
# Basic setup script for Docker Hub installation

# Create necessary directories
mkdir -p uploads chroma_db storage .streamlit logs

# Create basic .streamlit config
mkdir -p .streamlit
cat > .streamlit/config.toml << 'CONFIG_EOF'
[server]
address = "localhost"
port = 8501

[browser]
serverAddress = "localhost"
CONFIG_EOF

# Ensure a .env exists (seed from .env.example) so docker-compose has a place to
# read variable substitutions from.
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "📝 Created .env from .env.example"
fi

# Auto-detect the host LAN IP and write it into .env so realtime voice
# (WebRTC / coturn) works for LAN access without manual editing.
if command -v python3 >/dev/null 2>&1; then
    PYBIN=python3
elif command -v python >/dev/null 2>&1; then
    PYBIN=python
else
    PYBIN=""
fi

if [ -n "$PYBIN" ]; then
    "$PYBIN" - << 'PY'
import os
import re
import socket
import subprocess
import sys


def get_ifconfig_ips():
    ips = {}
    try:
        out = subprocess.check_output(["ifconfig"], stderr=subprocess.DEVNULL, text=True)
        current_iface = None
        for line in out.splitlines():
            if line and not line[0].isspace():
                current_iface = line.split(":")[0].strip()
            elif line.strip().startswith("inet "):
                parts = line.strip().split()
                if len(parts) > 1:
                    ip = parts[1]
                    if ip.startswith("addr:"):
                        ip = ip[5:]
                    if current_iface:
                        ips[ip] = current_iface
    except Exception:
        pass
    return ips


def get_ip_addr_ips():
    ips = {}
    try:
        out = subprocess.check_output(["ip", "addr"], stderr=subprocess.DEVNULL, text=True)
        current_iface = None
        for line in out.splitlines():
            m = re.match(r"^\d+:\s+([^:]+):", line)
            if m:
                current_iface = m.group(1).strip()
            elif "inet " in line:
                parts = line.strip().split()
                for p in parts:
                    if p.startswith("inet"):
                        continue
                    ip = p.split("/")[0]
                    if current_iface:
                        ips[ip] = current_iface
                    break
    except Exception:
        pass
    return ips


def get_udp_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def get_hostname_ips():
    try:
        return socket.gethostbyname_ex(socket.gethostname())[2]
    except Exception:
        return []


def is_ignored_interface(iface):
    iface = iface.lower()
    for prefix in ["lo", "tun", "tap", "utun", "ppp", "gif", "stf", "docker", "veth", "br-"]:
        if iface.startswith(prefix):
            return True
    return False


def is_physical_interface(iface):
    iface = iface.lower()
    for prefix in ["en", "eth", "wlan", "wifi"]:
        if iface.startswith(prefix):
            return True
    return False


def detect_lan_ip():
    """Return the primary physical LAN IP, ignoring loopbacks, Docker and VPN interfaces.

    Falls back to hostname IP or UDP connect IP if no physical interface is active.
    """
    ifconfig_map = get_ifconfig_ips()
    ip_addr_map = get_ip_addr_ips()

    iface_map = {}
    iface_map.update(ifconfig_map)
    iface_map.update(ip_addr_map)

    hostname_ips = get_hostname_ips()
    udp_ip = get_udp_ip()

    all_ips = set(iface_map.keys()) | set(hostname_ips)
    if udp_ip:
        all_ips.add(udp_ip)

    physical_ips = []
    other_valid_ips = []
    vpn_ips = []

    for ip in all_ips:
        if ip.startswith("127.") or ip.startswith("169.254."):
            continue
        iface = iface_map.get(ip)
        if iface:
            if is_ignored_interface(iface):
                vpn_ips.append((ip, iface))
            elif is_physical_interface(iface):
                physical_ips.append((ip, iface))
            else:
                other_valid_ips.append((ip, iface))
        else:
            other_valid_ips.append((ip, "unknown"))

    if physical_ips:
        return physical_ips[0][0]
    if other_valid_ips:
        for ip, iface in other_valid_ips:
            if ip in hostname_ips:
                return ip
        return other_valid_ips[0][0]
    if vpn_ips:
        for ip, iface in vpn_ips:
            if ip == udp_ip:
                return ip
        return vpn_ips[0][0]

    return udp_ip or None


env_path = ".env"

ip = detect_lan_ip()
if not ip:
    print("⚠️  Could not auto-detect LAN IP; leaving HOST_LAN_IP at its default (127.0.0.1).")
    sys.exit(0)

lines = []
if os.path.exists(env_path):
    with open(env_path) as f:
        lines = f.read().splitlines()

# Find an existing, non-commented HOST_LAN_IP assignment.
current = None
idx = None
for i, line in enumerate(lines):
    if line.lstrip().startswith("#"):
        continue
    m = re.match(r"\s*HOST_LAN_IP\s*=\s*(.*)\s*$", line)
    if m:
        current = m.group(1).strip()
        idx = i
        break

# Only write when the value is missing, empty, or still the loopback placeholder,
# so we never clobber an IP the user set deliberately.
if current not in (None, "", "127.0.0.1"):
    print(f"✓ HOST_LAN_IP already set to {current}; leaving unchanged.")
    sys.exit(0)

new_line = f"HOST_LAN_IP={ip}"
if idx is not None:
    lines[idx] = new_line
else:
    lines.append(new_line)

with open(env_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"🌐 Set HOST_LAN_IP={ip} in .env for realtime voice (WebRTC) LAN access.")
PY
fi

echo "✅ Basic setup completed"
