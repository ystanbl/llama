#!/bin/bash
# Llama - Local AI Platform Installer
# Run as root or with sudo on a fresh Ubuntu 22.04+ system
#
# Usage:
#   sudo bash install.sh              # Full install
#   sudo bash install.sh --update     # Update app only (git pull + restart)
set -e

# ── Configuration ────────────────────────────────────────────────────────────
SERVICE_USER="llama"
HOME_DIR="/home/$SERVICE_USER"
REPO_DIR="$HOME_DIR/llama"
CHAT_DIR="$REPO_DIR/unified-chat"
SD_DIR="$HOME_DIR/stable-diffusion-webui"
SD_MODELS_DIR="/opt/sd-models"
REPO_URL="https://github.com/ystanbl/llama.git"

echo "=== Llama Local AI Platform Installer ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo bash install.sh"
    exit 1
fi

# ── Update mode ──────────────────────────────────────────────────────────────
if [ "$1" = "--update" ]; then
    echo "Updating from git..."
    sudo -u "$SERVICE_USER" bash -c "cd $REPO_DIR && git pull"
    # Re-install service files in case they changed
    cp "$REPO_DIR/config/systemd/unified-chat.service" /etc/systemd/system/
    cp "$REPO_DIR/config/systemd/sd-webui.service" /etc/systemd/system/
    cp "$REPO_DIR/config/systemd/ollama.service" /etc/systemd/system/
    cp "$REPO_DIR/config/telegraf/telegraf.conf" /etc/telegraf/telegraf.conf
    systemctl daemon-reload
    systemctl restart unified-chat
    echo "Update complete. Services restarted."
    exit 0
fi

# ── 1. System packages ──────────────────────────────────────────────────────
echo "[1/8] Installing system packages..."
apt update -qq
apt install -y -qq python3 python3-venv python3-pip \
    lilypond timidity fluid-soundfont-gm ffmpeg \
    influxdb telegraf grafana curl git openssl

# ── 2. Create service user ──────────────────────────────────────────────────
echo "[2/8] Creating service user '$SERVICE_USER'..."
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd -r -m -s /bin/bash "$SERVICE_USER"
    echo "  User '$SERVICE_USER' created"
else
    echo "  User '$SERVICE_USER' already exists"
fi

# ── 3. Clone repository ─────────────────────────────────────────────────────
echo "[3/8] Setting up repository..."
if [ ! -d "$REPO_DIR/.git" ]; then
    sudo -u "$SERVICE_USER" git clone "$REPO_URL" "$REPO_DIR"
    echo "  Cloned to $REPO_DIR"
else
    sudo -u "$SERVICE_USER" bash -c "cd $REPO_DIR && git pull"
    echo "  Repository updated"
fi

# ── 4. Install Ollama ───────────────────────────────────────────────────────
echo "[4/8] Installing Ollama..."
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "  Ollama already installed"
fi

cp "$REPO_DIR/config/systemd/ollama.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now ollama
echo "  Pulling mistral model (this may take a while)..."
sudo -u ollama ollama pull mistral || ollama pull mistral

# ── 5. Install Stable Diffusion WebUI ───────────────────────────────────────
echo "[5/8] Installing Stable Diffusion WebUI..."
if [ ! -d "$SD_DIR" ]; then
    sudo -u "$SERVICE_USER" git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git "$SD_DIR"
    cd "$SD_DIR"
    sudo -u "$SERVICE_USER" bash webui.sh --skip-torch-cuda-test --use-cpu all --no-half --exit
    # Fix CLIP if needed
    if [ -d "$SD_DIR/venv" ]; then
        sudo -u "$SERVICE_USER" "$SD_DIR/venv/bin/pip" install wheel 2>/dev/null || true
        sudo -u "$SERVICE_USER" "$SD_DIR/venv/bin/pip" install --no-build-isolation \
            git+https://github.com/openai/CLIP.git 2>/dev/null || true
    fi
else
    echo "  SD WebUI already installed at $SD_DIR"
fi

# Set up shared models directory
mkdir -p "$SD_MODELS_DIR"
chmod 777 "$SD_MODELS_DIR"
# Symlink SD WebUI's model directory to the shared location
SD_MODELS_LINK="$SD_DIR/models/Stable-diffusion"
if [ -d "$SD_MODELS_LINK" ] && [ ! -L "$SD_MODELS_LINK" ]; then
    # Move any existing models to shared dir, then replace with symlink
    mv "$SD_MODELS_LINK"/* "$SD_MODELS_DIR/" 2>/dev/null || true
    rm -rf "$SD_MODELS_LINK"
fi
if [ ! -L "$SD_MODELS_LINK" ]; then
    ln -sfn "$SD_MODELS_DIR" "$SD_MODELS_LINK"
fi

cp "$REPO_DIR/config/systemd/sd-webui.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now sd-webui

# ── 6. Install Unified Chat ────────────────────────────────────────────────
echo "[6/8] Installing Unified Chat..."

# Create Python venv if needed
if [ ! -d "$CHAT_DIR/venv" ]; then
    sudo -u "$SERVICE_USER" python3 -m venv "$CHAT_DIR/venv"
fi
sudo -u "$SERVICE_USER" "$CHAT_DIR/venv/bin/pip" install -q -r "$CHAT_DIR/requirements.txt"

# Generate self-signed SSL cert if needed
if [ ! -f "$CHAT_DIR/cert.pem" ]; then
    echo "  Generating self-signed SSL certificate..."
    HOSTNAME_VAL=$(hostname)
    sudo -u "$SERVICE_USER" openssl req -x509 -newkey rsa:2048 \
        -keyout "$CHAT_DIR/key.pem" -out "$CHAT_DIR/cert.pem" \
        -days 365 -nodes -subj "/CN=$HOSTNAME_VAL" 2>/dev/null
fi

# Create music output directory
mkdir -p "$CHAT_DIR/music_output"
chown -R "$SERVICE_USER:$SERVICE_USER" "$CHAT_DIR"

cp "$REPO_DIR/config/systemd/unified-chat.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now unified-chat

# ── 7. Setup monitoring stack ───────────────────────────────────────────────
echo "[7/8] Setting up monitoring stack..."
systemctl enable --now influxdb

# Create InfluxDB database and retention policy
influx -execute "CREATE DATABASE telegraf" 2>/dev/null || true
influx -execute "ALTER RETENTION POLICY autogen ON telegraf DURATION 1d REPLICATION 1 DEFAULT" 2>/dev/null || true

cp "$REPO_DIR/config/telegraf/telegraf.conf" /etc/telegraf/telegraf.conf
systemctl enable --now telegraf

systemctl enable --now grafana-server

# Configure Grafana anonymous access
sed -i '/\[auth.anonymous\]/,/^$/ { s/enabled = false/enabled = true/; s/org_role = Viewer/org_role = Admin/ }' /etc/grafana/grafana.ini
systemctl restart grafana-server

# Wait for Grafana to start, then import dashboard
sleep 3
curl -s -X POST http://admin:admin@localhost:3000/api/dashboards/db \
    -H "Content-Type: application/json" \
    -d @"$REPO_DIR/config/grafana/dashboard.json" >/dev/null 2>&1 || true

# ── 8. Sudoers for service restarts from web UI ────────────────────────────
echo "[8/8] Configuring sudoers for web-based service management..."
cat > /etc/sudoers.d/unified-chat << SUDOERS
$SERVICE_USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart sd-webui, /usr/bin/systemctl restart ollama, /usr/bin/systemctl restart unified-chat
SUDOERS
chmod 0440 /etc/sudoers.d/unified-chat

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installation Complete ==="
echo ""
IP=$(hostname -I | awk '{print $1}')
echo "Services:"
echo "  Chat Interface:   https://$IP (self-signed cert)"
echo "  Task Status:      https://$IP/status"
echo "  Grafana:          http://$IP:3000"
echo "  Ollama API:       http://$IP:11434"
echo "  SD WebUI API:     http://$IP:7860"
echo ""
echo "To update later:"
echo "  sudo bash $REPO_DIR/install.sh --update"
echo ""
echo "Or manually:"
echo "  cd $REPO_DIR && sudo -u $SERVICE_USER git pull"
echo "  sudo systemctl restart unified-chat"
