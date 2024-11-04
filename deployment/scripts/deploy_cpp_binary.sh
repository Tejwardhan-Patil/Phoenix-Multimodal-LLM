#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Variables
BINARY_PATH="/cpp/binary"
DEPLOY_DIR="/opt/deployed_binaries"
SERVICE_NAME="cpp_model_service"
SYSTEMD_PATH="/systemd/system"
LOG_DIR="/var/log/${SERVICE_NAME}"

# Create deployment directory if it doesn't exist
if [ ! -d "$DEPLOY_DIR" ]; then
    mkdir -p "$DEPLOY_DIR"
fi

# Copy the binary to the deployment directory
cp "$BINARY_PATH" "$DEPLOY_DIR/"

# Ensure the binary is executable
chmod +x "$DEPLOY_DIR/$(basename "$BINARY_PATH")"

# Create log directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# Create a systemd service file for the C++ binary
cat <<EOL > "$SYSTEMD_PATH/${SERVICE_NAME}.service"
[Unit]
Description=C++ Model Service
After=network.target

[Service]
ExecStart=$DEPLOY_DIR/$(basename "$BINARY_PATH")
WorkingDirectory=$DEPLOY_DIR
StandardOutput=append:$LOG_DIR/output.log
StandardError=append:$LOG_DIR/error.log
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOL

# Reload systemd to apply the new service
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable "$SERVICE_NAME"

# Start the service
systemctl start "$SERVICE_NAME"

echo "Deployment of C++ binary completed successfully."