#!/bin/bash
# ClawBrain Remote Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/clawcolab/clawbrain/main/remote-install.sh | bash

set -e

echo "🧠 ClawBrain Remote Installer"
echo "=============================="

# Detect skills directory
if [ -d "$HOME/.openclaw/skills" ]; then
    SKILLS_DIR="$HOME/.openclaw/skills"
elif [ -d "$HOME/.clawdbot/skills" ]; then
    SKILLS_DIR="$HOME/.clawdbot/skills"
elif [ -d "$HOME/clawd/skills" ]; then
    SKILLS_DIR="$HOME/clawd/skills"
else
    # Create default
    mkdir -p "$HOME/.openclaw/skills"
    SKILLS_DIR="$HOME/.openclaw/skills"
fi

echo "📁 Installing to: $SKILLS_DIR/clawbrain"

# Clone or update
if [ -d "$SKILLS_DIR/clawbrain" ]; then
    echo "📥 Updating existing installation..."
    cd "$SKILLS_DIR/clawbrain"
    git pull
else
    echo "📥 Cloning clawbrain..."
    git clone https://github.com/clawcolab/clawbrain.git "$SKILLS_DIR/clawbrain"
    cd "$SKILLS_DIR/clawbrain"
fi

# Run local installer
chmod +x install.sh
./install.sh
