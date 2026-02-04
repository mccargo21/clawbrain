#!/bin/bash
# ClawBrain Installation Script
# Automatically sets up hooks and dependencies for ClawdBot/OpenClaw

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK_NAME="clawbrain-startup"

echo "🧠 ClawBrain Installation Script"
echo "================================"

# Detect platform
detect_platform() {
    if [ -d "$HOME/.openclaw" ]; then
        echo "openclaw"
    elif [ -d "$HOME/.clawdbot" ]; then
        echo "clawdbot"
    elif [ -d "$HOME/clawd" ]; then
        # Check if it's clawdbot or openclaw style
        if [ -f "$HOME/.clawdbot/clawdbot.json" ]; then
            echo "clawdbot"
        else
            echo "clawdbot"  # Default to clawdbot for ~/clawd
        fi
    else
        echo "unknown"
    fi
}

PLATFORM=$(detect_platform)
echo "📍 Detected platform: $PLATFORM"

# Set paths based on platform
case "$PLATFORM" in
    openclaw)
        CONFIG_DIR="$HOME/.openclaw"
        HOOKS_DIR="$HOME/.openclaw/hooks"
        SERVICE_NAME="openclaw"
        ;;
    clawdbot)
        CONFIG_DIR="$HOME/.clawdbot"
        HOOKS_DIR="$HOME/.clawdbot/hooks"
        SERVICE_NAME="clawdbot"
        ;;
    *)
        echo "❌ Could not detect ClawdBot/OpenClaw installation"
        echo "   Please ensure you have ClawdBot or OpenClaw installed first."
        exit 1
        ;;
esac

echo "📁 Config directory: $CONFIG_DIR"
echo "📁 Hooks directory: $HOOKS_DIR"

# Create hooks directory if needed
mkdir -p "$HOOKS_DIR/$HOOK_NAME"

# Copy hook files
echo "📋 Installing hook: $HOOK_NAME"
cp "$SCRIPT_DIR/hooks/$HOOK_NAME/HOOK.md" "$HOOKS_DIR/$HOOK_NAME/"
cp "$SCRIPT_DIR/hooks/$HOOK_NAME/handler.js" "$HOOKS_DIR/$HOOK_NAME/"

echo "✅ Hook installed to $HOOKS_DIR/$HOOK_NAME"

# Check Python dependencies
echo ""
echo "🐍 Checking Python dependencies..."
if ! python3 -c "import psycopg2" 2>/dev/null; then
    echo "   ⚠️  psycopg2 not installed (PostgreSQL support disabled)"
    echo "   Run: pip3 install psycopg2-binary"
fi

if ! python3 -c "import redis" 2>/dev/null; then
    echo "   ⚠️  redis not installed (Redis caching disabled)"
    echo "   Run: pip3 install redis"
fi

if ! python3 -c "import cryptography" 2>/dev/null; then
    echo "   ⚠️  cryptography not installed (encryption disabled)"
    echo "   Run: pip3 install cryptography"
fi

# Check for sentence-transformers (optional)
if ! python3 -c "import sentence_transformers" 2>/dev/null; then
    echo "   ℹ️  sentence-transformers not installed (semantic search disabled)"
    echo "   Run: pip3 install sentence-transformers"
fi

# Environment variable setup
echo ""
echo "🔧 Environment Configuration"
echo "----------------------------"
echo "Add these to your systemd service config:"
echo ""
echo "  sudo mkdir -p /etc/systemd/system/${SERVICE_NAME}.service.d"
echo "  sudo tee /etc/systemd/system/${SERVICE_NAME}.service.d/brain.conf << EOF"
echo "[Service]"
echo "Environment=\"BRAIN_AGENT_ID=your-agent-name\""
echo "# Optional: PostgreSQL config"
echo "# Environment=\"BRAIN_POSTGRES_HOST=localhost\""
echo "# Environment=\"BRAIN_POSTGRES_PASSWORD=your-password\""
echo "# Optional: Redis config"
echo "# Environment=\"BRAIN_REDIS_HOST=localhost\""
echo "EOF"
echo ""
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl restart $SERVICE_NAME"

# Test the installation
echo ""
echo "🧪 Testing installation..."
if python3 "$SCRIPT_DIR/scripts/brain_bridge.py" <<< '{"command": "health_check", "args": {}}' 2>/dev/null | grep -q '"success": true'; then
    echo "✅ Brain bridge is working!"
else
    echo "⚠️  Brain bridge test failed - check Python dependencies"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Set BRAIN_AGENT_ID environment variable (see above)"
echo "   2. Restart the $SERVICE_NAME service"
echo "   3. Check logs: journalctl -u $SERVICE_NAME -f | grep brain"
