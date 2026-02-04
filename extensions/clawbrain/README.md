# ClawBrain Plugin for OpenClaw 🧠

Personal AI Memory System with automatic refresh on service restart.

## Features

- **Auto-Initialize**: Detects and connects to brain on gateway startup
- **Memory Tools**: `brain_recall`, `brain_remember`, `brain_context`, `brain_learn`
- **Session Memory**: Automatically saves session context on `/new` command
- **Bootstrap Integration**: Injects `MEMORY.md` with context on agent bootstrap
- **Personality Prompt**: Generates personalized system prompts

## Installation

### Method 1: Clone to OpenClaw Plugins

```bash
cd ~/.openclaw/plugins
git clone https://github.com/clawcolab/clawbrain.git
```

### Method 2: Manual Installation

```bash
# Clone anywhere
git clone https://github.com/clawcolab/clawbrain.git

# Install dependencies
cd clawbrain/extensions/clawbrain
npm install

# Link to OpenClaw plugins
ln -s $(pwd) ~/.openclaw/plugins/clawbrain
```

## Configuration

Add to `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "clawbrain": {
        "enabled": true,
        "settings": {
          "storage_backend": "auto",
          "sqlite_path": "~/.openclaw/workspace/brain.db",
          "auto_recall": true,
          "recall_limit": 5,
          "agent_id": "assistant",
          "user_id": "user"
        }
      }
    },
    "slots": {
      "memory": "clawbrain"
    }
  },
  "hooks": {
    "internal": {
      "enabled": true
    }
  }
}
```

## Enable Boot Hook

```bash
openclaw hooks enable boot-md
```

## Restart Gateway

```bash
openclaw gateway restart
```

## How It Works

### On Gateway Startup (`gateway:startup`)

1. Detects existing brain instance (SQLite or PostgreSQL)
2. Runs health check and syncs recent memories
3. Reports memory count to console

### On Agent Bootstrap (`agent:bootstrap`)

1. Generates formatted `MEMORY.md` content
2. Includes user profile and recent memories
3. Injects into agent's bootstrap files

### On Session Reset (`command:new`)

1. Captures current session messages
2. Saves to brain as conversation memory
3. Auto-tags for future recall

## Available Tools

| Tool | Description |
|------|-------------|
| `brain_recall` | Search memories by query, type, or tags |
| `brain_remember` | Store new memories with tags |
| `brain_context` | Get full context for personalization |
| `brain_learn` | Learn user preferences |
| `brain_health` | Check brain status |

## Python Requirements

The plugin requires Python 3.8+ with:

```bash
pip install sentence-transformers psycopg2-binary redis
```

## Storage Backends

- **SQLite** (default): Zero setup, local file database
- **PostgreSQL**: Production-ready, configure with environment variables:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=brain_db
export POSTGRES_USER=brain_user
export POSTGRES_PASSWORD=your_password
```

## Files

- `handler.ts` - OpenClaw plugin handler with hooks and tools
- `PLUGIN.md` - Plugin manifest and documentation
- `package.json` - NPM package config
- `scripts/brain_bridge.py` - Python bridge for ClawBrain commands
- `templates/BOOT.md` - Boot checklist template
- `templates/MEMORY.md` - Memory context template

## License

MIT
