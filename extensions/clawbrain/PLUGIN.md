---
name: clawbrain
description: "Claw Brain Memory System - Personal AI memory with soul, bonding, and learning for OpenClaw"
homepage: https://github.com/clawcolab/clawbrain
metadata:
  {
    "openclaw":
      {
        "emoji": "🧠",
        "events": ["gateway:startup", "agent:bootstrap", "command:new"],
        "category": "memory",
        "requires": { "files": ["clawbrain.py", "scripts/brain_bridge.py"] },
        "provides": { "slot": "memory" },
        "install": [
          { "id": "bundled", "kind": "bundled", "label": "Bundled with ClawBrain" },
          { "id": "git", "kind": "git", "url": "https://github.com/clawcolab/clawbrain.git", "label": "Install from GitHub" }
        ],
      },
  }
---

# ClawBrain Plugin for OpenClaw 🧠

Personal AI Memory System with Soul, Bonding, and Learning capabilities.

## Features

- **Auto-Initialize**: Detects and connects to brain on gateway startup
- **Memory Tools**: `brain_recall` and `brain_remember` tools for agent use
- **Session Memory**: Automatically saves session context on `/new` command
- **Bootstrap Integration**: Injects memory context into agent bootstrap
- **Personality Prompt**: Generates personalized system prompts

## How It Works

1. **On Gateway Startup** (`gateway:startup`):
   - Detects existing brain instance (SQLite/PostgreSQL)
   - Loads and indexes recent memories
   - Prepares personality context

2. **On Agent Bootstrap** (`agent:bootstrap`):
   - Injects `MEMORY.md` with relevant context
   - Provides personalized system prompt

3. **On Session Reset** (`command:new`):
   - Saves current session to memory
   - Indexes conversation for future recall

## Tools Provided

### `brain_recall`

Search and retrieve memories from the brain.

```
brain_recall query="project deadlines" memory_type="task" limit=5
```

### `brain_remember`

Store new memories with optional tags.

```
brain_remember content="User prefers concise responses" type="preference" tags=["communication"]
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
          "recall_limit": 5
        }
      }
    },
    "slots": {
      "memory": "clawbrain"
    }
  }
}
```

### PostgreSQL + Redis (Production)

For production deployments with PostgreSQL and Redis:

```json
{
  "plugins": {
    "entries": {
      "clawbrain": {
        "enabled": true,
        "settings": {
          "storage_backend": "auto",
          "postgres_host": "localhost",
          "postgres_port": 5432,
          "postgres_db": "brain_db",
          "postgres_user": "brain_user",
          "postgres_password": "your_password",
          "redis_host": "localhost",
          "redis_port": 6379,
          "auto_recall": true,
          "recall_limit": 10
        }
      }
    },
    "slots": {
      "memory": "clawbrain"
    }
  }
}
```

Or use environment variables:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=brain_db
export POSTGRES_USER=brain_user
export POSTGRES_PASSWORD=your_password
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### Storage Behavior

| Backend | Primary Storage | Caching | Best For |
|---------|-----------------|---------|----------|
| SQLite (default) | Local file | None | Development, single-user |
| PostgreSQL | Network DB | None | Multi-user, no caching |
| PostgreSQL + Redis | Network DB | Redis | Production, high performance |

**Auto-detection priority:**
1. If PostgreSQL is reachable → use PostgreSQL
2. Otherwise → fallback to SQLite
3. If Redis is available → enable caching layer

**Redis caches:**
- Conversation history (fast retrieval)
- Session data with TTL
- Working memory

**Graceful degradation:** If PostgreSQL fails, falls back to SQLite. If Redis is unavailable, caching is skipped (no errors).

## Events

| Event | Action |
|-------|--------|
| `gateway:startup` | Initialize brain, load memories |
| `agent:bootstrap` | Inject memory context |
| `command:new` | Save session to memory |

## Integration with BOOT.md

Create `~/.openclaw/workspace/BOOT.md`:

```markdown
# BOOT.md - Brain Initialization

On startup:
1. Check brain health with brain_recall
2. Load recent context
3. Greet user if configured
```
