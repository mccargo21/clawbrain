# Claw Brain 🧠

**Personal AI Memory System for AI Agents**

A sophisticated memory and learning system that enables truly personalized AI-human communication.

## Features

- 🎭 **Soul/Personality** - 6 evolving traits (humor, empathy, curiosity...)
- 👤 **User Profile** - Learns your preferences, interests, communication style
- 💭 **Conversation State** - Real-time mood detection and context tracking
- 📚 **Learning Insights** - Continuously learns and improves from interactions
- 🧠 **get_full_context()** - Everything for personalized responses

## Quick Start

```bash
pip install git+https://github.com/clawcolab/clawbrain.git
```

```python
from brain import Brain

# Uses PostgreSQL + Redis (auto-detected), falls back to SQLite
brain = Brain()

# Get full context for personalized responses
context = brain.get_full_context(
    session_key="chat_123",
    user_id="pranab",
    agent_id="moltbot",
    message="Hey, how's it going?"
)

# Returns: user profile, mood, intent, memories, response guidance
```

## Storage Backends

| Backend | Usage |
|---------|-------|
| **PostgreSQL** | Production - auto-detected |
| **Redis** | Caching - auto-detected |
| **SQLite** | Fallback - works out of the box |

## For ClawDBot

Install as a skill:
```bash
pip install git+https://github.com/clawcolab/clawbrain-skill.git
```

## Repository Structure

- **clawbrain** - Core memory library (this repo)
- **clawbrain-skill** - ClawDBot integration skill

## License

MIT
