# Brain v3 - Personal AI Memory System

**A sophisticated AI agent memory system with SQLite fallback.**

## Features

- 🎭 Soul/Personality - 6 evolving traits (humor, empathy, curiosity...)
- 👤 User Profile - Learns user preferences, interests, communication style
- 💭 Conversation State - Real-time mood/intent detection
- 📚 Learning Insights - Continuous improvement from corrections
- 🧠 get_full_context() - Everything for personalized responses

## Quick Start

```bash
pip install git+https://github.com/clawcolab/brain-v3.git
```

```python
from brain import Brain

# Uses SQLite by default (no PostgreSQL/Redis needed!)
brain = Brain()

# Or explicitly:
brain = Brain({"storage_backend": "sqlite"})

# Get full context for personalized response
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
| SQLite | Default - works out of the box! |
| PostgreSQL | For production: `{"storage_backend": "postgresql"}` |
| Redis | Optional caching: `{"use_redis": true}` |

## Auto-Detection

Brain v3 auto-detects available storage:
1. PostgreSQL (if available and configured)
2. SQLite (fallback - always works!)

## API

- `get_full_context()` - Main context retrieval
- `process_message()` - Message handling entry point
- `detect_user_mood()` - Emotional analysis
- `detect_user_intent()` - Message classification
- `learn_user_preference()` - Auto-learn from interactions
- `generate_personality_prompt()` - Dynamic LLM prompts

## Installation

```bash
pip install git+https://github.com/clawcolab/brain-v3.git
```

## Requirements

- Python 3.10+
- Optional: PostgreSQL, Redis (auto-detected, not required)
- Optional: sentence-transformers (for embeddings)

## License

MIT
