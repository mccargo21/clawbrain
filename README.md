# Brain v3 - AI Agent Memory System

Persistent, encrypted memory for AI agents.

## Install

```bash
pip install git+https://github.com/clawcolab/brain-v3.git
```

## Configure

Set environment variables:
```bash
export BRAIN_POSTGRES_HOST=your-host
export BRAIN_POSTGRES_USER=your-user
export BRAIN_POSTGRES_PASSWORD=your-password
export BRAIN_REDIS_HOST=your-redis-host
```

## Usage

```python
from brain import Brain

brain = Brain()
brain.remember(agent_id="jarvis", memory_type="fact", content="...")
memories = brain.recall(agent_id="jarvis", query="...")
```
