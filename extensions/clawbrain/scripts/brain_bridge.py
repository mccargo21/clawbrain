#!/usr/bin/env python3
"""
ClawBrain Bridge for OpenClaw Plugin

This script acts as a bridge between the TypeScript OpenClaw plugin and the
Python ClawBrain module. It receives JSON commands via stdin and outputs
JSON results via stdout.

Usage (from TypeScript):
    echo '{"command": "health_check", "args": {}, "config": {...}}' | python3 brain_bridge.py

Commands:
    - health_check: Check brain connection status
    - sync: Sync and index recent memories  
    - recall: Search memories
    - remember: Store a memory
    - get_full_context: Get full context for personalization
    - learn_user_preference: Learn a user preference
    - generate_personality_prompt: Generate personality system prompt
"""

import sys
import json
import os
from pathlib import Path

# Add parent directory to path to import clawbrain
plugin_dir = Path(__file__).parent.parent
repo_dir = plugin_dir.parent
sys.path.insert(0, str(repo_dir))

try:
    from clawbrain import Brain
    BRAIN_AVAILABLE = True
except ImportError as e:
    BRAIN_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Global brain instance (singleton per process)
_brain_instance = None


def get_brain(config: dict) -> "Brain":
    """Get or create brain instance with config."""
    global _brain_instance
    
    if _brain_instance is None:
        _brain_instance = Brain(config)
    
    return _brain_instance


def cmd_health_check(brain: Brain, args: dict) -> dict:
    """Check brain health and connection status."""
    health = brain.health_check()
    return {
        "healthy": all(health.values()),
        "storage": brain.storage_backend,
        "details": health,
    }


def cmd_sync(brain: Brain, args: dict) -> dict:
    """Sync and count memories."""
    agent_id = args.get("agent_id", "openclaw")
    memories = brain.recall(agent_id=agent_id, limit=100)
    
    return {
        "memories_count": len(memories),
        "last_sync": memories[0].created_at if memories else None,
        "storage_backend": brain.storage_backend,
    }


def cmd_recall(brain: Brain, args: dict) -> dict:
    """Search memories."""
    agent_id = args.get("agent_id", "openclaw")
    query = args.get("query", "")
    memory_type = args.get("memory_type")
    limit = args.get("limit", 10)
    tags = args.get("tags", [])
    
    # If tags provided, search by tags
    if tags:
        memories = brain.search_by_tags(
            tags=tags,
            agent_id=agent_id,
            memory_type=memory_type,
            limit=limit
        )
    else:
        memories = brain.recall(
            agent_id=agent_id,
            query=query,
            memory_type=memory_type,
            limit=limit
        )
    
    return {
        "memories": [
            {
                "id": m.id,
                "type": m.memory_type,
                "key": m.key,
                "content": m.content,
                "summary": m.summary,
                "tags": m.tags,
                "importance": m.importance,
                "created_at": m.created_at,
            }
            for m in memories
        ],
        "count": len(memories),
    }


def cmd_remember(brain: Brain, args: dict) -> dict:
    """Store a memory."""
    agent_id = args.get("agent_id", "openclaw")
    memory_type = args.get("memory_type", "knowledge")
    content = args.get("content", "")
    key = args.get("key")
    tags = args.get("tags", [])
    importance = args.get("importance", 5)
    auto_tag = args.get("auto_tag", False)
    
    if not content:
        return {"error": "Content is required"}
    
    memory = brain.remember(
        agent_id=agent_id,
        memory_type=memory_type,
        content=content,
        key=key,
        tags=tags,
        auto_tag=auto_tag,
        importance=importance,
    )
    
    return {
        "success": True,
        "memory_id": memory.id,
        "type": memory.memory_type,
        "tags": memory.tags,
    }


def cmd_get_full_context(brain: Brain, args: dict) -> dict:
    """Get full context for personalization."""
    session_key = args.get("session_key", "openclaw_main")
    user_id = args.get("user_id", "default")
    agent_id = args.get("agent_id", "openclaw")
    message = args.get("message", "")
    
    context = brain.get_full_context(
        session_key=session_key,
        user_id=user_id,
        agent_id=agent_id,
        message=message
    )
    
    return context


def cmd_learn_user_preference(brain: Brain, args: dict) -> dict:
    """Learn a user preference."""
    user_id = args.get("user_id", "default")
    pref_type = args.get("pref_type", "interest")
    value = args.get("value", "")
    
    if not value:
        return {"error": "Value is required"}
    
    brain.learn_user_preference(
        user_id=user_id,
        preference_type=pref_type,
        value=value
    )
    
    return {
        "success": True,
        "learned": f"{pref_type}: {value}",
    }


def cmd_generate_personality_prompt(brain: Brain, args: dict) -> dict:
    """Generate personality system prompt."""
    agent_id = args.get("agent_id", "openclaw")
    user_id = args.get("user_id", "default")
    
    prompt = brain.generate_personality_prompt(
        agent_id=agent_id,
        user_id=user_id
    )
    
    return {
        "prompt": prompt,
    }


def cmd_get_user_profile(brain: Brain, args: dict) -> dict:
    """Get user profile."""
    user_id = args.get("user_id", "default")
    
    profile = brain.get_user_profile(user_id)
    
    return {
        "user_id": profile.user_id,
        "name": profile.name,
        "nickname": profile.nickname,
        "preferred_name": profile.preferred_name,
        "interests": profile.interests,
        "expertise_areas": profile.expertise_areas,
        "learning_topics": profile.learning_topics,
        "communication_preferences": profile.communication_preferences,
        "total_interactions": profile.total_interactions,
        "first_interaction": profile.first_interaction,
        "last_interaction": profile.last_interaction,
    }


def cmd_detect_mood(brain: Brain, args: dict) -> dict:
    """Detect user mood from message."""
    message = args.get("message", "")
    
    if not message:
        return {"mood": "neutral", "confidence": 0.5, "all_moods": {}}
    
    return brain.detect_user_mood(message)


def cmd_detect_intent(brain: Brain, args: dict) -> dict:
    """Detect user intent from message."""
    message = args.get("message", "")
    
    if not message:
        return {"intent": "statement"}
    
    return {"intent": brain.detect_user_intent(message)}


def cmd_refresh_on_startup(brain: Brain, args: dict) -> dict:
    """Refresh brain state on OpenClaw service startup."""
    agent_id = args.get("agent_id", "openclaw")
    user_id = args.get("user_id", "default")
    
    return brain.refresh_on_startup(agent_id=agent_id, user_id=user_id)


def cmd_save_session(brain: Brain, args: dict) -> dict:
    """Save session messages to memory."""
    session_key = args.get("session_key", "")
    messages = args.get("messages", [])
    agent_id = args.get("agent_id", "openclaw")
    tags = args.get("tags", [])
    
    if not session_key:
        return {"error": "session_key is required"}
    
    return brain.save_session_to_memory(
        session_key=session_key,
        messages=messages,
        agent_id=agent_id,
        tags=tags
    )


def cmd_get_startup_context(brain: Brain, args: dict) -> dict:
    """Get formatted context for MEMORY.md injection."""
    agent_id = args.get("agent_id", "openclaw")
    user_id = args.get("user_id", "default")
    
    content = brain.get_startup_context(agent_id=agent_id, user_id=user_id)
    
    return {
        "content": content,
        "format": "markdown",
    }


# Command dispatcher
COMMANDS = {
    "health_check": cmd_health_check,
    "sync": cmd_sync,
    "recall": cmd_recall,
    "remember": cmd_remember,
    "get_full_context": cmd_get_full_context,
    "learn_user_preference": cmd_learn_user_preference,
    "generate_personality_prompt": cmd_generate_personality_prompt,
    "get_user_profile": cmd_get_user_profile,
    "detect_mood": cmd_detect_mood,
    "detect_intent": cmd_detect_intent,
    "refresh_on_startup": cmd_refresh_on_startup,
    "save_session": cmd_save_session,
    "get_startup_context": cmd_get_startup_context,
}


def main():
    """Main entry point - reads JSON from stdin, outputs JSON to stdout."""
    
    # Check if brain module is available
    if not BRAIN_AVAILABLE:
        result = {
            "error": f"ClawBrain module not available: {IMPORT_ERROR}",
            "success": False,
        }
        print(json.dumps(result))
        return
    
    # Read input from stdin
    try:
        input_data = sys.stdin.read()
        request = json.loads(input_data)
    except json.JSONDecodeError as e:
        result = {"error": f"Invalid JSON input: {e}", "success": False}
        print(json.dumps(result))
        return
    
    # Extract command, args, and config
    command = request.get("command", "")
    args = request.get("args", {})
    config = request.get("config", {})
    
    # Merge environment variables into config
    if not config.get("postgres_host"):
        config["postgres_host"] = os.environ.get("POSTGRES_HOST", "localhost")
    if not config.get("postgres_port"):
        config["postgres_port"] = int(os.environ.get("POSTGRES_PORT", "5432"))
    if not config.get("postgres_db"):
        config["postgres_db"] = os.environ.get("POSTGRES_DB", "brain_db")
    if not config.get("postgres_user"):
        config["postgres_user"] = os.environ.get("POSTGRES_USER", "brain_user")
    if not config.get("postgres_password"):
        config["postgres_password"] = os.environ.get("POSTGRES_PASSWORD", "")
    
    # Pass agent_id from config to args if not set
    if "agent_id" not in args and "agent_id" in config:
        args["agent_id"] = config["agent_id"]
    if "user_id" not in args and "user_id" in config:
        args["user_id"] = config["user_id"]
    
    # Validate command
    if command not in COMMANDS:
        result = {
            "error": f"Unknown command: {command}. Available: {list(COMMANDS.keys())}",
            "success": False,
        }
        print(json.dumps(result))
        return
    
    # Get or create brain instance
    try:
        brain = get_brain(config)
    except Exception as e:
        result = {"error": f"Failed to initialize brain: {e}", "success": False}
        print(json.dumps(result))
        return
    
    # Execute command
    try:
        handler = COMMANDS[command]
        result = handler(brain, args)
        result["success"] = result.get("success", True)
    except Exception as e:
        result = {"error": f"Command failed: {e}", "success": False}
    
    # Output result
    print(json.dumps(result))


if __name__ == "__main__":
    main()
