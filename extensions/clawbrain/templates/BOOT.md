# BOOT.md - Brain Initialization

On gateway startup, initialize the brain and load context:

1. Check brain health: Run `brain_health` to verify connection
2. If healthy, retrieve recent context using `brain_context`
3. If user context exists, use it for personalized responses

## Startup Checklist

- [ ] Brain connection verified
- [ ] User profile loaded
- [ ] Recent memories indexed

## On Session Start

When starting a new session:
1. Load relevant memories for the current context
2. Apply user preferences for response style

## Memory Refresh

After each conversation:
1. Save important information with `brain_remember`
2. Learn user preferences with `brain_learn`

---

_This file runs automatically on OpenClaw gateway startup when the boot-md hook is enabled._
