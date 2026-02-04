/**
 * ClawBrain Plugin Handler for OpenClaw
 * 
 * Provides memory integration with hooks for:
 * - gateway:startup - Initialize brain and load context
 * - agent:bootstrap - Inject memory into bootstrap files
 * - command:new - Save session to memory
 * 
 * Tools provided:
 * - brain_recall - Search memories
 * - brain_remember - Store memories
 * - brain_context - Get full context for personalization
 * 
 * Note: This plugin is designed for the OpenClaw ecosystem.
 * When installing, ensure @types/node is available.
 */

/// <reference types="node" />

import { spawn } from "child_process";
import * as path from "path";
import * as os from "os";

// Type definitions (OpenClaw provides these when installed as plugin)
type HookHandler = (event: { type: string; action: string; context?: unknown }) => Promise<void>;

interface OpenClawConfig {
  workspace?: { dir?: string };
  plugins?: {
    entries?: Record<string, unknown>;
  };
}

interface AgentBootstrapHookContext {
  workspaceDir: string;
  bootstrapFiles: Array<{
    name: string;
    path: string;
    content: string;
    missing: boolean;
  }>;
}

// Plugin state
interface BrainState {
  initialized: boolean;
  healthy: boolean;
  storageBackend: string;
  lastSync: string | null;
  memoriesCount: number;
}

let brainState: BrainState = {
  initialized: false,
  healthy: false,
  storageBackend: "unknown",
  lastSync: null,
  memoriesCount: 0,
};

// Configuration interface
interface BrainPluginConfig {
  enabled?: boolean;
  settings?: {
    storage_backend?: "auto" | "sqlite" | "postgresql";
    sqlite_path?: string;
    postgres_host?: string;
    postgres_port?: number;
    postgres_db?: string;
    postgres_user?: string;
    postgres_password?: string;
    auto_recall?: boolean;
    recall_limit?: number;
    agent_id?: string;
    user_id?: string;
  };
}

// Resolve plugin directory
function getPluginDir(): string {
  return path.dirname(new URL(import.meta.url).pathname);
}

// Get workspace directory
function getWorkspaceDir(cfg?: OpenClawConfig): string {
  return cfg?.workspace?.dir || path.join(os.homedir(), ".openclaw", "workspace");
}

// Python bridge - runs clawbrain commands
async function runBrainCommand(
  command: string,
  args: Record<string, unknown> = {},
  cfg?: OpenClawConfig
): Promise<{ success: boolean; data?: unknown; error?: string }> {
  const pluginDir = getPluginDir();
  const bridgeScript = path.join(pluginDir, "scripts", "brain_bridge.py");
  const workspaceDir = getWorkspaceDir(cfg);
  
  // Get plugin settings
  const pluginConfig = cfg?.plugins?.entries?.clawbrain as BrainPluginConfig | undefined;
  const settings = pluginConfig?.settings || {};
  
  // Build config for Python bridge
  const brainConfig = {
    storage_backend: settings.storage_backend || "auto",
    sqlite_path: settings.sqlite_path?.replace("~", os.homedir()) || 
                 path.join(workspaceDir, "brain.db"),
    postgres_host: settings.postgres_host || process.env.POSTGRES_HOST,
    postgres_port: settings.postgres_port || process.env.POSTGRES_PORT,
    postgres_db: settings.postgres_db || process.env.POSTGRES_DB,
    postgres_user: settings.postgres_user || process.env.POSTGRES_USER,
    postgres_password: settings.postgres_password || process.env.POSTGRES_PASSWORD,
    agent_id: settings.agent_id || "openclaw",
    user_id: settings.user_id || "user",
  };
  
  const input = JSON.stringify({
    command,
    args,
    config: brainConfig,
  });
  
  return new Promise((resolve) => {
    const proc = spawn("python3", [bridgeScript], {
      cwd: pluginDir,
      env: { ...process.env },
    });
    
    let stdout = "";
    let stderr = "";
    
    proc.stdout.on("data", (data: Buffer) => {
      stdout += data.toString();
    });
    
    proc.stderr.on("data", (data: Buffer) => {
      stderr += data.toString();
    });
    
    proc.stdin.write(input);
    proc.stdin.end();
    
    proc.on("close", (code) => {
      if (code !== 0) {
        resolve({ success: false, error: stderr || `Process exited with code ${code}` });
        return;
      }
      
      try {
        const result = JSON.parse(stdout);
        resolve({ success: true, data: result });
      } catch {
        resolve({ success: false, error: `Invalid JSON response: ${stdout}` });
      }
    });
    
    proc.on("error", (err) => {
      resolve({ success: false, error: `Failed to run bridge: ${err.message}` });
    });
  });
}

// ========== HOOKS ==========

/**
 * Gateway Startup Hook
 * Initializes brain connection and loads context
 */
const onGatewayStartup: HookHandler = async (event) => {
  if (event.type !== "gateway" || event.action !== "startup") {
    return;
  }
  
  const context = (event.context ?? {}) as { cfg?: OpenClawConfig; workspaceDir?: string };
  const cfg = context.cfg;
  
  console.log("[clawbrain] Gateway startup - initializing brain...");
  
  // Use the new refresh_on_startup method which does everything
  const refreshResult = await runBrainCommand("refresh_on_startup", {}, cfg);
  
  if (refreshResult.success && refreshResult.data) {
    const data = refreshResult.data as {
      success: boolean;
      sync?: { memories_count: number; storage_backend: string };
      health?: { storage: boolean };
      refreshed_at?: string;
    };
    
    if (data.success) {
      brainState = {
        initialized: true,
        healthy: data.health?.storage || false,
        storageBackend: data.sync?.storage_backend || "unknown",
        lastSync: data.refreshed_at || new Date().toISOString(),
        memoriesCount: data.sync?.memories_count || 0,
      };
      console.log(`[clawbrain] Brain initialized and refreshed (${brainState.storageBackend}, ${brainState.memoriesCount} memories)`);
    } else {
      console.warn("[clawbrain] Brain refresh failed:", data);
      brainState.initialized = false;
      brainState.healthy = false;
    }
  } else {
    // Fallback to simple health check
    const health = await runBrainCommand("health_check", {}, cfg);
    
    if (health.success && health.data) {
      const healthData = health.data as { storage: string; healthy: boolean };
      brainState = {
        initialized: true,
        healthy: healthData.healthy,
        storageBackend: healthData.storage || "unknown",
        lastSync: new Date().toISOString(),
        memoriesCount: 0,
      };
      console.log(`[clawbrain] Brain initialized (${brainState.storageBackend})`);
      
      // Sync recent memories
      const syncResult = await runBrainCommand("sync", {}, cfg);
      if (syncResult.success && syncResult.data) {
        const syncData = syncResult.data as { memories_count?: number };
        brainState.memoriesCount = syncData.memories_count || 0;
        console.log(`[clawbrain] Synced ${brainState.memoriesCount} memories`);
      }
    } else {
      console.warn("[clawbrain] Brain health check failed:", refreshResult.error || health.error);
      brainState.initialized = false;
      brainState.healthy = false;
    }
  }
};

/**
 * Agent Bootstrap Hook
 * Injects memory context into bootstrap files
 */
const onAgentBootstrap: HookHandler = async (event: { type: string; action: string; context?: unknown }) => {
  if (event.type !== "agent" || event.action !== "bootstrap") {
    return;
  }
  
  if (!brainState.initialized) {
    return;
  }
  
  const context = event.context as AgentBootstrapHookContext | undefined;
  if (!context?.workspaceDir) {
    return;
  }
  
  console.log("[clawbrain] Agent bootstrap - injecting memory context...");
  
  // Use the new get_startup_context for formatted MEMORY.md content
  const ctxResult = await runBrainCommand("get_startup_context", {});
  
  if (!ctxResult.success || !ctxResult.data) {
    return;
  }
  
  const startupContext = ctxResult.data as { content: string };
  
  // Inject as MEMORY.md bootstrap file
  if (context.bootstrapFiles && startupContext.content) {
    context.bootstrapFiles.push({
      name: "MEMORY.md",
      path: path.join(context.workspaceDir, "MEMORY.md"),
      content: startupContext.content,
      missing: false,
    });
    console.log("[clawbrain] Injected MEMORY.md with brain context");
  }
};

/**
 * Command New Hook
 * Saves session to memory when /new is issued
 */
const onCommandNew: HookHandler = async (event: { type: string; action: string; context?: unknown }) => {
  if (event.type !== "command" || event.action !== "new") {
    return;
  }
  
  if (!brainState.initialized) {
    return;
  }
  
  console.log("[clawbrain] Session reset - saving to memory...");
  
  const context = event.context as { sessionKey?: string; messages?: Array<{ role: string; content: string }> } | undefined;
  
  if (context?.messages?.length) {
    // Use the new save_session command
    const saveResult = await runBrainCommand("save_session", {
      session_key: context.sessionKey || `session_${Date.now()}`,
      messages: context.messages,
      tags: ["session", "conversation", "auto-saved"],
    });
    
    if (saveResult.success) {
      brainState.lastSync = new Date().toISOString();
      console.log("[clawbrain] Session saved to memory");
    } else {
      console.warn("[clawbrain] Failed to save session:", saveResult.error);
    }
  }
};

// ========== TOOLS ==========

/**
 * Tool definitions for OpenClaw agent
 */
export const tools = [
  {
    name: "brain_recall",
    description: "Search and retrieve memories from the brain. Use this to find relevant context, past conversations, preferences, or learned information.",
    parameters: {
      type: "object" as const,
      properties: {
        query: {
          type: "string",
          description: "Search query to find relevant memories",
        },
        memory_type: {
          type: "string",
          description: "Type of memory to search (e.g., 'conversation', 'preference', 'knowledge', 'task')",
        },
        limit: {
          type: "number",
          description: "Maximum number of memories to return (default: 5)",
        },
        tags: {
          type: "array",
          items: { type: "string" },
          description: "Filter by tags",
        },
      },
      required: ["query"],
    },
    handler: async (params: { query: string; memory_type?: string; limit?: number; tags?: string[] }) => {
      const result = await runBrainCommand("recall", {
        query: params.query,
        memory_type: params.memory_type,
        limit: params.limit || 5,
        tags: params.tags,
      });
      
      if (!result.success) {
        return { error: result.error };
      }
      
      return result.data;
    },
  },
  {
    name: "brain_remember",
    description: "Store a new memory in the brain. Use this to save important information, preferences, facts, or context for future recall.",
    parameters: {
      type: "object" as const,
      properties: {
        content: {
          type: "string",
          description: "The content to remember",
        },
        memory_type: {
          type: "string",
          description: "Type of memory (e.g., 'preference', 'knowledge', 'task', 'conversation')",
        },
        key: {
          type: "string",
          description: "Optional unique key for this memory",
        },
        tags: {
          type: "array",
          items: { type: "string" },
          description: "Tags for categorization",
        },
        importance: {
          type: "number",
          description: "Importance level 1-10 (default: 5)",
        },
      },
      required: ["content", "memory_type"],
    },
    handler: async (params: { content: string; memory_type: string; key?: string; tags?: string[]; importance?: number }) => {
      const result = await runBrainCommand("remember", {
        content: params.content,
        memory_type: params.memory_type,
        key: params.key,
        tags: params.tags || [],
        importance: params.importance || 5,
      });
      
      if (!result.success) {
        return { error: result.error };
      }
      
      return { success: true, message: "Memory stored successfully" };
    },
  },
  {
    name: "brain_context",
    description: "Get full context for personalized responses. Returns user profile, recent memories, mood detection, and response guidance.",
    parameters: {
      type: "object" as const,
      properties: {
        message: {
          type: "string",
          description: "Current user message for mood/intent analysis",
        },
        session_key: {
          type: "string",
          description: "Session identifier",
        },
      },
      required: [],
    },
    handler: async (params: { message?: string; session_key?: string }) => {
      const result = await runBrainCommand("get_full_context", {
        message: params.message || "",
        session_key: params.session_key || "default",
      });
      
      if (!result.success) {
        return { error: result.error };
      }
      
      return result.data;
    },
  },
  {
    name: "brain_learn",
    description: "Learn a user preference or insight. Use this to record user preferences, communication style, interests, etc.",
    parameters: {
      type: "object" as const,
      properties: {
        preference_type: {
          type: "string",
          enum: ["interest", "expertise", "learning", "communication"],
          description: "Type of preference to learn",
        },
        value: {
          type: "string",
          description: "The preference value to learn",
        },
        user_id: {
          type: "string",
          description: "User identifier (optional)",
        },
      },
      required: ["preference_type", "value"],
    },
    handler: async (params: { preference_type: string; value: string; user_id?: string }) => {
      const result = await runBrainCommand("learn_user_preference", {
        pref_type: params.preference_type,
        value: params.value,
        user_id: params.user_id || "default",
      });
      
      if (!result.success) {
        return { error: result.error };
      }
      
      return { success: true, message: `Learned ${params.preference_type}: ${params.value}` };
    },
  },
  {
    name: "brain_health",
    description: "Check brain health and status. Returns storage backend, connection status, and memory count.",
    parameters: {
      type: "object" as const,
      properties: {},
      required: [],
    },
    handler: async () => {
      const result = await runBrainCommand("health_check", {});
      
      const healthData = result.success && typeof result.data === "object" && result.data !== null
        ? result.data as Record<string, unknown>
        : { error: result.error };
      
      return {
        ...brainState,
        ...healthData,
      };
    },
  },
];

// ========== PLUGIN EXPORT ==========

export const hooks = [
  { event: "gateway:startup", handler: onGatewayStartup },
  { event: "agent:bootstrap", handler: onAgentBootstrap },
  { event: "command:new", handler: onCommandNew },
];

export const plugin = {
  id: "clawbrain",
  name: "ClawBrain Memory System",
  description: "Personal AI memory with soul, bonding, and learning",
  version: "3.0.0",
  hooks,
  tools,
};

export default plugin;
