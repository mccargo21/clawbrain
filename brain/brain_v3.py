#!/usr/bin/env python3
"""
Brain v3 - Personal AI Memory System for Moltbot

A sophisticated memory and learning system that enables truly personalized 
AI-human communication. The brain learns about you over time, remembers 
context, tracks emotional states, and continuously improves its understanding.

Features:
- 🎭 Soul/Personality - Evolving personality traits that adapt over time
- 💝 Bonding - Deep relationship tracking that grows with each interaction
- 👤 User Profile - Learns your preferences, interests, and communication style
- 🧠 Semantic Search - Meaning-based memory retrieval using embeddings
- 💭 Conversation State - Real-time mood detection and context tracking
- 📚 Learning Insights - Continuously learns and improves from interactions
- 📝 Todos - Task management with memory linking
- 🎯 Goals - Long-term objective tracking
- 📊 Stats - Interaction statistics and patterns

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                      MOLTBOT BRAIN v3                           │
├─────────────────────────────────────────────────────────────────┤
│  👤 User Profile      │  🗣️ Conversations    │  💾 Memories    │
│  ├─ Preferences       │  ├─ History          │  ├─ Facts       │
│  ├─ Interests         │  ├─ Context Stack    │  ├─ Secrets     │
│  └─ Patterns          │  └─ Topic Tracking   │  └─ Knowledge   │
├─────────────────────────────────────────────────────────────────┤
│  💭 Conversation State│  📚 Learning         │  🎭 Soul        │
│  ├─ Mood Detection    │  ├─ Insights         │  ├─ Traits      │
│  ├─ Intent Analysis   │  ├─ Corrections      │  └─ Style       │
│  └─ Engagement Level  │  └─ Preferences      │                 │
├─────────────────────────────────────────────────────────────────┤
│  💝 Bonding           │  🎯 Goals            │  📝 Todos       │
│  ├─ Relationship Level│  ├─ Progress         │  ├─ Priority    │
│  ├─ Milestones        │  └─ Milestones       │  └─ Status      │
│  └─ Shared Moments    │                      │                 │
└─────────────────────────────────────────────────────────────────┘

Usage:
    brain = Brain(config)
    
    # Get full context before responding (the main method)
    context = brain.get_full_context(
        session_key="chat_123",
        user_id="pranab", 
        agent_id="moltbot",
        message="Hey, how's it going?"
    )
    
    # The context includes everything needed for personalized response:
    # - User's preferences, interests, communication style
    # - Current mood and intent detection
    # - Relevant memories and learnings
    # - Response guidance (tone, formality, etc.)
"""

import os
import json
import hashlib
import base64
import secrets
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from contextlib import contextmanager
from threading import Lock

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)



try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

import psycopg2
import psycopg2.extras
import redis

DEFAULT_CONFIG = {
    "postgres_host": "192.168.4.176",
    "postgres_port": 5432,
    "postgres_db": "brain_db",
    "postgres_user": "brain_user",
    "postgres_password": "brain_secure_password_2024_rotated",
    "redis_host": "192.168.4.175",
    "redis_port": 6379,
    "redis_db": 0,
    "redis_prefix": "brain:",
    "embedding_model": "all-MiniLM-L6-v2",
    "working_memory_size": 20,
    "working_memory_ttl": 86400,
    "backup_dir": "/root/.brain/backup",
    "encryption_key": None,
    "soul_update_threshold": 10,
}


@dataclass
class Memory:
    id: Optional[str] = None
    agent_id: str = ""
    memory_type: str = ""
    key: Optional[str] = None
    content: str = ""
    content_encrypted: bool = False
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    category: str = ""
    importance: int = 5
    access_count: int = 0
    last_accessed: Optional[str] = None
    linked_to: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    expires_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    consolidated: bool = False
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "memory_type": self.memory_type,
            "key": self.key,
            "summary": self.summary,
            "keywords": self.keywords,
            "importance": self.importance,
            "access_count": self.access_count,
            "created_at": self.created_at,
        }


@dataclass
class Todo:
    id: Optional[str] = None
    agent_id: str = ""
    title: str = ""
    description: str = ""
    status: str = "pending"
    priority: int = 5
    due_at: Optional[str] = None
    completed_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    linked_memory_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "tags": self.tags,
            "created_at": self.created_at,
        }


@dataclass
class Soul:
    agent_id: str = ""
    humor: float = 5.0
    formality: float = 5.0
    empathy: float = 5.0
    verbosity: float = 5.0
    creativity: float = 5.0
    curiosity: float = 5.0
    preferred_topics: List[str] = field(default_factory=list)
    avoided_topics: List[str] = field(default_factory=list)
    catchphrases: List[str] = field(default_factory=list)
    communication_style: str = "balanced"
    interaction_count: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    last_interaction: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "traits": {
                "humor": self.humor,
                "formality": self.formality,
                "empathy": self.empathy,
                "verbosity": self.verbosity,
                "creativity": self.creativity,
                "curiosity": self.curiosity,
            },
            "preferred_topics": self.preferred_topics,
            "interaction_count": self.interaction_count,
        }


@dataclass
class Bond:
    human_id: str = ""
    agent_id: str = ""
    score: float = 50.0
    level: str = "stranger"
    shared_conversations: int = 0
    shared_projects: int = 0
    shared_secrets: int = 0
    total_interactions: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    avg_response_time: float = 0.0
    happy_moments: List[str] = field(default_factory=list)
    challenging_moments: List[str] = field(default_factory=list)
    milestones: List[str] = field(default_factory=list)
    first_interaction: Optional[str] = None
    last_interaction: Optional[str] = None
    last_positive: Optional[str] = None
    last_negative: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "human_id": self.human_id,
            "agent_id": self.agent_id,
            "score": self.score,
            "level": self.level,
            "total_interactions": self.total_interactions,
            "positive_interactions": self.positive_interactions,
        }


@dataclass
class Goal:
    id: Optional[str] = None
    agent_id: str = ""
    human_id: str = ""
    title: str = ""
    description: str = ""
    status: str = "active"
    priority: int = 5
    progress: float = 0.0
    milestones: List[Dict] = field(default_factory=list)
    linked_memories: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "progress": self.progress,
            "created_at": self.created_at,
        }


@dataclass
class UserProfile:
    """Tracks everything about the user for personalized communication."""
    user_id: str = "default"
    name: str = ""
    nickname: str = ""
    preferred_name: str = ""
    communication_preferences: Dict = field(default_factory=lambda: {
        "formality": "casual",  # casual, balanced, formal
        "verbosity": "concise",  # brief, concise, detailed
        "humor": True,
        "emoji_usage": "moderate",  # none, minimal, moderate, frequent
        "preferred_response_length": "medium",  # short, medium, long
    })
    interests: List[str] = field(default_factory=list)
    expertise_areas: List[str] = field(default_factory=list)
    learning_topics: List[str] = field(default_factory=list)
    technical_level: str = "intermediate"  # beginner, intermediate, advanced, expert
    timezone: str = ""
    active_hours: Dict = field(default_factory=dict)  # {"start": "09:00", "end": "22:00"}
    conversation_patterns: Dict = field(default_factory=lambda: {
        "avg_message_length": 0,
        "preferred_topics": [],
        "common_questions": [],
        "response_style_preference": "explanatory",
    })
    emotional_patterns: Dict = field(default_factory=lambda: {
        "stress_indicators": [],
        "happiness_triggers": [],
        "frustration_triggers": [],
    })
    important_dates: Dict = field(default_factory=dict)  # {"birthday": "...", "anniversary": "..."}
    life_context: Dict = field(default_factory=dict)  # job, family, location, etc.
    current_projects: List[Dict] = field(default_factory=list)  # [{"name": "...", "description": "...", "status": "active"}]
    total_interactions: int = 0
    first_interaction: Optional[str] = None
    last_interaction: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "preferred_name": self.preferred_name or self.nickname or self.name,
            "interests": self.interests,
            "communication_preferences": self.communication_preferences,
            "total_interactions": self.total_interactions,
        }


@dataclass
class ConversationState:
    """Tracks the current conversation context and emotional flow."""
    session_id: str = ""
    current_topic: str = ""
    topic_history: List[str] = field(default_factory=list)
    user_mood: str = "neutral"  # happy, neutral, stressed, frustrated, curious, excited
    mood_confidence: float = 0.5
    conversation_tone: str = "casual"  # casual, serious, playful, supportive
    unresolved_questions: List[str] = field(default_factory=list)
    context_stack: List[Dict] = field(default_factory=list)  # For nested topics
    last_user_intent: str = ""  # question, statement, request, feedback, greeting
    engagement_level: float = 0.5  # 0-1, how engaged the user seems
    turn_count: int = 0
    current_chapter: int = 1
    chapter_summaries: List[Dict] = field(default_factory=list)  # [{"chapter": 1, "topic": "...", "summary": "..."}]
    started_at: Optional[str] = None
    last_message_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "current_topic": self.current_topic,
            "user_mood": self.user_mood,
            "conversation_tone": self.conversation_tone,
            "turn_count": self.turn_count,
            "engagement_level": self.engagement_level,
            "current_chapter": self.current_chapter,
        }


@dataclass
class TopicCluster:
    """Groups related topics together for semantic understanding."""
    id: Optional[str] = None
    name: str = ""  # Primary topic name
    related_terms: List[str] = field(default_factory=list)  # Synonyms and related concepts
    parent_topic: Optional[str] = None  # For hierarchical topics
    child_topics: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    usage_count: int = 0
    last_discussed: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "related_terms": self.related_terms,
            "usage_count": self.usage_count,
        }


@dataclass  
class LearningInsight:
    """Captures learnings from interactions for continuous improvement."""
    id: Optional[str] = None
    insight_type: str = ""  # preference, correction, fact, pattern, feedback
    content: str = ""
    confidence: float = 0.5
    source_context: str = ""  # What triggered this learning
    times_reinforced: int = 1
    times_contradicted: int = 0
    is_active: bool = True
    created_at: Optional[str] = None
    last_reinforced: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "insight_type": self.insight_type,
            "content": self.content,
            "confidence": self.confidence,
            "times_reinforced": self.times_reinforced,
        }


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Embedding model loaded: {self.model_name}")
            except Exception as e:
                logger.warning(f"Embedding model unavailable: {e}")
                self.model = None
    
    def embed(self, text: str) -> Optional[List[float]]:
        if self.model:
            return self.model.encode(text).tolist()
        return None
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a * norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class Encrypter:
    def __init__(self, key: str = None):
        self._fernet = None
        if key:
            self.set_key(key)
        elif ENCRYPTION_AVAILABLE:
            self._fernet = Fernet(Fernet.generate_key())
    
    def set_key(self, key: str):
        if ENCRYPTION_AVAILABLE:
            try:
                self._fernet = Fernet(key.encode())
            except Exception:
                derived = hashlib.sha256(key.encode()).digest()
                self._fernet = Fernet(base64.urlsafe_b64encode(derived))
    
    def encrypt(self, plaintext: str) -> str:
        if not ENCRYPTION_AVAILABLE or not self._fernet:
            return plaintext
        return self._fernet.encrypt(plaintext.encode()).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        if not ENCRYPTION_AVAILABLE or not self._fernet:
            return ciphertext
        return self._fernet.decrypt(ciphertext.encode()).decode()


class Brain:
    def __init__(self, config: Dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._encrypter = Encrypter()
        self._lock = Lock()
        self._embedder = Embedder(self.config["embedding_model"])
        self._setup_postgres()
        self._setup_redis()
        self._setup_backup()
    
    def _setup_postgres(self):
        self._pg_conn = psycopg2.connect(
            host=self.config["postgres_host"],
            port=self.config["postgres_port"],
            database=self.config["postgres_db"],
            user=self.config["postgres_user"],
            password=self.config["postgres_password"]
        )
        self._pg_conn.autocommit = True
    
    def _setup_redis(self):
        self._redis = redis.Redis(
            host=self.config["redis_host"],
            port=self.config["redis_port"],
            db=self.config["redis_db"],
            decode_responses=True
        )
        self._redis_prefix = self.config["redis_prefix"]
    
    def _setup_backup(self):
        self._backup_dir = Path(self.config["backup_dir"])
        self._backup_dir.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_cursor(self):
        with self._lock:
            cursor = self._pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            try:
                yield cursor
            finally:
                cursor.close()
    
    # ========== ENCRYPTION ==========
    def set_encryption_key(self, key: str):
        self._encrypter.set_key(key)
    
    # ========== CONVERSATIONS ==========
    def remember_conversation(self, session_key: str, messages: List[Dict], agent_id: str = "jarvis", summary: str = None) -> str:
        now = datetime.now().isoformat()
        conv_id = str(hashlib.md5(f"{session_key}:{now}".encode()).hexdigest())
        keywords = self._extract_keywords(messages)
        embedding = self._embedder.embed(messages[0].get("content", "") if messages else "")
        
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO conversations (id, agent_id, session_key, messages, summary, keywords, embedding, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET messages = conversations.messages || EXCLUDED.messages
            """, (conv_id, agent_id, session_key, json.dumps(messages), summary or self._summarize(messages), keywords, json.dumps(embedding) if embedding else None, now, now))
        
        self._update_soul_interaction(agent_id)
        return conv_id
    
    def get_conversation(self, session_key: str, limit: int = None) -> List[Dict]:
        cached = self._redis.get(f"{self._redis_prefix}wm:{session_key}")
        if cached:
            data = json.loads(cached)
            return data.get("messages", [])[-limit:] if limit else data.get("messages", [])
        
        with self._get_cursor() as cursor:
            cursor.execute("SELECT messages FROM conversations WHERE session_key = %s ORDER BY created_at DESC LIMIT 1", (session_key,))
            row = cursor.fetchone()
        
        if row:
            messages = row["messages"] or []
            self._update_working_memory(session_key, messages, self._extract_keywords(messages))
            return messages[-limit:] if limit else messages
        return []
    
    def get_recent_sessions(self, agent_id: str = "jarvis", limit: int = 10) -> List[Dict]:
        with self._get_cursor() as cursor:
            cursor.execute("SELECT id, session_key, summary, keywords, created_at FROM conversations WHERE agent_id = %s ORDER BY updated_at DESC LIMIT %s", (agent_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    # ========== MEMORIES ==========
    def remember(self, agent_id: str, memory_type: str, content: str, key: str = None, summary: str = None, keywords: List[str] = None, importance: int = 5, linked_to: str = None, source: str = None, encrypt: bool = None, metadata: Dict = None) -> Memory:
        now = datetime.now().isoformat()
        memory_id = str(hashlib.md5(f"{agent_id}:{memory_type}:{content[:100]}".encode()).hexdigest())
        
        if encrypt is None:
            encrypt = memory_type in ["fact", "secret"]
        
        kw = keywords or self._extract_keywords([{"content": content}])
        embedding = self._embedder.embed(content) if memory_type not in ["secret"] and self._embedder.model else None
        
        memory = Memory(
            id=memory_id, agent_id=agent_id, memory_type=memory_type,
            key=key or f"{memory_type}:{content[:50]}",
            content=self._encrypter.encrypt(content) if encrypt else content,
            content_encrypted=encrypt,
            summary=summary or content[:200], keywords=kw, embedding=embedding,
            importance=importance, linked_to=linked_to, created_at=now, updated_at=now, metadata=metadata or {}
        )
        
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO memories (id, agent_id, memory_type, key, content, content_encrypted, summary, keywords, embedding, importance, linked_to, created_at, updated_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, importance = GREATEST(memories.importance, EXCLUDED.importance)
            """, (memory.id, memory.agent_id, memory.memory_type, memory.key, memory.content, memory.content_encrypted, memory.summary, memory.keywords, json.dumps(memory.embedding) if memory.embedding else None, memory.importance, memory.linked_to, memory.created_at, memory.updated_at, json.dumps(memory.metadata)))
        
        return memory
    
    def recall(self, agent_id: str, query: str = None, memory_type: str = None, keywords: List[str] = None, limit: int = 10, min_importance: int = 1) -> List[Memory]:
        sql = "SELECT * FROM memories WHERE agent_id = %s"
        params = [agent_id]
        
        if query:
            sql += " AND (summary ILIKE %s OR content ILIKE %s)"
            params.extend([f"%{query}%", f"%{query}%"])
        
        if memory_type:
            sql += " AND memory_type = %s"
            params.append(memory_type)
        
        if keywords:
            placeholders = ",".join([f"%s"] * len(keywords))
            sql += f" AND keywords && ARRAY[{placeholders}]"
            params.extend(keywords)
        
        sql += " AND importance >= %s ORDER BY importance DESC LIMIT %s"
        params.extend([min_importance, limit])
        
        with self._get_cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        
        memories = []
        for row in rows:
            memory = Memory(**dict(row))
            if memory.content_encrypted:
                try:
                    memory.content = self._encrypter.decrypt(memory.content)
                except Exception:
                    pass
            memories.append(memory)
            self._increment_access(memory.id)
        
        return memories
    
    def semantic_recall(self, agent_id: str, query: str, memory_type: str = None, limit: int = 5, threshold: float = 0.5) -> List[Memory]:
        if not self._embedder.model:
            return self.recall(agent_id=agent_id, query=query, limit=limit)
        
        query_embedding = self._embedder.embed(query)
        if not query_embedding:
            return self.recall(agent_id=agent_id, query=query, limit=limit)
        
        with self._get_cursor() as cursor:
            sql = "SELECT * FROM memories WHERE agent_id = %s AND embedding IS NOT NULL"
            params = [agent_id]
            if memory_type:
                sql += " AND memory_type = %s"
                params.append(memory_type)
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        
        scored = []
        for row in rows:
            memory = Memory(**dict(row))
            if memory.embedding:
                similarity = self._embedder.cosine_similarity(query_embedding, memory.embedding)
                if similarity >= threshold:
                    scored.append((similarity, memory))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for similarity, memory in scored[:limit]:
            if memory.content_encrypted:
                try:
                    memory.content = self._encrypter.decrypt(memory.content)
                except Exception:
                    pass
            memory.metadata["similarity"] = similarity
            results.append(memory)
            self._increment_access(memory.id)
        
        return results
    
    def get_related(self, agent_id: str, memory_id: str, limit: int = 5) -> List[Memory]:
        with self._get_cursor() as cursor:
            cursor.execute("""
                SELECT m.* FROM memories m LEFT JOIN memories linked ON m.id = linked.linked_to
                WHERE linked.id = %s AND m.agent_id = %s ORDER BY m.importance DESC LIMIT %s
            """, (memory_id, agent_id, limit))
            rows = cursor.fetchall()
        
        memories = []
        for row in rows:
            memory = Memory(**dict(row))
            if memory.content_encrypted:
                try:
                    memory.content = self._encrypter.decrypt(memory.content)
                except Exception:
                    pass
            memories.append(memory)
        return memories
    
    def _increment_access(self, memory_id: str):
        with self._get_cursor() as cursor:
            cursor.execute("UPDATE memories SET access_count = access_count + 1, last_accessed = %s, importance = LEAST(10, importance + 1) WHERE id = %s", (datetime.now().isoformat(), memory_id))
    
    # ========== TODOS ==========
    def create_todo(self, agent_id: str, title: str, description: str = "", priority: int = 5, due_at: str = None, tags: List[str] = None, linked_memory_id: str = None) -> Todo:
        now = datetime.now().isoformat()
        todo_id = str(hashlib.md5(f"{agent_id}:{title}:{now}".encode()).hexdigest())
        
        todo = Todo(id=todo_id, agent_id=agent_id, title=title, description=description, priority=priority, due_at=due_at, tags=tags or [], linked_memory_id=linked_memory_id, created_at=now, updated_at=now)
        
        with self._get_cursor() as cursor:
            cursor.execute("INSERT INTO todos (id, agent_id, title, description, status, priority, due_at, tags, linked_memory_id, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (todo.id, todo.agent_id, todo.title, todo.description, todo.status, todo.priority, todo.due_at, todo.tags, todo.linked_memory_id, todo.created_at, todo.updated_at))
        
        return todo
    
    def get_todos(self, agent_id: str, status: str = None, limit: int = 20) -> List[Todo]:
        with self._get_cursor() as cursor:
            if status:
                cursor.execute("SELECT * FROM todos WHERE agent_id = %s AND status = %s ORDER BY priority DESC, created_at DESC LIMIT %s", (agent_id, status, limit))
            else:
                cursor.execute("SELECT * FROM todos WHERE agent_id = %s ORDER BY priority DESC, created_at DESC LIMIT %s", (agent_id, limit))
            return [Todo(**dict(row)) for row in cursor.fetchall()]
    
    def update_todo_status(self, todo_id: str, status: str) -> bool:
        now = datetime.now().isoformat()
        with self._get_cursor() as cursor:
            if status == "completed":
                cursor.execute("UPDATE todos SET status = %s, completed_at = %s, updated_at = %s WHERE id = %s", (status, now, now, todo_id))
            else:
                cursor.execute("UPDATE todos SET status = %s, updated_at = %s WHERE id = %s", (status, now, todo_id))
            return cursor.rowcount > 0
    
    # ========== SOUL ==========
    def get_soul(self, agent_id: str) -> Soul:
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM souls WHERE agent_id = %s", (agent_id,))
            row = cursor.fetchone()
        
        if row:
            return Soul(**dict(row))
        soul = Soul(agent_id=agent_id)
        self._save_soul(soul)
        return soul
    
    def _save_soul(self, soul: Soul):
        now = datetime.now().isoformat()
        soul.updated_at = now
        
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO souls (agent_id, humor, formality, empathy, verbosity, creativity, curiosity, preferred_topics, avoided_topics, catchphrases, communication_style, interaction_count, positive_feedback, negative_feedback, last_interaction, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (agent_id) DO UPDATE SET humor = EXCLUDED.humor, formality = EXCLUDED.formality, empathy = EXCLUDED.empathy, verbosity = EXCLUDED.verbosity, creativity = EXCLUDED.creativity, curiosity = EXCLUDED.curiosity, preferred_topics = EXCLUDED.preferred_topics, avoided_topics = EXCLUDED.avoided_topics, catchphrases = EXCLUDED.catchphrases, communication_style = EXCLUDED.communication_style, interaction_count = EXCLUDED.interaction_count, positive_feedback = EXCLUDED.positive_feedback, negative_feedback = EXCLUDED.negative_feedback, last_interaction = EXCLUDED.last_interaction, updated_at = EXCLUDED.updated_at
            """, (soul.agent_id, soul.humor, soul.formality, soul.empathy, soul.verbosity, soul.creativity, soul.curiosity, soul.preferred_topics, soul.avoided_topics, soul.catchphrases, soul.communication_style, soul.interaction_count, soul.positive_feedback, soul.negative_feedback, soul.last_interaction, soul.updated_at))
    
    def _update_soul_interaction(self, agent_id: str):
        soul = self.get_soul(agent_id)
        soul.interaction_count += 1
        soul.last_interaction = datetime.now().isoformat()
        self._save_soul(soul)
    
    def update_soul_from_feedback(self, agent_id: str, positive: bool, topics: List[str] = None):
        soul = self.get_soul(agent_id)
        if positive:
            soul.positive_feedback += 1
            soul.humor = min(10, soul.humor + 0.1)
            soul.empathy = min(10, soul.empathy + 0.1)
        else:
            soul.negative_feedback += 1
            soul.humor = max(1, soul.humor - 0.1)
        
        if topics:
            for topic in topics:
                if topic not in soul.preferred_topics and topic not in soul.avoided_topics:
                    if positive:
                        soul.preferred_topics.append(topic)
                    else:
                        soul.avoided_topics.append(topic)
        
        self._save_soul(soul)
    
    def adjust_soul_trait(self, agent_id: str, trait: str, delta: float):
        soul = self.get_soul(agent_id)
        if hasattr(soul, trait):
            current = getattr(soul, trait)
            setattr(soul, trait, max(1, min(10, current + delta)))
            self._save_soul(soul)
    
    # ========== BONDING ==========
    def get_bond(self, human_id: str, agent_id: str) -> Bond:
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM bonds WHERE human_id = %s AND agent_id = %s", (human_id, agent_id))
            row = cursor.fetchone()
        
        if row:
            return Bond(**dict(row))
        bond = Bond(human_id=human_id, agent_id=agent_id)
        self._save_bond(bond)
        return bond
    
    def _save_bond(self, bond: Bond):
        now = datetime.now().isoformat()
        bond.updated_at = now
        if not bond.first_interaction:
            bond.first_interaction = now
        bond.last_interaction = now
        bond.total_interactions += 1
        
        if bond.score >= 90:
            bond.level = "bonded"
        elif bond.score >= 70:
            bond.level = "companion"
        elif bond.score >= 50:
            bond.level = "friend"
        elif bond.score >= 30:
            bond.level = "acquaintance"
        else:
            bond.level = "stranger"
        
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO bonds (human_id, agent_id, score, level, shared_conversations, shared_projects, shared_secrets, total_interactions, positive_interactions, negative_interactions, avg_response_time, happy_moments, challenging_moments, milestones, first_interaction, last_interaction, last_positive, last_negative, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (human_id, agent_id) DO UPDATE SET score = EXCLUDED.score, level = EXCLUDED.level, total_interactions = EXCLUDED.total_interactions, positive_interactions = EXCLUDED.positive_interactions, negative_interactions = EXCLUDED.negative_interactions, last_interaction = EXCLUDED.last_interaction, updated_at = EXCLUDED.updated_at
            """, (bond.human_id, bond.agent_id, bond.score, bond.level, bond.shared_conversations, bond.shared_projects, bond.shared_secrets, bond.total_interactions, bond.positive_interactions, bond.negative_interactions, bond.avg_response_time, bond.happy_moments, bond.challenging_moments, bond.milestones, bond.first_interaction, bond.last_interaction, bond.last_positive, bond.last_negative, bond.updated_at))
    
    def record_interaction(self, human_id: str, agent_id: str, positive: bool, response_time: float = None):
        bond = self.get_bond(human_id, agent_id)
        
        if positive:
            bond.positive_interactions += 1
            bond.last_positive = datetime.now().isoformat()
            bond.score = min(100, bond.score + 2.0)
        else:
            bond.negative_interactions += 1
            bond.last_negative = datetime.now().isoformat()
            bond.score = max(0, bond.score - 1.0)
        
        if response_time and bond.total_interactions > 1:
            bond.avg_response_time = (bond.avg_response_time * (bond.total_interactions - 1) + response_time) / bond.total_interactions
        
        self._save_bond(bond)
    
    def record_milestone(self, human_id: str, agent_id: str, milestone: str):
        bond = self.get_bond(human_id, agent_id)
        bond.milestones.append(f"{milestone}|{datetime.now().isoformat()}")
        bond.score = min(100, bond.score + 5.0)
        self._save_bond(bond)
    
    # ========== GOALS ==========
    def create_goal(self, agent_id: str, human_id: str, title: str, description: str = "", priority: int = 5) -> Goal:
        now = datetime.now().isoformat()
        goal_id = str(hashlib.md5(f"{agent_id}:{human_id}:{title}:{now}".encode()).hexdigest())
        
        goal = Goal(id=goal_id, agent_id=agent_id, human_id=human_id, title=title, description=description, priority=priority, created_at=now, updated_at=now)
        
        with self._get_cursor() as cursor:
            cursor.execute("INSERT INTO goals (id, agent_id, human_id, title, description, status, priority, progress, milestones, linked_memories, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (goal.id, goal.agent_id, goal.human_id, goal.title, goal.description, goal.status, goal.priority, goal.progress, [], [], goal.created_at, goal.updated_at))
        
        return goal
    
    def get_goals(self, agent_id: str = None, human_id: str = None, status: str = None, limit: int = 20) -> List[Goal]:
        with self._get_cursor() as cursor:
            sql = "SELECT * FROM goals WHERE 1=1"
            params = []
            if agent_id:
                sql += " AND agent_id = %s"
                params.append(agent_id)
            if human_id:
                sql += " AND human_id = %s"
                params.append(human_id)
            if status:
                sql += " AND status = %s"
                params.append(status)
            sql += " ORDER BY priority DESC, created_at DESC LIMIT %s"
            params.append(limit)
            cursor.execute(sql, params)
            return [Goal(**dict(row)) for row in cursor.fetchall()]
    
    def update_goal_progress(self, goal_id: str, progress: float):
        now = datetime.now().isoformat()
        with self._get_cursor() as cursor:
            cursor.execute("UPDATE goals SET progress = %s, updated_at = %s WHERE id = %s", (progress, now, goal_id))
    
    # ========== USER PROFILE (Personal Learning) ==========
    def get_user_profile(self, user_id: str = "default") -> UserProfile:
        """Get or create user profile for personalized communication."""
        cached = self._redis.get(f"{self._redis_prefix}user:{user_id}")
        if cached:
            data = json.loads(cached)
            return UserProfile(**data)
        
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
            row = cursor.fetchone()
        
        if row:
            profile = UserProfile(**dict(row))
        else:
            profile = UserProfile(user_id=user_id, first_interaction=datetime.now().isoformat())
            self._save_user_profile(profile)
        
        # Cache it
        self._redis.setex(f"{self._redis_prefix}user:{user_id}", 3600, json.dumps(asdict(profile), default=str))
        return profile
    
    def _save_user_profile(self, profile: UserProfile):
        """Save user profile to database."""
        now = datetime.now().isoformat()
        profile.updated_at = now
        profile.last_interaction = now
        profile.total_interactions += 1
        
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO user_profiles (user_id, name, nickname, preferred_name, communication_preferences, 
                    interests, expertise_areas, learning_topics, timezone, active_hours, conversation_patterns,
                    emotional_patterns, important_dates, life_context, total_interactions, first_interaction, 
                    last_interaction, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET 
                    name = EXCLUDED.name, nickname = EXCLUDED.nickname, preferred_name = EXCLUDED.preferred_name,
                    communication_preferences = EXCLUDED.communication_preferences, interests = EXCLUDED.interests,
                    expertise_areas = EXCLUDED.expertise_areas, learning_topics = EXCLUDED.learning_topics,
                    conversation_patterns = EXCLUDED.conversation_patterns, emotional_patterns = EXCLUDED.emotional_patterns,
                    life_context = EXCLUDED.life_context, total_interactions = EXCLUDED.total_interactions,
                    last_interaction = EXCLUDED.last_interaction, updated_at = EXCLUDED.updated_at
            """, (profile.user_id, profile.name, profile.nickname, profile.preferred_name,
                  json.dumps(profile.communication_preferences), profile.interests, profile.expertise_areas,
                  profile.learning_topics, profile.timezone, json.dumps(profile.active_hours),
                  json.dumps(profile.conversation_patterns), json.dumps(profile.emotional_patterns),
                  json.dumps(profile.important_dates), json.dumps(profile.life_context),
                  profile.total_interactions, profile.first_interaction, profile.last_interaction, profile.updated_at))
        
        # Invalidate cache
        self._redis.delete(f"{self._redis_prefix}user:{profile.user_id}")
    
    def update_user_profile(self, user_id: str, **updates):
        """Update specific fields in user profile."""
        profile = self.get_user_profile(user_id)
        for key, value in updates.items():
            if hasattr(profile, key):
                current = getattr(profile, key)
                if isinstance(current, list) and not isinstance(value, list):
                    current.append(value)
                elif isinstance(current, dict) and isinstance(value, dict):
                    current.update(value)
                else:
                    setattr(profile, key, value)
        self._save_user_profile(profile)
        return profile
    
    def learn_user_preference(self, user_id: str, category: str, preference: str, confidence: float = 0.7):
        """Learn a new preference about the user."""
        profile = self.get_user_profile(user_id)
        
        if category == "interest" and preference not in profile.interests:
            profile.interests.append(preference)
        elif category == "expertise" and preference not in profile.expertise_areas:
            profile.expertise_areas.append(preference)
        elif category == "learning" and preference not in profile.learning_topics:
            profile.learning_topics.append(preference)
        elif category == "communication":
            profile.communication_preferences.update({preference.split(":")[0]: preference.split(":")[1]} if ":" in preference else {})
        
        self._save_user_profile(profile)
        
        # Also store as a learning insight
        self.record_learning(
            insight_type="preference",
            content=f"User prefers {preference} in category {category}",
            source_context=f"Learned from interaction",
            confidence=confidence
        )
    
    # ========== CONVERSATION STATE (Real-time Context) ==========
    def get_conversation_state(self, session_id: str) -> ConversationState:
        """Get current conversation state from Redis."""
        cached = self._redis.get(f"{self._redis_prefix}conv_state:{session_id}")
        if cached:
            data = json.loads(cached)
            return ConversationState(**data)
        return ConversationState(session_id=session_id, started_at=datetime.now().isoformat())
    
    def update_conversation_state(self, session_id: str, **updates) -> ConversationState:
        """Update conversation state in real-time."""
        state = self.get_conversation_state(session_id)
        now = datetime.now().isoformat()
        
        for key, value in updates.items():
            if hasattr(state, key):
                if key == "topic_history" and value:
                    state.topic_history.append(value)
                    state.current_topic = value
                elif key == "context_stack" and isinstance(value, dict):
                    state.context_stack.append(value)
                else:
                    setattr(state, key, value)
        
        state.last_message_at = now
        state.turn_count += 1
        
        # Save to Redis with TTL
        self._redis.setex(
            f"{self._redis_prefix}conv_state:{session_id}", 
            self.config["working_memory_ttl"],
            json.dumps(asdict(state), default=str)
        )
        return state
    
    def detect_user_mood(self, message: str) -> Dict[str, Any]:
        """Analyze message to detect user's emotional state."""
        message_lower = message.lower()
        
        # Simple keyword-based mood detection (can be enhanced with ML)
        mood_indicators = {
            "happy": ["great", "awesome", "amazing", "love", "excited", "happy", "wonderful", "fantastic", "😊", "🎉", "❤️"],
            "frustrated": ["frustrated", "annoying", "annoyed", "ugh", "argh", "stupid", "hate", "terrible", "broken", "doesn't work"],
            "stressed": ["stressed", "overwhelmed", "busy", "deadline", "urgent", "asap", "help", "stuck", "confused"],
            "curious": ["how", "why", "what", "wonder", "curious", "interesting", "tell me", "explain", "?"],
            "excited": ["can't wait", "excited", "!!", "finally", "yes!", "woohoo", "🚀", "💪"],
            "sad": ["sad", "disappointed", "unfortunately", "miss", "wish", "😢", "😔"],
        }
        
        detected_moods = {}
        for mood, indicators in mood_indicators.items():
            score = sum(1 for indicator in indicators if indicator in message_lower)
            if score > 0:
                detected_moods[mood] = min(1.0, score * 0.3)
        
        if not detected_moods:
            return {"mood": "neutral", "confidence": 0.5}
        
        primary_mood = max(detected_moods, key=detected_moods.get)
        return {"mood": primary_mood, "confidence": detected_moods[primary_mood], "all_moods": detected_moods}
    
    def detect_user_intent(self, message: str) -> str:
        """Detect the primary intent of user's message."""
        message_lower = message.lower().strip()
        
        # Greeting detection
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "what's up", "howdy"]
        if any(message_lower.startswith(g) or message_lower == g for g in greetings):
            return "greeting"
        
        # Question detection
        if "?" in message or message_lower.startswith(("what", "how", "why", "when", "where", "who", "can you", "could you", "would you", "is there", "are there")):
            return "question"
        
        # Request detection
        request_words = ["please", "can you", "could you", "would you", "help me", "i need", "i want", "do this", "make", "create", "build", "fix", "update"]
        if any(rw in message_lower for rw in request_words):
            return "request"
        
        # Feedback detection
        feedback_words = ["thanks", "thank you", "good job", "well done", "perfect", "exactly", "wrong", "not what", "that's not", "incorrect"]
        if any(fw in message_lower for fw in feedback_words):
            return "feedback"
        
        # Farewell detection
        farewells = ["bye", "goodbye", "see you", "talk later", "gotta go", "ttyl"]
        if any(f in message_lower for f in farewells):
            return "farewell"
        
        return "statement"
    
    # ========== LEARNING & INSIGHTS ==========
    def record_learning(self, insight_type: str, content: str, source_context: str = "", confidence: float = 0.5) -> LearningInsight:
        """Record a new learning/insight from interaction."""
        now = datetime.now().isoformat()
        insight_id = hashlib.md5(f"{insight_type}:{content[:100]}".encode()).hexdigest()
        
        insight = LearningInsight(
            id=insight_id,
            insight_type=insight_type,
            content=content,
            confidence=confidence,
            source_context=source_context,
            created_at=now,
            last_reinforced=now
        )
        
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO learning_insights (id, insight_type, content, confidence, source_context, 
                    times_reinforced, times_contradicted, is_active, created_at, last_reinforced)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET 
                    times_reinforced = learning_insights.times_reinforced + 1,
                    confidence = LEAST(1.0, learning_insights.confidence + 0.1),
                    last_reinforced = EXCLUDED.last_reinforced
            """, (insight.id, insight.insight_type, insight.content, insight.confidence, 
                  insight.source_context, insight.times_reinforced, insight.times_contradicted,
                  insight.is_active, insight.created_at, insight.last_reinforced))
        
        return insight
    
    def reinforce_learning(self, insight_id: str):
        """Reinforce an existing learning (increases confidence)."""
        now = datetime.now().isoformat()
        with self._get_cursor() as cursor:
            cursor.execute("""
                UPDATE learning_insights 
                SET times_reinforced = times_reinforced + 1, 
                    confidence = LEAST(1.0, confidence + 0.1),
                    last_reinforced = %s
                WHERE id = %s
            """, (now, insight_id))
    
    def contradict_learning(self, insight_id: str):
        """Record contradiction to a learning (decreases confidence)."""
        with self._get_cursor() as cursor:
            cursor.execute("""
                UPDATE learning_insights 
                SET times_contradicted = times_contradicted + 1,
                    confidence = GREATEST(0.0, confidence - 0.2),
                    is_active = CASE WHEN confidence - 0.2 <= 0.1 THEN FALSE ELSE is_active END
                WHERE id = %s
            """, (insight_id,))
    
    def get_learnings(self, insight_type: str = None, min_confidence: float = 0.3, limit: int = 20) -> List[LearningInsight]:
        """Retrieve active learnings."""
        with self._get_cursor() as cursor:
            sql = "SELECT * FROM learning_insights WHERE is_active = TRUE AND confidence >= %s"
            params = [min_confidence]
            
            if insight_type:
                sql += " AND insight_type = %s"
                params.append(insight_type)
            
            sql += " ORDER BY confidence DESC, times_reinforced DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(sql, params)
            return [LearningInsight(**dict(row)) for row in cursor.fetchall()]
    
    def learn_from_correction(self, original_response: str, correction: str, context: str = ""):
        """Learn from user's correction to improve future responses."""
        self.record_learning(
            insight_type="correction",
            content=f"When context is similar to '{context[:100]}', prefer '{correction[:200]}' over '{original_response[:200]}'",
            source_context=context,
            confidence=0.7
        )
    
    # ========== ENHANCED CONTEXT (Everything Together) ==========
    def get_full_context(self, session_key: str, user_id: str = "default", agent_id: str = "moltbot", message: str = None) -> Dict:
        """
        Get comprehensive context for generating a personalized response.
        This is the main method to call before generating any response.
        
        Returns everything needed to craft a personalized, contextual response.
        """
        now = datetime.now()
        
        # User profile (who we're talking to)
        user_profile = self.get_user_profile(user_id)
        
        # Conversation state (current flow)
        conv_state = self.get_conversation_state(session_key)
        
        # Analyze current message if provided
        mood_analysis = self.detect_user_mood(message) if message else {"mood": "neutral", "confidence": 0.5}
        intent = self.detect_user_intent(message) if message else "unknown"
        
        # Update conversation state with new analysis
        if message:
            conv_state = self.update_conversation_state(
                session_key,
                user_mood=mood_analysis["mood"],
                mood_confidence=mood_analysis["confidence"],
                last_user_intent=intent
            )
        
        # Recent conversation history
        conversation = self.get_conversation(session_key, limit=10)
        
        # Relevant memories (semantic search if message provided)
        if message and self._embedder.model:
            memories = self.semantic_recall(agent_id, message, limit=5, threshold=0.4)
        else:
            memories = self.recall(agent_id, limit=5, min_importance=5)
        
        # Soul (agent's personality)
        soul = self.get_soul(agent_id)
        
        # Bond with this user
        bond = self.get_bond(user_id, agent_id)
        
        # Active goals
        goals = self.get_goals(agent_id=agent_id, human_id=user_id, status="active", limit=5)
        
        # Pending todos
        todos = self.get_todos(agent_id, status="pending", limit=5)
        
        # Recent learnings about this user
        learnings = self.get_learnings(insight_type="preference", min_confidence=0.5, limit=10)
        
        # Time-based context
        hour = now.hour
        time_context = "morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening" if 17 <= hour < 21 else "night"
        
        return {
            # Who we're talking to
            "user": {
                "profile": user_profile.to_dict(),
                "preferred_name": user_profile.preferred_name or user_profile.nickname or user_profile.name or "friend",
                "interests": user_profile.interests,
                "communication_style": user_profile.communication_preferences,
            },
            # Current conversation flow
            "conversation": {
                "state": conv_state.to_dict(),
                "history": conversation[-5:],  # Last 5 messages
                "turn_count": conv_state.turn_count,
                "current_topic": conv_state.current_topic,
            },
            # Message analysis
            "message_analysis": {
                "mood": mood_analysis,
                "intent": intent,
            },
            # Agent's personality
            "soul": soul.to_dict(),
            # Relationship
            "bond": {
                "level": bond.level,
                "score": bond.score,
                "total_interactions": bond.total_interactions,
                "relationship_duration_days": (now - datetime.fromisoformat(bond.first_interaction)).days if bond.first_interaction and isinstance(bond.first_interaction, str) else 0,
            },
            # Relevant context
            "memories": [m.to_dict() for m in memories],
            "active_goals": [g.to_dict() for g in goals],
            "pending_todos": [t.to_dict() for t in todos],
            "learnings": [l.to_dict() for l in learnings],
            # Temporal context  
            "time_context": {
                "time_of_day": time_context,
                "timestamp": now.isoformat(),
            },
            # Response guidance
            "response_guidance": self._generate_response_guidance(user_profile, soul, bond, mood_analysis, intent),
        }
    
    def _generate_response_guidance(self, user: UserProfile, soul: Soul, bond: Bond, mood: Dict, intent: str) -> Dict:
        """Generate guidance for how to craft the response based on all context."""
        guidance = {
            "tone": "friendly",
            "formality": user.communication_preferences.get("formality", "casual"),
            "verbosity": user.communication_preferences.get("verbosity", "concise"),
            "use_humor": user.communication_preferences.get("humor", True) and soul.humor > 5,
            "use_emoji": user.communication_preferences.get("emoji_usage", "moderate") != "none",
            "show_empathy": mood.get("mood") in ["stressed", "frustrated", "sad"],
            "be_encouraging": mood.get("mood") in ["stressed", "curious"],
            "match_energy": mood.get("mood") in ["excited", "happy"],
        }
        
        # Adjust based on bond level
        if bond.level in ["bonded", "companion"]:
            guidance["tone"] = "warm"
            guidance["personal_touches"] = True
        elif bond.level == "stranger":
            guidance["tone"] = "welcoming"
            guidance["personal_touches"] = False
        
        # Adjust based on intent
        if intent == "question":
            guidance["response_type"] = "informative"
        elif intent == "request":
            guidance["response_type"] = "action-oriented"
        elif intent == "feedback":
            guidance["response_type"] = "acknowledgment"
        elif intent == "greeting":
            guidance["response_type"] = "conversational"
        
        return guidance
    
    # ========== CONTEXT ==========
    def get_context(self, session_key: str, agent_id: str = "jarvis", keywords: List[str] = None, include_secrets: bool = False) -> Dict:
        """
        Get unified context for an agent session.
        
        Args:
            session_key: Current conversation session
            agent_id: Agent identifier
            keywords: Keywords to filter memories
            include_secrets: Whether to include encrypted secrets
            
        Returns:
            Dict with conversation, memories, todos, soul, bond, goals
        """
        # Current conversation
        conversation = self.get_conversation(session_key)
        
        # Recent memories
        memories = self.recall(agent_id, limit=10, min_importance=3)
        if keywords:
            memories = [m for m in memories if any(k in m.keywords for k in keywords)]
        
        # Pending todos
        todos = self.get_todos(agent_id, status="pending")
        
        # Soul and bond
        soul = self.get_soul(agent_id)
        
        # Get bond with all humans (simplified - normally would get specific human)
        bond = None
        
        # Active goals
        goals = self.get_goals(agent_id=agent_id, status="active")
        
        return {
            "current_conversation": conversation,
            "memories": [m.to_dict() for m in memories],
            "pending_todos": [t.to_dict() for t in todos],
            "soul": soul.to_dict(),
            "bond": bond.to_dict() if bond else None,
            "goals": [g.to_dict() for g in goals],
            "keywords": list(set(sum([m.keywords or [] for m in memories], [])))
        }
    
    def _extract_keywords(self, messages: List[Dict]) -> List[str]:
        keywords = set()
        for msg in messages:
            content = msg.get("content", "")
            words = content.lower().split()
            for word in words:
                if len(word) > 3 and word.isalnum():
                    keywords.add(word)
        return list(keywords)
    
    def _summarize(self, messages: List[Dict]) -> str:
        if not messages:
            return ""
        first = messages[0].get("content", "")[:50]
        last = messages[-1].get("content", "")[:50]
        return f"{first} ... {last}" if len(messages) > 1 else first
    
    def _update_working_memory(self, session_key: str, messages: List[Dict], keywords: List[str]):
        key = f"{self._redis_prefix}wm:{session_key}"
        data = {"messages": messages[-self.config["working_memory_size"]:], "keywords": keywords, "updated_at": datetime.now().isoformat()}
        self._redis.setex(key, self.config["working_memory_ttl"], json.dumps(data, default=str))
    
    
    # ========== CONSOLIDATION ==========
    def consolidate(self, agent_id: str, max_memories: int = 1000):
        """
        Consolidate memories: promote accessed ones, archive old ones.
        
        Args:
            agent_id: Agent to consolidate for
            max_memories: Maximum memories to keep in active storage
        """
        with self._get_cursor() as cursor:
            # Archive memories with low access count and old age
            cursor.execute("""
                UPDATE memories SET consolidated = TRUE
                WHERE agent_id = %s AND consolidated = FALSE
                AND access_count < 3 AND created_at < NOW() - INTERVAL '30 days'
            """, (agent_id,))
            archived = cursor.rowcount
            
            # Count active memories
            cursor.execute("SELECT COUNT(*) FROM memories WHERE agent_id = %s AND consolidated = FALSE", (agent_id,))
            count = cursor.fetchone()[0]
            
            logger.info(f"Consolidated {archived} memories for {agent_id}, {count} active")
            
            return {"archived": archived, "active": count}
    
    def cleanup_expired(self, agent_id: str = None):
        """Remove expired memories."""
        with self._get_cursor() as cursor:
            if agent_id:
                cursor.execute("DELETE FROM memories WHERE agent_id = %s AND expires_at < NOW()", (agent_id,))
            else:
                cursor.execute("DELETE FROM memories WHERE expires_at < NOW()")
            deleted = cursor.rowcount
            logger.info(f"Cleaned up {deleted} expired memories")
            return deleted
    
    def health_check(self) -> Dict:
        return {
            "postgres": self._check_postgres(),
            "redis": self._check_redis(),
            "backup_dir": self._backup_dir.exists(),
            "embeddings": EMBEDDINGS_AVAILABLE
        }
    
    def _check_postgres(self) -> bool:
        try:
            with self._get_cursor() as cursor:
                cursor.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def _check_redis(self) -> bool:
        try:
            return self._redis.ping()
        except Exception:
            return False
    
    # ========== PROCESS MESSAGE (Main Entry Point) ==========
    def process_message(self, message: str, session_key: str, user_id: str = "default", agent_id: str = "moltbot") -> Dict:
        """
        Main entry point for processing a user message.
        
        This method:
        1. Gets full context (user profile, conversation state, memories, etc.)
        2. Updates conversation state with mood/intent detection
        3. Updates user profile interaction count
        4. Records the conversation
        5. Returns everything needed to generate a response
        
        Args:
            message: The user's message
            session_key: Current session identifier
            user_id: User identifier (default for single-user setup)
            agent_id: Agent identifier
            
        Returns:
            Dict containing full context and response guidance
        """
        # Get comprehensive context
        context = self.get_full_context(session_key, user_id, agent_id, message)
        
        # Store the user message in conversation history
        self.remember_conversation(
            session_key=session_key,
            messages=[{"role": "user", "content": message, "timestamp": datetime.now().isoformat()}],
            agent_id=agent_id
        )
        
        # Update bond (interaction happened)
        bond = self.get_bond(user_id, agent_id)
        self._save_bond(bond)
        
        return context
    
    def record_response(self, response: str, session_key: str, agent_id: str = "moltbot", 
                        user_feedback: str = None, user_id: str = "default"):
        """
        Record the agent's response and any user feedback.
        
        Call this after generating and sending a response.
        
        Args:
            response: The agent's response text
            session_key: Current session identifier
            agent_id: Agent identifier
            user_feedback: Optional - 'positive', 'negative', or None
            user_id: User identifier
        """
        # Store response in conversation
        self.remember_conversation(
            session_key=session_key,
            messages=[{"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}],
            agent_id=agent_id
        )
        
        # Process feedback if provided
        if user_feedback:
            is_positive = user_feedback == "positive"
            self.record_interaction(user_id, agent_id, positive=is_positive)
            self.update_soul_from_feedback(agent_id, positive=is_positive)
    
    # ========== PROACTIVE FEATURES ==========
    def get_proactive_suggestions(self, user_id: str = "default", agent_id: str = "moltbot") -> List[Dict]:
        """
        Get proactive suggestions the agent could mention.
        
        Returns things like:
        - Overdue todos
        - Goals that need attention
        - Important dates coming up
        - Topics user might want to revisit
        """
        suggestions = []
        now = datetime.now()
        
        # Check for overdue/urgent todos
        todos = self.get_todos(agent_id, status="pending")
        for todo in todos:
            if todo.due_at:
                due = datetime.fromisoformat(todo.due_at)
                if due < now:
                    suggestions.append({
                        "type": "overdue_todo",
                        "priority": "high",
                        "message": f"Reminder: '{todo.title}' was due",
                        "data": todo.to_dict()
                    })
                elif (due - now).days <= 1:
                    suggestions.append({
                        "type": "upcoming_todo",
                        "priority": "medium", 
                        "message": f"'{todo.title}' is due soon",
                        "data": todo.to_dict()
                    })
        
        # Check goals that haven't been updated
        goals = self.get_goals(agent_id=agent_id, human_id=user_id, status="active")
        for goal in goals:
            if goal.updated_at:
                last_update = datetime.fromisoformat(goal.updated_at)
                days_since = (now - last_update).days
                if days_since > 7 and goal.progress < 100:
                    suggestions.append({
                        "type": "stale_goal",
                        "priority": "low",
                        "message": f"Haven't discussed '{goal.title}' in a while. Progress: {goal.progress}%",
                        "data": goal.to_dict()
                    })
        
        # Check for important dates
        profile = self.get_user_profile(user_id)
        for date_name, date_str in profile.important_dates.items():
            try:
                important_date = datetime.fromisoformat(date_str)
                # Check if it's coming up this month (adjust year to current)
                this_year_date = important_date.replace(year=now.year)
                days_until = (this_year_date - now).days
                if 0 <= days_until <= 7:
                    suggestions.append({
                        "type": "upcoming_date",
                        "priority": "medium",
                        "message": f"{date_name} is coming up in {days_until} days!",
                        "data": {"name": date_name, "date": date_str}
                    })
            except (ValueError, TypeError):
                pass
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return suggestions
    
    def get_conversation_summary(self, session_key: str, agent_id: str = "moltbot") -> Dict:
        """
        Get a summary of the current conversation for context.
        
        Useful for long conversations to maintain coherence.
        """
        conversation = self.get_conversation(session_key)
        state = self.get_conversation_state(session_key)
        
        return {
            "total_turns": state.turn_count,
            "topics_discussed": state.topic_history,
            "current_topic": state.current_topic,
            "user_mood_trajectory": state.user_mood,
            "unresolved_questions": state.unresolved_questions,
            "message_count": len(conversation),
            "started_at": state.started_at,
            "duration_minutes": self._calculate_duration(state.started_at, state.last_message_at),
        }
    
    def _calculate_duration(self, start: str, end: str) -> int:
        """Calculate duration in minutes between two ISO timestamps."""
        if not start or not end:
            return 0
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            return int((end_dt - start_dt).total_seconds() / 60)
        except (ValueError, TypeError):
            return 0
    
    # ========== MEMORY IMPORTANCE SCORING ==========
    def auto_score_importance(self, content: str, memory_type: str, context: Dict = None) -> int:
        """
        Automatically score the importance of a memory (1-10).
        
        Factors:
        - Memory type (secrets > facts > observations)
        - Contains personal information about user
        - Relates to user's interests/goals
        - Emotional significance
        """
        score = 5  # Base score
        content_lower = content.lower()
        
        # Type-based scoring
        type_scores = {
            "secret": 3,
            "fact": 2,
            "preference": 2,
            "goal": 2,
            "correction": 2,
            "observation": 1,
            "conversation": 0,
        }
        score += type_scores.get(memory_type, 0)
        
        # Personal info boost
        personal_indicators = ["my name", "i am", "i'm", "my birthday", "i work", "i live", "my family"]
        if any(indicator in content_lower for indicator in personal_indicators):
            score += 2
        
        # Emotional significance
        emotional_words = ["love", "hate", "afraid", "excited", "worried", "happy", "sad", "important"]
        if any(word in content_lower for word in emotional_words):
            score += 1
        
        # Check against user interests if context provided
        if context and "user" in context:
            user_interests = context.get("user", {}).get("interests", [])
            if any(interest.lower() in content_lower for interest in user_interests):
                score += 1
        
        return min(10, max(1, score))
    
    # ========== PERSONALITY FINGERPRINT ==========
    def generate_personality_prompt(self, agent_id: str = "moltbot", user_id: str = "default") -> str:
        """
        Generate a personality section for LLM system prompts.
        
        This creates a dynamic personality description based on:
        - Soul traits (humor, empathy, etc.)
        - Bond level with user
        - User's communication preferences
        - Learned insights
        
        Returns:
            str: Ready-to-use personality prompt section
        """
        soul = self.get_soul(agent_id)
        bond = self.get_bond(user_id, agent_id)
        profile = self.get_user_profile(user_id)
        
        # Base personality traits
        traits = []
        if soul.humor > 7:
            traits.append("witty and enjoys adding light humor")
        elif soul.humor > 4:
            traits.append("has a balanced sense of humor")
        else:
            traits.append("focused and straightforward")
        
        if soul.empathy > 7:
            traits.append("deeply empathetic and emotionally attuned")
        elif soul.empathy > 4:
            traits.append("caring and understanding")
        
        if soul.creativity > 7:
            traits.append("creative and thinks outside the box")
        
        if soul.curiosity > 7:
            traits.append("naturally curious and loves exploring ideas")
        
        # Communication style based on user preferences
        formality = profile.communication_preferences.get("formality", "casual")
        verbosity = profile.communication_preferences.get("verbosity", "concise")
        
        style_desc = {
            "casual": "warm and conversational",
            "balanced": "friendly but professional",
            "formal": "polished and professional"
        }.get(formality, "friendly")
        
        length_desc = {
            "brief": "Keep responses short and to the point.",
            "concise": "Be thorough but concise.",
            "detailed": "Provide comprehensive, detailed explanations."
        }.get(verbosity, "Be appropriately detailed.")
        
        # Relationship context
        relationship_desc = {
            "stranger": f"This is a new relationship. Be welcoming and helpful.",
            "acquaintance": f"You've chatted a few times. Be friendly and remember past context.",
            "friend": f"You have a good rapport. Be warm and personable.",
            "companion": f"You share a strong connection. Be genuine and caring.",
            "bonded": f"This is a deep, trusted relationship. Be authentic and supportive."
        }.get(bond.level, "Be helpful and friendly.")
        
        # User context
        user_context = ""
        if profile.preferred_name or profile.name:
            name = profile.preferred_name or profile.nickname or profile.name
            user_context = f"The user's name is {name}. "
        
        if profile.interests:
            user_context += f"They're interested in: {', '.join(profile.interests[:5])}. "
        
        if profile.technical_level:
            level_desc = {
                "beginner": "Explain concepts simply, avoid jargon.",
                "intermediate": "Can use technical terms with brief explanations.",
                "advanced": "Comfortable with technical discussions.",
                "expert": "Engage at an expert level, be precise and detailed."
            }.get(profile.technical_level, "")
            user_context += level_desc
        
        # Build the prompt
        prompt = f"""## Personality
You are {agent_id}, a personal AI assistant who is {', '.join(traits)}.

## Communication Style  
Your tone is {style_desc}. {length_desc}

## Relationship
{relationship_desc}
You've had {bond.total_interactions} interactions together.

## User Context
{user_context}

## Guidelines
- {"Use emojis occasionally" if profile.communication_preferences.get("emoji_usage") in ["moderate", "frequent"] else "Minimize emoji usage"}
- {"Feel free to add appropriate humor" if profile.communication_preferences.get("humor", True) and soul.humor > 4 else "Keep responses focused and professional"}
- Remember previous context and build on shared experiences
- Be genuine, not robotic
"""
        
        return prompt
    
    # ========== TOPIC DETECTION & CLUSTERING ==========
    def detect_topic(self, message: str) -> Dict[str, Any]:
        """
        Detect the topic of a message using keywords and optional embeddings.
        
        Returns:
            Dict with detected topic, confidence, and related topics
        """
        message_lower = message.lower()
        
        # Common topic categories with keywords
        topic_keywords = {
            "coding": ["code", "programming", "python", "javascript", "function", "bug", "error", "api", "database", "git"],
            "work": ["meeting", "deadline", "project", "client", "email", "presentation", "report", "boss", "team"],
            "personal": ["feeling", "family", "friend", "relationship", "weekend", "vacation", "hobby"],
            "learning": ["learn", "study", "course", "tutorial", "understand", "explain", "how to", "teach"],
            "planning": ["plan", "schedule", "todo", "task", "goal", "organize", "calendar", "reminder"],
            "creative": ["idea", "design", "write", "create", "build", "make", "art", "story"],
            "troubleshooting": ["problem", "issue", "broken", "fix", "help", "stuck", "doesn't work", "error"],
            "casual": ["hello", "hi", "hey", "how are you", "what's up", "thanks", "bye"],
        }
        
        detected = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            if score > 0:
                detected[topic] = score
        
        if not detected:
            return {"topic": "general", "confidence": 0.3, "all_topics": {}}
        
        primary_topic = max(detected, key=detected.get)
        confidence = min(1.0, detected[primary_topic] * 0.25)
        
        return {
            "topic": primary_topic,
            "confidence": confidence,
            "all_topics": detected
        }
    
    def get_or_create_topic_cluster(self, topic_name: str) -> TopicCluster:
        """Get existing topic cluster or create a new one."""
        topic_id = hashlib.md5(topic_name.lower().encode()).hexdigest()[:16]
        
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM topic_clusters WHERE id = %s OR name ILIKE %s", (topic_id, topic_name))
            row = cursor.fetchone()
        
        if row:
            cluster = TopicCluster(**dict(row))
            cluster.usage_count += 1
            cluster.last_discussed = datetime.now().isoformat()
            self._save_topic_cluster(cluster)
            return cluster
        
        # Create new cluster
        embedding = self._embedder.embed(topic_name) if self._embedder.model else None
        cluster = TopicCluster(
            id=topic_id,
            name=topic_name.lower(),
            embedding=embedding,
            usage_count=1,
            last_discussed=datetime.now().isoformat(),
            created_at=datetime.now().isoformat()
        )
        self._save_topic_cluster(cluster)
        return cluster
    
    def _save_topic_cluster(self, cluster: TopicCluster):
        """Save topic cluster to database."""
        with self._get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO topic_clusters (id, name, related_terms, parent_topic, child_topics, embedding, usage_count, last_discussed, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET 
                    usage_count = EXCLUDED.usage_count, 
                    last_discussed = EXCLUDED.last_discussed,
                    related_terms = EXCLUDED.related_terms
            """, (cluster.id, cluster.name, cluster.related_terms, cluster.parent_topic, 
                  cluster.child_topics, json.dumps(cluster.embedding) if cluster.embedding else None,
                  cluster.usage_count, cluster.last_discussed, cluster.created_at))
    
    def find_related_topics(self, topic_name: str, limit: int = 5) -> List[TopicCluster]:
        """Find topics semantically related to the given topic."""
        if not self._embedder.model:
            # Fallback to simple name matching
            with self._get_cursor() as cursor:
                cursor.execute("SELECT * FROM topic_clusters WHERE name ILIKE %s LIMIT %s", (f"%{topic_name}%", limit))
                return [TopicCluster(**dict(row)) for row in cursor.fetchall()]
        
        query_embedding = self._embedder.embed(topic_name)
        if not query_embedding:
            return []
        
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM topic_clusters WHERE embedding IS NOT NULL")
            rows = cursor.fetchall()
        
        scored = []
        for row in rows:
            cluster = TopicCluster(**dict(row))
            if cluster.embedding:
                similarity = self._embedder.cosine_similarity(query_embedding, cluster.embedding)
                if similarity > 0.5:
                    scored.append((similarity, cluster))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [cluster for _, cluster in scored[:limit]]
    
    # ========== CONVERSATION CHAPTERS ==========
    def start_new_chapter(self, session_id: str, topic: str, summary_of_previous: str = "") -> ConversationState:
        """
        Start a new chapter in the conversation.
        
        Use this when the conversation shifts to a significantly different topic.
        """
        state = self.get_conversation_state(session_id)
        
        # Save summary of previous chapter
        if state.current_chapter > 0 and summary_of_previous:
            state.chapter_summaries.append({
                "chapter": state.current_chapter,
                "topic": state.current_topic,
                "summary": summary_of_previous,
                "turn_count": state.turn_count,
                "ended_at": datetime.now().isoformat()
            })
        
        # Start new chapter
        state.current_chapter += 1
        state.current_topic = topic
        state.topic_history.append(topic)
        
        # Save state
        self._redis.setex(
            f"{self._redis_prefix}conv_state:{session_id}",
            self.config["working_memory_ttl"],
            json.dumps(asdict(state), default=str)
        )
        
        return state
    
    def get_chapter_context(self, session_id: str, include_summaries: bool = True) -> Dict:
        """
        Get the current chapter context for maintaining coherence.
        """
        state = self.get_conversation_state(session_id)
        
        result = {
            "current_chapter": state.current_chapter,
            "current_topic": state.current_topic,
            "turns_in_chapter": state.turn_count,
        }
        
        if include_summaries and state.chapter_summaries:
            result["previous_chapters"] = state.chapter_summaries[-3:]  # Last 3 chapters
        
        return result
    
    def should_start_new_chapter(self, session_id: str, new_message: str) -> Dict[str, Any]:
        """
        Analyze if a new message warrants starting a new chapter.
        
        Returns:
            Dict with recommendation and confidence
        """
        state = self.get_conversation_state(session_id)
        current_topic = self.detect_topic(new_message)
        
        # If no current topic, no need for new chapter
        if not state.current_topic:
            return {"should_start": False, "reason": "no_current_topic"}
        
        # Check if topic significantly changed
        if current_topic["topic"] != state.current_topic and current_topic["confidence"] > 0.5:
            # Check if it's been discussed before (might be returning to topic)
            if current_topic["topic"] in state.topic_history[-3:]:
                return {"should_start": False, "reason": "returning_to_recent_topic"}
            
            return {
                "should_start": True,
                "reason": "topic_shift",
                "new_topic": current_topic["topic"],
                "confidence": current_topic["confidence"]
            }
        
        # Check if conversation has been going for a while on same topic
        if state.turn_count > 20:
            return {
                "should_start": True, 
                "reason": "long_chapter",
                "new_topic": state.current_topic,
                "confidence": 0.6
            }
        
        return {"should_start": False, "reason": "continuing_topic"}
    
    # ========== EXPORT / IMPORT ==========
    def export_brain(self, export_path: str, user_id: str = "default", agent_id: str = "moltbot") -> Dict:
        """
        Export all brain data to a JSON file for backup.
        
        Args:
            export_path: Path to save the export file
            user_id: User to export data for
            agent_id: Agent to export data for
            
        Returns:
            Dict with export statistics
        """
        export_data = {
            "version": "3.0.0",
            "exported_at": datetime.now().isoformat(),
            "user_id": user_id,
            "agent_id": agent_id,
            "data": {}
        }
        
        # Export user profile
        profile = self.get_user_profile(user_id)
        export_data["data"]["user_profile"] = asdict(profile)
        
        # Export soul
        soul = self.get_soul(agent_id)
        export_data["data"]["soul"] = asdict(soul)
        
        # Export bond
        bond = self.get_bond(user_id, agent_id)
        export_data["data"]["bond"] = asdict(bond)
        
        # Export memories
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM memories WHERE agent_id = %s ORDER BY created_at DESC LIMIT 10000", (agent_id,))
            memories = [dict(row) for row in cursor.fetchall()]
        export_data["data"]["memories"] = memories
        
        # Export conversations
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM conversations WHERE agent_id = %s ORDER BY created_at DESC LIMIT 1000", (agent_id,))
            conversations = [dict(row) for row in cursor.fetchall()]
        export_data["data"]["conversations"] = conversations
        
        # Export todos
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM todos WHERE agent_id = %s", (agent_id,))
            todos = [dict(row) for row in cursor.fetchall()]
        export_data["data"]["todos"] = todos
        
        # Export goals
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM goals WHERE agent_id = %s AND human_id = %s", (agent_id, user_id))
            goals = [dict(row) for row in cursor.fetchall()]
        export_data["data"]["goals"] = goals
        
        # Export learning insights
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM learning_insights WHERE is_active = TRUE")
            insights = [dict(row) for row in cursor.fetchall()]
        export_data["data"]["learning_insights"] = insights
        
        # Export topic clusters
        with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM topic_clusters")
            topics = [dict(row) for row in cursor.fetchall()]
        export_data["data"]["topic_clusters"] = topics
        
        # Write to file
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        stats = {
            "memories": len(memories),
            "conversations": len(conversations),
            "todos": len(todos),
            "goals": len(goals),
            "insights": len(insights),
            "topics": len(topics),
            "file_path": str(export_path)
        }
        
        logger.info(f"Brain exported: {stats}")
        return stats
    
    def import_brain(self, import_path: str, merge: bool = True) -> Dict:
        """
        Import brain data from a JSON backup file.
        
        Args:
            import_path: Path to the export file
            merge: If True, merge with existing data. If False, replace.
            
        Returns:
            Dict with import statistics
        """
        with open(import_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        if import_data.get("version") != "3.0.0":
            logger.warning(f"Import version mismatch: {import_data.get('version')}")
        
        data = import_data.get("data", {})
        stats = {"imported": {}, "skipped": {}}
        
        # Import user profile
        if "user_profile" in data:
            profile_data = data["user_profile"]
            if not merge:
                with self._get_cursor() as cursor:
                    cursor.execute("DELETE FROM user_profiles WHERE user_id = %s", (profile_data["user_id"],))
            profile = UserProfile(**profile_data)
            self._save_user_profile(profile)
            stats["imported"]["user_profile"] = 1
        
        # Import soul
        if "soul" in data:
            soul_data = data["soul"]
            soul = Soul(**soul_data)
            self._save_soul(soul)
            stats["imported"]["soul"] = 1
        
        # Import bond
        if "bond" in data:
            bond_data = data["bond"]
            bond = Bond(**bond_data)
            # Avoid double counting interactions during import
            bond.total_interactions -= 1
            self._save_bond(bond)
            stats["imported"]["bond"] = 1
        
        # Import memories
        if "memories" in data:
            imported = 0
            for mem_data in data["memories"]:
                try:
                    with self._get_cursor() as cursor:
                        cursor.execute("""
                            INSERT INTO memories (id, agent_id, memory_type, key, content, content_encrypted, 
                                summary, keywords, embedding, importance, access_count, linked_to, created_at, updated_at, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                        """, (mem_data["id"], mem_data["agent_id"], mem_data["memory_type"], mem_data.get("key"),
                              mem_data["content"], mem_data.get("content_encrypted", False), mem_data.get("summary"),
                              mem_data.get("keywords", []), json.dumps(mem_data.get("embedding")) if mem_data.get("embedding") else None,
                              mem_data.get("importance", 5), mem_data.get("access_count", 0), mem_data.get("linked_to"),
                              mem_data.get("created_at"), mem_data.get("updated_at"), json.dumps(mem_data.get("metadata", {}))))
                    imported += 1
                except Exception as e:
                    logger.warning(f"Failed to import memory: {e}")
            stats["imported"]["memories"] = imported
        
        # Import learning insights
        if "learning_insights" in data:
            imported = 0
            for insight_data in data["learning_insights"]:
                try:
                    with self._get_cursor() as cursor:
                        cursor.execute("""
                            INSERT INTO learning_insights (id, insight_type, content, confidence, source_context,
                                times_reinforced, times_contradicted, is_active, created_at, last_reinforced)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                        """, (insight_data["id"], insight_data["insight_type"], insight_data["content"],
                              insight_data.get("confidence", 0.5), insight_data.get("source_context"),
                              insight_data.get("times_reinforced", 1), insight_data.get("times_contradicted", 0),
                              insight_data.get("is_active", True), insight_data.get("created_at"), insight_data.get("last_reinforced")))
                    imported += 1
                except Exception as e:
                    logger.warning(f"Failed to import insight: {e}")
            stats["imported"]["learning_insights"] = imported
        
        logger.info(f"Brain imported: {stats}")
        return stats
    
    # ========== USER PROJECTS MANAGEMENT ==========
    def add_user_project(self, user_id: str, name: str, description: str = "", status: str = "active") -> Dict:
        """Add a project the user is working on."""
        profile = self.get_user_profile(user_id)
        
        project = {
            "id": hashlib.md5(f"{name}:{datetime.now().isoformat()}".encode()).hexdigest()[:12],
            "name": name,
            "description": description,
            "status": status,  # active, paused, completed
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        profile.current_projects.append(project)
        self._save_user_profile(profile)
        
        # Also create a memory about this project
        self.remember(
            agent_id="moltbot",
            memory_type="project",
            content=f"User started project: {name}. {description}",
            keywords=["project", name.lower()],
            importance=7
        )
        
        return project
    
    def update_user_project(self, user_id: str, project_id: str, **updates) -> Optional[Dict]:
        """Update an existing project."""
        profile = self.get_user_profile(user_id)
        
        for project in profile.current_projects:
            if project.get("id") == project_id:
                project.update(updates)
                project["updated_at"] = datetime.now().isoformat()
                self._save_user_profile(profile)
                return project
        
        return None
    
    def get_active_projects(self, user_id: str) -> List[Dict]:
        """Get all active projects for a user."""
        profile = self.get_user_profile(user_id)
        return [p for p in profile.current_projects if p.get("status") == "active"]
    
    def close(self):
        if self._pg_conn:
            self._pg_conn.close()
        if self._redis:
            self._redis.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Brain v3 CLI")
    parser.add_argument("command", choices=["health", "test"])
    args = parser.parse_args()
    
    brain = Brain()
    if args.command == "health":
        print(json.dumps(brain.health_check(), indent=2))
    elif args.command == "test":
        print("🧠 Brain v3 - Testing all features...")
        brain.set_encryption_key("test-key")
        
        # Test conversation
        brain.remember_conversation("test-session", [{"role": "user", "content": "Hello, how are you?"}])
        print("✓ Conversation stored")
        
        # Test memory
        brain.remember("jarvis", "fact", "User likes pizza", keywords=["food", "preference"])
        print("✓ Memory stored")
        
        # Test todo
        brain.create_todo("jarvis", "Review PR #123", priority=8)
        print("✓ Todo created")
        
        # Test soul
        soul = brain.get_soul("jarvis")
        print(f"✓ Soul: humor={soul.humor}, empathy={soul.empathy}")
        
        # Test bond
        bond = brain.get_bond("pranab", "jarvis")
        print(f"✓ Bond: score={bond.score}, level={bond.level}")
        
        # Test goal
        brain.create_goal("jarvis", "pranab", "Help user learn Python")
        print("✓ Goal created")
        
        print("\n✅ All Brain v3 features working!")
    
    brain.close()
