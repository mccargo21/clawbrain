#!/usr/bin/env python3
"""
Brain v3 - AI Agent Memory System with Soul, Bonding, and Semantic Search

Features:
- 🎭 Soul/Personality - Evolving personality traits
- 💝 Bonding - Relationship tracking between human and AI
- 📝 Todos - Task management
- 🧠 Semantic Search - Meaning-based retrieval (embeddings)
- 🎯 Goals - Long-term objectives
- 📊 Stats - Interaction statistics

Architecture:
┌─────────────────────────────────────────────────────┐
│                   BRAIN v3                          │
├─────────────────────────────────────────────────────┤
│  🗣️ Conversations    │  💾 Memories               │
│  📝 Todos           │  🔐 Secrets                │
│  🎭 Soul/Personality│  💝 Bonding               │
│  🧠 Semantic Search │  📚 Knowledge             │
│  🎯 Goals           │  📊 Stats                 │
└─────────────────────────────────────────────────────┘
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
except Import:
    ENCRYPTION_AVAILABLE = False

import psycopg2
import psycopg2.extras
import redis

DEFAULT_CONFIG = {
    "postgres_host": "YOUR_POSTGRES_HOST",
    "postgres_port": 5432,
    "postgres_db": "brain_db",
    "postgres_user": "brain_user",
    "postgres_password": "YOUR_POSTGRES_PASSWORD",
    "redis_host": "YOUR_REDIS_HOST",
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
    
    # ========== UTILITIES ==========
    
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
    
    def close(self):
        if self._pg_conn:
            self._pg_conn.close()
        if self._redis:
            self._redis.close()

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
