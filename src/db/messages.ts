import type { MessageRecord, OpenAIChatMessage, TokenUsage } from "../types";
import { sha256Hex } from "../utils/hash";
import { newId } from "../utils/ids";
import { nowIso } from "../utils/time";

function contentToText(content: OpenAIChatMessage["content"]): string {
  if (typeof content === "string") return content;
  if (content == null) return "";
  return JSON.stringify(content);
}

export async function saveUserMessages(
  db: D1Database,
  input: {
    conversationId: string;
    namespace: string;
    source: string;
    messages: OpenAIChatMessage[];
    requestModel: string;
    upstreamModel: string;
    upstreamProvider: string;
    stream: boolean;
  }
): Promise<string[]> {
  const lastUserMessage = [...input.messages].reverse().find((message) => message.role === "user");
  const userMessages = lastUserMessage ? [lastUserMessage] : [];
  const ids: string[] = [];

  for (const message of userMessages) {
    const content = contentToText(message.content);
    const id = newId("msg");
    const hash = await sha256Hex(`${input.conversationId}:${id}:${message.role}:${content}`);
    ids.push(id);

    await db
      .prepare(
        `INSERT INTO messages (
          id, conversation_id, namespace, role, content, source, client_message_hash,
          upstream_model, upstream_provider, request_model, stream, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .bind(
        id,
        input.conversationId,
        input.namespace,
        "user",
        content,
        input.source,
        hash,
        input.upstreamModel,
        input.upstreamProvider,
        input.requestModel,
        input.stream ? 1 : 0,
        nowIso()
      )
      .run();
  }

  return ids;
}

export async function saveAssistantMessage(
  db: D1Database,
  input: {
    conversationId: string;
    namespace: string;
    source: string;
    content: string;
    requestModel: string;
    upstreamModel: string;
    provider: string;
    stream: boolean;
    finishReason?: string | null;
    usage?: TokenUsage;
    cacheMode?: string | null;
    cacheTtl?: string | null;
  }
): Promise<string> {
  const id = newId("msg");
  const usage = input.usage || {};

  await db
    .prepare(
      `INSERT INTO messages (
        id, conversation_id, namespace, role, content, source, upstream_model,
        upstream_provider, request_model, stream, finish_reason, token_input,
        token_output, cache_mode, cache_ttl, cache_hit, cache_read_tokens,
        cache_creation_tokens, raw_usage_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .bind(
      id,
      input.conversationId,
      input.namespace,
      "assistant",
      input.content,
      input.source,
      input.upstreamModel,
      input.provider,
      input.requestModel,
      input.stream ? 1 : 0,
      input.finishReason || null,
      usage.prompt_tokens ?? usage.input_tokens ?? null,
      usage.completion_tokens ?? usage.output_tokens ?? null,
      input.cacheMode ?? null,
      input.cacheTtl ?? null,
      typeof usage.cache_read_input_tokens === "number" && usage.cache_read_input_tokens > 0 ? 1 : 0,
      usage.cache_read_input_tokens ?? null,
      usage.cache_creation_input_tokens ?? null,
      JSON.stringify(usage),
      nowIso()
    )
    .run();

  return id;
}

export async function getMessagesByIds(
  db: D1Database,
  input: { namespace: string; ids: string[] }
): Promise<MessageRecord[]> {
  if (input.ids.length === 0) return [];

  const placeholders = input.ids.map(() => "?").join(", ");
  const result = await db
    .prepare(
      `SELECT id, conversation_id, namespace, role, content, source, created_at
       FROM messages
       WHERE namespace = ? AND id IN (${placeholders})
       ORDER BY created_at ASC`
    )
    .bind(input.namespace, ...input.ids)
    .all<MessageRecord>();

  return result.results ?? [];
}

export async function countMessagesAfterTimestamp(
  db: D1Database,
  namespace: string,
  afterCreatedAt: string | null
): Promise<number> {
  if (!afterCreatedAt) {
    const row = await db
      .prepare(
        `SELECT COUNT(*) as cnt FROM messages
         WHERE namespace = ? AND role IN ('user', 'assistant')`
      )
      .bind(namespace)
      .first<{ cnt: number }>();
    return row?.cnt ?? 0;
  }

  const row = await db
    .prepare(
      `SELECT COUNT(*) as cnt FROM messages
       WHERE namespace = ? AND role IN ('user', 'assistant') AND created_at > ?`
    )
    .bind(namespace, afterCreatedAt)
    .first<{ cnt: number }>();
  return row?.cnt ?? 0;
}

export async function listMessagesByNamespace(
  db: D1Database,
  namespace: string,
  afterCreatedAt: string | null,
  limit: number
): Promise<MessageRecord[]> {
  let sql = `SELECT id, conversation_id, namespace, role, content, source, created_at
             FROM messages
             WHERE namespace = ? AND role IN ('user', 'assistant')`;
  const binds: unknown[] = [namespace];

  if (afterCreatedAt) {
    sql += ` AND created_at > ?`;
    binds.push(afterCreatedAt);
  }

  sql += ` ORDER BY created_at ASC LIMIT ?`;
  binds.push(limit);

  const result = await db.prepare(sql).bind(...binds).all<MessageRecord>();
  return result.results ?? [];
}

export async function listMessagesByNamespaceInRange(
  db: D1Database,
  input: {
    namespace: string;
    startCreatedAt: string;
    endCreatedAt: string;
    afterCreatedAt?: string | null;
    limit: number;
  }
): Promise<MessageRecord[]> {
  let sql = `SELECT id, conversation_id, namespace, role, content, source, created_at
             FROM messages
             WHERE namespace = ?
               AND role IN ('user', 'assistant')
               AND created_at >= ?
               AND created_at < ?`;
  const binds: unknown[] = [input.namespace, input.startCreatedAt, input.endCreatedAt];

  if (input.afterCreatedAt) {
    sql += ` AND created_at > ?`;
    binds.push(input.afterCreatedAt);
  }

  sql += ` ORDER BY created_at ASC LIMIT ?`;
  binds.push(input.limit);

  const result = await db.prepare(sql).bind(...binds).all<MessageRecord>();
  return result.results ?? [];
}

// 关键词搜原始对话台账(④):content 子串匹配,可选时间范围,offset 偏移翻页。
// 给哥哥的 memory_transcript 用——看「当时逐字原话」,补 ③ 记忆都是提炼过的之缺。
export async function searchMessagesByKeyword(
  db: D1Database,
  input: {
    namespace: string;
    keyword: string;
    startCreatedAt?: string | null;
    endCreatedAt?: string | null;
    order?: "asc" | "desc";
    limit: number;
    offset: number;
  }
): Promise<MessageRecord[]> {
  // 转义 LIKE 的通配符(% _)和转义符(\),免得关键词含这些字符时被当通配符。中文一般无影响。
  const escaped = input.keyword.replace(/[\\%_]/g, (ch) => `\\${ch}`);
  let sql = `SELECT id, conversation_id, namespace, role, content, source, created_at
             FROM messages
             WHERE namespace = ?
               AND role IN ('user', 'assistant')
               AND content LIKE ? ESCAPE '\\'`;
  const binds: unknown[] = [input.namespace, `%${escaped}%`];

  if (input.startCreatedAt) {
    sql += ` AND created_at >= ?`;
    binds.push(input.startCreatedAt);
  }
  if (input.endCreatedAt) {
    sql += ` AND created_at < ?`;
    binds.push(input.endCreatedAt);
  }

  sql += ` ORDER BY created_at ${input.order === "desc" ? "DESC" : "ASC"} LIMIT ? OFFSET ?`;
  binds.push(input.limit, input.offset);

  const result = await db.prepare(sql).bind(...binds).all<MessageRecord>();
  return result.results ?? [];
}

// 给定一条台账消息 id，取它「前后各 N 条」上下文。按 created_at 全局时间相邻取
// (台账基本单源、created_at 是真实对话时序，时间相邻即同一段对话流;不锁 conversation_id
// 免得历史导入/换 session 把同段对话切成多个 conversation 时上下文在边界断)。
// memory_transcript 第二步「展开」用:先 keyword 搜到命中那句,再带它的 id 来看前因后果。
export async function getMessageContextById(
  db: D1Database,
  input: { namespace: string; id: string; before: number; after: number }
): Promise<{ hit: MessageRecord | null; context: MessageRecord[] }> {
  const hit = await db
    .prepare(
      `SELECT id, conversation_id, namespace, role, content, source, created_at
       FROM messages
       WHERE namespace = ? AND id = ?`
    )
    .bind(input.namespace, input.id)
    .first<MessageRecord>();
  if (!hit) return { hit: null, context: [] };

  const beforeRows = await db
    .prepare(
      `SELECT id, conversation_id, namespace, role, content, source, created_at
       FROM messages
       WHERE namespace = ? AND role IN ('user', 'assistant') AND created_at < ?
       ORDER BY created_at DESC LIMIT ?`
    )
    .bind(input.namespace, hit.created_at, input.before)
    .all<MessageRecord>();

  const afterRows = await db
    .prepare(
      `SELECT id, conversation_id, namespace, role, content, source, created_at
       FROM messages
       WHERE namespace = ? AND role IN ('user', 'assistant') AND created_at > ?
       ORDER BY created_at ASC LIMIT ?`
    )
    .bind(input.namespace, hit.created_at, input.after)
    .all<MessageRecord>();

  const before = (beforeRows.results ?? []).reverse();
  const after = afterRows.results ?? [];
  return { hit, context: [...before, hit, ...after] };
}

export async function saveIngestMessages(
  db: D1Database,
  input: {
    conversationId: string;
    namespace: string;
    source: string;
    messages: Array<{ role: string; content: string; created_at?: string }>;
  }
): Promise<string[]> {
  const ids: string[] = [];

  // 幂等去重(根治回流重复爆炸):给每条算稳定指纹 = sha256(角色:正文:时间)。
  // ⚠️ 指纹**绝不含 conversation_id/session_id**——爆炸的真实机制就是 compact/resume/换手机
  // 产生新 session、把整条 transcript 重发;只有指纹跨 session 一致,才能把这些重发去掉。
  // 先批量查已在库的指纹(client_message_hash 列有索引 idx_messages_hash,快),只插新的。
  // 这样客户端不管怎么跨 session/compact/远程重发,数据库都只存一条——彻底断根。
  const prepared: Array<{ id: string; role: string; content: string; createdAt: string; hash: string }> = [];
  for (const message of input.messages) {
    const content = contentToText(message.content);
    if (!content) continue;
    const hash = await sha256Hex(`${message.role}:${content}:${message.created_at ?? ""}`);
    prepared.push({
      id: newId("msg"),
      role: message.role,
      content,
      createdAt: message.created_at ?? nowIso(),
      hash
    });
  }
  if (prepared.length === 0) return ids;

  const seen = new Set<string>();
  const CHUNK = 100;
  for (let i = 0; i < prepared.length; i += CHUNK) {
    const slice = prepared.slice(i, i + CHUNK);
    const placeholders = slice.map(() => "?").join(", ");
    const rows = await db
      .prepare(
        `SELECT client_message_hash FROM messages
         WHERE namespace = ? AND client_message_hash IN (${placeholders})`
      )
      .bind(input.namespace, ...slice.map((p) => p.hash))
      .all<{ client_message_hash: string }>();
    for (const r of rows.results ?? []) {
      if (r.client_message_hash) seen.add(r.client_message_hash);
    }
  }

  for (const p of prepared) {
    if (seen.has(p.hash)) continue; // 已在库 → 跳过(幂等)
    seen.add(p.hash); // 同一批内自重复也只入一次
    ids.push(p.id);
    await db
      .prepare(
        `INSERT INTO messages (
          id, conversation_id, namespace, role, content, source, client_message_hash, stream, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .bind(
        p.id,
        input.conversationId,
        input.namespace,
        p.role,
        p.content,
        input.source,
        p.hash,
        0,
        p.createdAt
      )
      .run();
  }

  return ids;
}

// ---------------------------------------------------------------------------
// Admin: 原文台账(④)统计 / 定向清理。
// 仅用于"干净重建"——历史导入混入了 regenerate/编辑的废弃分支，需要按 source
// 整段清掉再重新导入主路径。普通运行不会调用这里。
// ---------------------------------------------------------------------------

export async function countMessagesBySource(
  db: D1Database,
  namespace: string
): Promise<Array<{ source: string | null; count: number; min_created_at: string | null; max_created_at: string | null }>> {
  const result = await db
    .prepare(
      `SELECT source, COUNT(*) AS count, MIN(created_at) AS min_created_at, MAX(created_at) AS max_created_at
       FROM messages
       WHERE namespace = ?
       GROUP BY source
       ORDER BY count DESC`
    )
    .bind(namespace)
    .all<{ source: string | null; count: number; min_created_at: string | null; max_created_at: string | null }>();
  return result.results ?? [];
}

export async function deleteMessagesBySource(
  db: D1Database,
  namespace: string,
  source: string
): Promise<number> {
  const result = await db
    .prepare("DELETE FROM messages WHERE namespace = ? AND source = ?")
    .bind(namespace, source)
    .run();
  return result.meta.changes ?? 0;
}

export async function deleteAllMessagesInNamespace(
  db: D1Database,
  namespace: string
): Promise<number> {
  const result = await db
    .prepare("DELETE FROM messages WHERE namespace = ?")
    .bind(namespace)
    .run();
  return result.meta.changes ?? 0;
}
