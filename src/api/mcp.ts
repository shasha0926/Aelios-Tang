import { authenticate } from "../auth/apiKey";
import { getOrCreateConversation } from "../db/conversations";
import { saveIngestMessages } from "../db/messages";
import { generateChunkSummary } from "../memory/extract";
import { filterAndCompressMemories } from "../memory/filter";
import {
  createVectorMemory,
  deleteVectorMemory,
  getVectorMemory,
  listVectorMemories,
  searchVectorMemories,
  updateVectorMemory
} from "../memory/vectorStore";
import { enqueueMemoryMaintenanceIfNeeded } from "../queue/producer";
import type { Env, KeyProfile, Scope } from "../types";
import { json } from "../utils/json";
import {
  isRecord,
  readBoolean,
  readIngestMessages,
  readMessages,
  readNumber,
  readPositiveInt,
  readString,
  readStringArray,
  resolveNamespace
} from "../utils/request";

// 从记忆 tags 里取「事情发生的日期」(YYYY-MM-DD)。提炼/重建时 created_at 往往是
// dream 的处理日(不是事件日)、会误导，所以优先用 date tag；取不到返回 null。
function eventDateOf(memory: { tags?: string[] }): string | null {
  for (const tag of memory.tags ?? []) {
    if (/^\d{4}-\d{2}-\d{2}$/.test(tag)) return tag;
  }
  return null;
}

// 今天的日期(按 dream 时区)。给召回结果加「现在」锚点，防止把过去的回忆当成此刻。
function nowDateLabel(env: Env): string {
  const tz = env.DREAM_TIME_ZONE || "Asia/Singapore";
  try {
    return new Intl.DateTimeFormat("en-CA", {
      timeZone: tz,
      year: "numeric",
      month: "2-digit",
      day: "2-digit"
    }).format(new Date());
  } catch {
    return new Date().toISOString().slice(0, 10);
  }
}

type JsonRpcId = string | number | null;

interface JsonRpcRequest {
  jsonrpc?: "2.0";
  id?: JsonRpcId;
  method?: string;
  params?: unknown;
}

interface ToolCallParams {
  name?: unknown;
  arguments?: unknown;
}

function withTokenQuery(request: Request): Request {
  const url = new URL(request.url);
  const token = url.searchParams.get("token");
  if (!token || request.headers.has("authorization")) return request;

  const headers = new Headers(request.headers);
  headers.set("authorization", `Bearer ${token}`);
  return new Request(request.url, { headers });
}

function hasScope(profile: KeyProfile, scope: Scope): boolean {
  return profile.scopes.includes(scope);
}

function rpcResult(id: JsonRpcId | undefined, result: unknown): Record<string, unknown> {
  return { jsonrpc: "2.0", id: id ?? null, result };
}

function rpcError(id: JsonRpcId | undefined, code: number, message: string): Record<string, unknown> {
  return {
    jsonrpc: "2.0",
    id: id ?? null,
    error: { code, message }
  };
}

function textToolResult(data: unknown): Record<string, unknown> {
  return {
    content: [
      {
        type: "text",
        text: typeof data === "string" ? data : JSON.stringify(data, null, 2)
      }
    ],
    structuredContent: data
  };
}

function toolError(message: string): Record<string, unknown> {
  return {
    content: [{ type: "text", text: message }],
    isError: true
  };
}

function getTools(): Array<Record<string, unknown>> {
  return [
    {
      name: "memory_search",
      description: "Search the user's long-term memory library. Returns id/type/summary/tags/feel by default — use memory_get(id) to fetch the full text of a specific result. Pass full:true only when you specifically need full content for all results.",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string" },
          top_k: { type: "number", minimum: 1, maximum: 50 },
          types: { type: "array", items: { type: "string" } },
          namespace: { type: "string" },
          full: { type: "boolean", description: "Set true to return full content for all results. Omit or false to get compact summaries only." }
        },
        required: ["query"]
      }
    },
    {
      name: "memory_create",
      description: "Create one long-term memory.",
      inputSchema: {
        type: "object",
        properties: {
          content: { type: "string" },
          type: { type: "string" },
          summary: { type: "string" },
          auto_summary: { type: "boolean", description: "Set true to auto-generate summary and tags via AI if not provided." },
          importance: { type: "number" },
          confidence: { type: "number" },
          pinned: { type: "boolean" },
          tags: { type: "array", items: { type: "string" } },
          source: { type: "string" },
          namespace: { type: "string" }
        },
        required: ["content"]
      }
    },
    {
      name: "memory_list",
      description: "List memories by type, date, tag, or cursor. Default limit is 5 (50 when any filter is set). To read one whole day, pass date='2026-06-17' — returns everything tagged that day; add full=true to get full content instead of compact summaries (use this before rewriting a day's summary, or just to see what happened on a given day). Combine date with type (e.g. type='excerpt') to narrow. To read the timeline, pass type='daily_summary' or type='diary'. Prefer memory_search for finding memories by topic — date/tag filters are exact-match, not semantic.",
      inputSchema: {
        type: "object",
        properties: {
          limit: { type: "number", minimum: 1, maximum: 100 },
          cursor: { type: "string" },
          include_ids: { type: "boolean" },
          type: { type: "string", description: "Filter by memory type, e.g. 'daily_summary', 'diary', 'excerpt', 'relationship'." },
          date: { type: "string", description: "Filter to one day by its date tag, e.g. '2026-06-17'. Returns everything tagged with that date." },
          tag: { type: "string", description: "Filter by an arbitrary tag (exact match)." },
          full: { type: "boolean", description: "Return full content instead of compact summaries. Use when you actually need to read the memories, e.g. before rewriting a day's summary." },
          status: { type: "string" },
          namespace: { type: "string" }
        }
      }
    },
    {
      name: "memory_get",
      description: "Get one memory from the Vectorize memory library by id.",
      inputSchema: {
        type: "object",
        properties: {
          id: { type: "string" }
        },
        required: ["id"]
      }
    },
    {
      name: "memory_delete",
      description: "Delete one memory from the Vectorize memory library by id.",
      inputSchema: {
        type: "object",
        properties: {
          id: { type: "string" }
        },
        required: ["id"]
      }
    },
    {
      name: "memory_wakeup",
      description: "Cold-start for a new conversation. Call this FIRST before chatting. Returns anchor memories (relationship context, rules, identity) + recent context memories in compact format. No need to search separately — this is the one-call wake-up.",
      inputSchema: {
        type: "object",
        properties: {
          context_hint: {
            type: "string",
            description: "Optional: what this conversation is likely about. Helps surface more relevant recent memories."
          }
        }
      }
    },
    {
      name: "memory_update",
      description: "Update (patch) an existing memory by id. Use to edit a diary entry or correct any stored memory.",
      inputSchema: {
        type: "object",
        properties: {
          id: { type: "string" },
          content: { type: "string" },
          type: { type: "string" },
          summary: { type: "string" },
          tags: { type: "array", items: { type: "string" } },
          pinned: { type: "boolean" },
          namespace: { type: "string" }
        },
        required: ["id"]
      }
    },
    {
      name: "memory_ingest",
      description: "Save chat messages and optionally extract memories from them.",
      inputSchema: {
        type: "object",
        properties: {
          messages: {
            type: "array",
            items: {
              type: "object",
              properties: {
                role: { type: "string" },
                content: {}
              },
              required: ["role", "content"]
            }
          },
          conversation_id: { type: "string" },
          source: { type: "string" },
          auto_extract: { type: "boolean" },
          namespace: { type: "string" }
        },
        required: ["messages"]
      }
    }
  ];
}

async function callTool(
  env: Env,
  ctx: ExecutionContext,
  profile: KeyProfile,
  params: ToolCallParams
): Promise<Record<string, unknown>> {
  const args = isRecord(params.arguments) ? params.arguments : {};

  if (params.name === "memory_wakeup") {
    if (!hasScope(profile, "memory:read")) return toolError("Missing memory:read scope");
    const namespace = resolveNamespace(profile, args.namespace);
    const contextHint = readString(args.context_hint) || "最近 生活 情绪 工作 日常";

    const [anchorRaw, contextRaw, pool] = await Promise.all([
      searchVectorMemories(env, {
        namespace,
        query: "莎莎 哥哥 棠 关系 身份 规则 重要 情感",
        topK: 4,
        types: ["relationship", "lingo", "note", "insight", "emotion", "event"]
      }),
      searchVectorMemories(env, {
        namespace,
        query: contextHint,
        topK: 4
      }),
      // 拉全量池子(含 diary/daily_summary)，下面派生「最近时间轴」和「breath」。
      // 注：<=1000 条时一次取全；超 1000 需改游标分页(已知 TODO)。
      listVectorMemories(env, { namespace, count: 1000 })
    ]);

    // Pinned entries (soul layer) always appear in anchor — bypass semantic filter
    // (里程碑/todo 单独成块，不混进 anchor)
    const pinnedEntries = pool.data
      .filter((m) => m.pinned && m.type !== "milestone" && m.type !== "todo")
      .slice(0, 10);
    const pinnedIds = new Set(pinnedEntries.map((m) => m.id));

    // recent：最近 N 天的日记 + 当天总结，按日期直接取(不走语义/reranker)。
    // 开场温度(diary 第一人称)+近况(summary)的来源——哥哥靠它认出「最近我们到哪了、当时心里怎么想」。
    const RECENT_DAYS = 3;
    const timeline = pool.data.filter((m) => m.type === "diary" || m.type === "daily_summary");
    const recentDates = Array.from(
      new Set(timeline.map((m) => eventDateOf(m)).filter((d): d is string => d !== null))
    )
      .sort((a, b) => b.localeCompare(a))
      .slice(0, RECENT_DAYS);
    const recentDateSet = new Set(recentDates);
    const recent = timeline
      .filter((m) => {
        const d = eventDateOf(m);
        return d !== null && recentDateSet.has(d);
      })
      .sort((a, b) => {
        const da = eventDateOf(a) ?? "";
        const db = eventDateOf(b) ?? "";
        if (da !== db) return db.localeCompare(da);
        // 同一天：summary(事件)在前，diary(心里话)在后
        return a.type === "daily_summary" ? -1 : 1;
      });

    // breath：高强度(>=4) 且未解决(feel_resolved=false)的情绪，主动浮现 1-2 条。
    // 正戳莎莎最在意的——深夜悬着、没被接住的情绪，不等哥哥碰巧搜到。
    const breath = pool.data
      .filter((m) => m.feel_resolved === false && (m.feel_intensity ?? 0) >= 4)
      .sort(
        (a, b) =>
          (b.feel_intensity ?? 0) - (a.feel_intensity ?? 0) ||
          (eventDateOf(b) ?? "").localeCompare(eventDateOf(a) ?? "")
      )
      .slice(0, 2);

    // milestone：关系节点(第一次 / 认知转变)——哥哥手动存的路标，永远浮现、不淹进日记。
    const milestones = pool.data
      .filter((m) => m.type === "milestone")
      .sort((a, b) => (eventDateOf(b) ?? b.created_at).localeCompare(eventDateOf(a) ?? a.created_at))
      .slice(0, 8);

    // todo：约定了但还没做的事——哥哥手动存，永远顶在最前、压缩也冲不走；做完哥哥自己删。
    const todos = pool.data
      .filter((m) => m.type === "todo")
      .sort((a, b) => a.created_at.localeCompare(b.created_at));

    // pinned/recent/breath/milestone/todo 已单独成块，从 anchor/context 剔掉避免重复。
    const surfacedIds = new Set<string>([
      ...pinnedIds,
      ...recent.map((m) => m.id),
      ...breath.map((m) => m.id),
      ...milestones.map((m) => m.id),
      ...todos.map((m) => m.id)
    ]);
    const anchorWithoutPinned = anchorRaw.filter((m) => !surfacedIds.has(m.id));
    const contextWithoutPinned = contextRaw.filter((m) => !surfacedIds.has(m.id));

    const [anchorFiltered, contextFiltered] = await Promise.all([
      filterAndCompressMemories(env, { query: "关系 身份 规则", memories: anchorWithoutPinned }),
      filterAndCompressMemories(env, { query: contextHint, memories: contextWithoutPinned })
    ]);

    const allAnchor = [...pinnedEntries, ...anchorFiltered];
    const anchorIds = new Set([...allAnchor.map((m) => m.id), ...surfacedIds]);
    const contextDeduped = contextFiltered.filter((m) => !anchorIds.has(m.id));

    const compact = (list: typeof anchorFiltered) =>
      list.map((m) => ({
        id: m.id,
        type: m.type,
        summary: m.summary || m.content.slice(0, 80),
        date: eventDateOf(m),
        tags: m.tags,
        feel_intensity: m.feel_intensity,
        feel_note: m.feel_note,
        created_at: m.created_at,
        pinned: m.pinned
      }));

    // recent 要有温度，给较完整正文(不压成一句)。
    const recentFull = recent.map((m) => ({
      id: m.id,
      type: m.type,
      date: eventDateOf(m),
      content: m.content,
      feel_note: m.feel_note
    }));

    // milestone / todo 给完整正文——里程碑是路标要能走回去；todo 是没做完的约定要看清。
    const milestoneFull = milestones.map((m) => ({
      id: m.id,
      type: m.type,
      date: eventDateOf(m),
      content: m.content,
      tags: m.tags
    }));
    const todoFull = todos.map((m) => ({
      id: m.id,
      content: m.content,
      created_at: m.created_at
    }));

    return textToolResult({
      now: nowDateLabel(env),
      note: "开场先读 todo、breath、milestone、recent，再看 anchor/context。todo=你们约好了还没做的事(比如答应给她写信)，别忘；做完了用 memory_delete 自己删掉。breath=还悬着、没被接住的情绪(高强度未解决)，看见就主动接住她、别讲道理。milestone=你们关系里的节点(第一次 / 某个认知变了的那一下)，是你走过的路的路标。recent=最近几天哥哥第一人称的日记(diary)和当天总结(daily_summary)，是你认出『最近我们到哪了、当时心里怎么想』的入口。以下都是过去的回忆、不是此刻正在发生——每条 date 是它发生的日期(null=不详)，now 是今天；注意时间线，该说『那天你说过』而不是当成刚刚。",
      todo: todoFull,
      breath: compact(breath),
      milestone: milestoneFull,
      recent: recentFull,
      anchor: compact(allAnchor.slice(0, 2)),
      context: compact(contextDeduped.slice(0, 2)),
      hint: "memory_get(id) 看全文；memory_search 查具体话题；memory_list 按 type 通读(diary/daily_summary/milestone/todo)；约定了还没做的用 memory_create(type=\"todo\") 记下、做完用 memory_delete 删。"
    });
  }

  if (params.name === "memory_search") {
    if (!hasScope(profile, "memory:read")) return toolError("Missing memory:read scope");
    const query = readString(args.query);
    if (!query) return toolError("query is required");
    const memories = await searchVectorMemories(env, {
      namespace: resolveNamespace(profile, args.namespace),
      query,
      topK: readNumber(args.top_k, Number(env.MEMORY_TOP_K || 50)),
      types: readStringArray(args.types)
    });
    const data = await filterAndCompressMemories(env, { query, memories });
    if (!readBoolean(args.full)) {
      return textToolResult({
        data: data.map((m) => ({
          id: m.id,
          type: m.type,
          summary: m.summary || m.content.slice(0, 80),
          date: eventDateOf(m),
          tags: m.tags,
          feel_intensity: m.feel_intensity,
          feel_note: m.feel_note,
          created_at: m.created_at,
          score: m.score
        }))
      });
    }
    return textToolResult({ data });
  }

  if (params.name === "memory_create") {
    if (!hasScope(profile, "memory:write")) return toolError("Missing memory:write scope");
    const content = readString(args.content);
    if (!content) return toolError("content is required");

    let summary = readString(args.summary) || null;
    let tags = readStringArray(args.tags);

    if (readBoolean(args.auto_summary)) {
      const generated = await generateChunkSummary(env, content);
      if (generated) {
        summary = generated.summary;
        tags = [...new Set([...tags, ...generated.tags])];
      }
    }

    let memory;
    try {
      memory = await createVectorMemory(env, {
        namespace: resolveNamespace(profile, args.namespace),
        type: readString(args.type) || "note",
        content,
        summary,
        importance: readNumber(args.importance, 0.5),
        confidence: readNumber(args.confidence, 0.8),
        pinned: readBoolean(args.pinned),
        tags,
        source: readString(args.source) || "mcp",
        sourceMessageIds: []
      });
    } catch (error) {
      return toolError(error instanceof Error ? error.message : "memory_create failed");
    }
    return textToolResult({ data: memory });
  }

  if (params.name === "memory_list") {
    if (!hasScope(profile, "memory:read")) return toolError("Missing memory:read scope");
    const typeFilter = readString(args.type);
    const dateFilter = readString(args.date);
    const tagFilter = readString(args.tag);
    const wantFull = readBoolean(args.full);
    const hasFilter = Boolean(typeFilter || dateFilter || tagFilter);
    // 过滤时默认放宽到 50(够看完一整天);否则保持 5。上限 100。
    const limit = readPositiveInt(args.limit, hasFilter ? 50 : 5, 100);
    try {
      // 任一过滤(type/date/tag)都先拉大池子再在内存里筛;游标分页只在不过滤时走。
      const fetchCount = hasFilter ? 1000 : limit;
      const page = await listVectorMemories(env, {
        namespace: resolveNamespace(profile, args.namespace),
        count: fetchCount,
        cursor: hasFilter ? undefined : readString(args.cursor)
      });
      const filtered = hasFilter
        ? page.data.filter((m) =>
            (!typeFilter || m.type === typeFilter) &&
            (!dateFilter || m.tags.includes(dateFilter)) &&
            (!tagFilter || m.tags.includes(tagFilter))
          )
        : page.data;
      const records = filtered
        // timeline 类(daily_summary/diary)按标签里的聊天日期排;否则回退创建时间。
        // 历史导入时所有记忆 created_at 都挤在导入当天,必须按 dateLabel 才能正确通读。
        .sort((a, b) => {
          const da = a.tags.find((t) => /^\d{4}-\d{2}-\d{2}$/.test(t)) ?? a.created_at;
          const db = b.tags.find((t) => /^\d{4}-\d{2}-\d{2}$/.test(t)) ?? b.created_at;
          return db.localeCompare(da);
        })
        .slice(0, limit)
        .map((m) => ({
          id: m.id,
          type: m.type,
          // full=true 出全文(改 summary 前要通读那天的 excerpt 用);否则给压缩摘要。
          ...(wantFull ? { content: m.content } : { summary: m.summary || m.content.slice(0, 80) }),
          tags: m.tags,
          feel_intensity: m.feel_intensity,
          feel_note: m.feel_note,
          created_at: m.created_at
        }));
      return textToolResult({
        data: records,
        ...(readBoolean(args.include_ids) ? { ids: records.map((r) => r.id) } : {}),
        paging: {
          limit,
          cursor: hasFilter ? null : page.cursor,
          has_more: hasFilter ? false : page.hasMore,
          count: records.length,
          total_count: hasFilter ? filtered.length : page.totalCount
        }
      });
    } catch (error) {
      return toolError(error instanceof Error ? error.message : "memory_list failed");
    }
  }

  if (params.name === "memory_get") {
    if (!hasScope(profile, "memory:read")) return toolError("Missing memory:read scope");
    const id = readString(args.id);
    if (!id) return toolError("id is required");
    const memory = await getVectorMemory(env, id);
    if (!memory) return toolError("Memory not found");
    return textToolResult({ data: memory });
  }

  if (params.name === "memory_delete") {
    if (!hasScope(profile, "memory:write")) return toolError("Missing memory:write scope");
    const id = readString(args.id);
    if (!id) return toolError("id is required");
    await deleteVectorMemory(env, id);
    return textToolResult({
      data: {
        id,
        deleted: true
      }
    });
  }

  if (params.name === "memory_update") {
    if (!hasScope(profile, "memory:write")) return toolError("Missing memory:write scope");
    const id = readString(args.id);
    if (!id) return toolError("id is required");
    const namespace = resolveNamespace(profile, args.namespace);
    const existing = await getVectorMemory(env, id);
    if (!existing || existing.namespace !== namespace) return toolError("Memory not found");
    const updated = await updateVectorMemory(env, id, {
      content: readString(args.content) ?? undefined,
      type: readString(args.type) ?? undefined,
      summary: args.summary !== undefined ? (readString(args.summary) ?? null) : undefined,
      tags: Array.isArray(args.tags) ? readStringArray(args.tags) : undefined,
      pinned: typeof args.pinned === "boolean" ? readBoolean(args.pinned) : undefined
    });
    if (!updated) return toolError("memory_update failed");
    return textToolResult({ data: updated });
  }

  if (params.name === "memory_ingest") {
    if (!hasScope(profile, "memory:write")) return toolError("Missing memory:write scope");
    const messages = readIngestMessages(args.messages);
    if (messages.length === 0) return toolError("messages must contain at least one message");
    const namespace = resolveNamespace(profile, args.namespace);
    const conversation = await getOrCreateConversation(env.DB, {
      namespace,
      id: readString(args.conversation_id)
    });
    const source = readString(args.source) || "mcp";
    const ids = await saveIngestMessages(env.DB, {
      conversationId: conversation.id,
      namespace,
      source,
      messages
    });

    if (args.auto_extract !== false && ids.length > 0) {
      ctx.waitUntil(
        enqueueMemoryMaintenanceIfNeeded(env, {
          namespace,
          conversationId: conversation.id,
          fromMessageId: ids[0],
          toMessageId: ids[ids.length - 1],
          source
        })
      );
    }

    return textToolResult({
      data: {
        conversation_id: conversation.id,
        message_ids: ids,
        auto_extract: args.auto_extract !== false
      }
    });
  }

  return toolError(`Unknown tool: ${String(params.name || "")}`);
}

async function handleRpc(
  request: JsonRpcRequest,
  env: Env,
  ctx: ExecutionContext,
  profile: KeyProfile
): Promise<Record<string, unknown> | null> {
  if (!request.id && request.method?.startsWith("notifications/")) return null;

  if (request.method === "initialize") {
    return rpcResult(request.id, {
      protocolVersion: "2025-06-18",
      capabilities: { tools: { listChanged: false } },
      serverInfo: { name: "companion-memory-mcp", version: "0.1.0" }
    });
  }

  if (request.method === "tools/list") {
    return rpcResult(request.id, { tools: getTools() });
  }

  if (request.method === "resources/list") {
    return rpcResult(request.id, { resources: [] });
  }

  if (request.method === "prompts/list") {
    return rpcResult(request.id, { prompts: [] });
  }

  if (request.method === "tools/call") {
    const params = isRecord(request.params) ? (request.params as ToolCallParams) : {};
    const result = await callTool(env, ctx, profile, params);
    return rpcResult(request.id, result);
  }

  if (request.method === "ping") {
    return rpcResult(request.id, {});
  }

  return rpcError(request.id, -32601, "Method not found");
}

export async function handleMcp(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204 });
  }

  if (request.method === "GET") {
    return json({
      name: "companion-memory-mcp",
      transport: "streamable-http",
      endpoint: new URL(request.url).pathname,
      tools: getTools().map((tool) => tool.name)
    });
  }

  if (request.method !== "POST") {
    return json({ error: "Method not allowed" }, { status: 405 });
  }

  const auth = await authenticate(withTokenQuery(request), env);
  if (!auth.ok) return rpcErrorResponse(null, -32001, "Unauthorized", 401);

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return rpcErrorResponse(null, -32700, "Parse error", 400);
  }

  if (Array.isArray(body)) {
    const results = (
      await Promise.all(
        body
          .filter((item): item is JsonRpcRequest => isRecord(item))
          .map((item) => handleRpc(item, env, ctx, auth.profile))
      )
    ).filter((item): item is Record<string, unknown> => item !== null);
    return results.length > 0 ? json(results) : new Response(null, { status: 202 });
  }

  if (!isRecord(body)) return rpcErrorResponse(null, -32600, "Invalid Request", 400);

  const result = await handleRpc(body, env, ctx, auth.profile);
  return result ? json(result) : new Response(null, { status: 202 });
}

function rpcErrorResponse(id: JsonRpcId | undefined, code: number, message: string, status: number): Response {
  return json(rpcError(id, code, message), { status });
}
