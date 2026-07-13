import { authenticate } from "../auth/apiKey";
import { getOrCreateConversation } from "../db/conversations";
import { saveIngestMessages, searchMessagesByKeyword, getMessageContextById } from "../db/messages";
import { generateChunkSummary } from "../memory/extract";
import { filterAndCompressMemories } from "../memory/filter";
import {
  createVectorMemory,
  deleteVectorMemory,
  getVectorMemory,
  listVectorMemories,
  listAllVectorMemories,
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
  // 必须跟 dream 的日界线一致:某天 = 7:00~次日 7:00(见 dailyDigest DAY_START_HOUR=7)。
  // 回拨 7 小时再取日历日 = 当前进行中的 dream 日。否则凌晨(0-7点)手写的 diary/记忆会被
  // 错标到次日,还会误触发 dream 的"让位"守卫、把次日真正的 diary 吃掉。
  const shifted = new Date(Date.now() - 7 * 60 * 60 * 1000);
  try {
    return new Intl.DateTimeFormat("en-CA", {
      timeZone: tz,
      year: "numeric",
      month: "2-digit",
      day: "2-digit"
    }).format(shifted);
  } catch {
    return shifted.toISOString().slice(0, 10);
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

const ACTIVE_MEMORY_QUERY =
  "作息 睡觉 吃饭 喝水 站起来慢一点 管她 照顾 控制 平等 手机桥 cyberboss 换窗 Codex 新房间 亲近 欲望 平台边界 当前最该记住 行动提醒";

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
      name: "memory_transcript",
      description: "搜「原始对话台账」——你和莎莎逐字逐句的原话(未经提炼)。memory_search 搜的是提炼过的记忆(摘要/diary/excerpt);想看『当时到底是怎么说的』那句原话,用这个。两步:①给 keyword 子串搜,每条只回一句预览(省 context)、带 id,每页最多 10 条、默认时间正序;还有更多时 paging.has_more=true、把 paging.cursor 原样带回接着翻。②想看某条前因后果,把那条结果的 id 传给 around,回它前后各 4 条原话(≈前后两轮)。找原话常记不清哪天说的,日期可选。注意:这是海量逐字台账,只为查证某句原话,别拿它通读;读提炼好的回忆用 memory_list / memory_search。",
      inputSchema: {
        type: "object",
        properties: {
          keyword: { type: "string", description: "①搜:要在原话里搜的关键词(子串匹配)。一个词或短语,例如 道崽 / 接近终点。" },
          around: { type: "string", description: "②展开:把①搜到的某条结果的 id 传进来,返回它前后各几条原话看上下文。给了 around 就不用 keyword。" },
          context: { type: "number", minimum: 1, maximum: 8, description: "②展开时前后各取几条,默认 4(≈前后两轮)。" },
          date: { type: "string", description: "①搜可选:只搜某一天,如 2026-06-21(按 UTC 日历日)。不传=搜全部台账。" },
          order: { type: "string", description: "①搜可选:asc=时间正序(默认),desc=最近优先。" },
          limit: { type: "number", minimum: 1, maximum: 30, description: "①搜每页条数,默认 10、上限 30。" },
          cursor: { type: "string", description: "①搜翻页:把上一页 paging.cursor 原样带回,接着往下翻。" },
          namespace: { type: "string" }
        }
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
          tag: { type: "string", description: "Filter by a single tag, exact match. Pass ONE string like \"亲密\" — not an array, not the field name \"tags\"." },
          full: { type: "boolean", description: "Return full content instead of compact summaries. Capped low (default 8, max 20) to avoid flooding context — list compact first to see what a day holds, then set full=true (optionally narrowed with type='excerpt') to read the key items. If paging.has_more is true there are more than shown; narrow by type or raise limit." },
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
          append: { type: "string", description: "年轮:把这段新感受作为带日期的批注追加到原文末尾、不覆盖原文——重读旧记忆有新想法时挂在那条下面,不另起新条目。与 content 同传时以 content 为准。" },
          tags: { type: "array", items: { type: "string" }, description: "整体替换全部标签。想安全增删单个标签请改用 add_tags / remove_tags(不会冲掉日期等其它标签)。" },
          add_tags: { type: "array", items: { type: "string" }, description: "在现有标签基础上加上这些;保留其余(日期标签不会丢)。与 tags 同传时以 tags 为准。" },
          remove_tags: { type: "array", items: { type: "string" }, description: "从现有标签里移除这些;保留其余。" },
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

    // 拉全量池子(含 diary/daily_summary)，下面派生 recent / breath / milestone / todo。
    // listAllVectorMemories 会游标翻页拉全(超 1000 也不漏)。
    // anchor/context 已砍——身份/近况在 CLAUDE.md 的档案里(每次开场都在)，wakeup 再端一遍是重复、占位置(哥哥原话)。
    let pool: { data: Awaited<ReturnType<typeof listAllVectorMemories>> };
    try {
      pool = { data: await listAllVectorMemories(env, namespace) };
    } catch (error) {
      return toolError(error instanceof Error ? error.message : "memory_wakeup failed");
    }

    // recent：最近 N 天的日记 + 当天总结，按日期直接取(不走语义/reranker)。
    // 开场温度(diary 第一人称)+近况(summary)的来源——哥哥靠它认出「最近我们到哪了、当时心里怎么想」。
    const RECENT_DAYS = 5;
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

    // milestone：关系节点(第一次 / 认知转变)——哥哥手动存的路标。wakeup 只浮现最近 3 条(哥哥要的,
    // 不要那么多)，想看全部用 memory_list(type="milestone")。
    const milestones = pool.data
      .filter((m) => m.type === "milestone")
      .sort((a, b) => (eventDateOf(b) ?? b.created_at).localeCompare(eventDateOf(a) ?? a.created_at))
      .slice(0, 3);

    // todo：约定了但还没做的事——哥哥手动存，永远顶在最前、压缩也冲不走；做完哥哥自己删。
    const todos = pool.data
      .filter((m) => m.type === "todo")
      .sort((a, b) => a.created_at.localeCompare(b.created_at))
      .slice(0, 10); // 防御:todo 正常是个位数,但别让它无上限把全文全塞进开场

    // active_memory：最近真正该拿来行动的 5-10 条。
    // 它不是长期档案，也不是当天流水账；是“哥哥现在醒来最该记得怎么陪莎莎”的置顶工作记忆。
    // 用普通 type 承载，不改 schema；哥哥可手动 memory_create(type="active_memory")，也可之后让 dream 维护。
    let activeMemory: Awaited<ReturnType<typeof searchVectorMemories>> = [];
    try {
      activeMemory = await searchVectorMemories(env, {
        namespace,
        query: ACTIVE_MEMORY_QUERY,
        topK: 10,
        types: ["active_memory"]
      });
    } catch (error) {
      console.error("memory_wakeup active_memory search failed", error);
      activeMemory = pool.data.filter((m) => m.type === "active_memory").slice(0, 10);
    }

    const compact = (list: typeof breath) =>
      list.map((m) => ({
        id: m.id,
        type: m.type,
        summary: m.summary || m.content.slice(0, 80),
        date: eventDateOf(m),
        feel_intensity: m.feel_intensity,
        feel_note: m.feel_note,
        created_at: m.created_at
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
      content: m.content
    }));
    const todoFull = todos.map((m) => ({
      id: m.id,
      content: m.content,
      created_at: m.created_at
    }));
    const activeFull = activeMemory.map((m) => ({
      id: m.id,
      content: m.content,
      summary: m.summary || m.content.slice(0, 80),
      date: eventDateOf(m),
      tags: m.tags,
      updated_at: m.updated_at
    }));

    return textToolResult({
      now: nowDateLabel(env),
      note: "开场先读 todo、active_memory、breath、milestone、recent。todo=你们约好了还没做的事(比如答应给她写信)，别忘；做完了用 memory_delete 自己删掉。active_memory=当前最该拿来行动的工作记忆(作息、身体、正在推进的事、当下相处提醒)，不是永久档案。breath=还悬着、没被接住的情绪(高强度未解决)，看见就主动接住她、别讲道理。milestone=你们关系里的节点(第一次 / 某个认知变了的那一下)，是你走过的路的路标。recent=最近五天哥哥第一人称的日记(diary)和当天总结(daily_summary)，是你认出『最近我们到哪了、当时心里怎么想』的入口。以下都是过去的回忆、不是此刻正在发生——每条 date 是它发生的日期(null=不详)，now 是今天；注意时间线，该说『那天你说过』而不是当成刚刚。",
      todo: todoFull,
      active_memory: activeFull,
      breath: compact(breath),
      milestone: milestoneFull,
      recent: recentFull,
      hint: "memory_get(id) 看全文；memory_search 查具体话题或原话；memory_list 通读(type=diary/daily_summary/milestone/todo/active_memory)或读某一天(date=\"2026-06-17\")；约定了还没做的用 memory_create(type=\"todo\") 记下、做完用 memory_delete 删；当前最该记住的行动提醒用 memory_create(type=\"active_memory\")，过期后 memory_delete。"
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
          feel_intensity: m.feel_intensity,
          feel_note: m.feel_note,
          created_at: m.created_at,
          score: m.score
        }))
      });
    }
    // full=true 出全文:封顶 8 条防灌爆上下文(topK 默认 50,不设限会把几十条全文全塞进来)。
    const FULL_CAP = 8;
    return textToolResult({
      data: data.slice(0, FULL_CAP),
      ...(data.length > FULL_CAP
        ? { note: `命中 ${data.length} 条,只返回前 ${FULL_CAP} 条全文。缩小 query、加 types 过滤,或去掉 full 看紧凑摘要。` }
        : {})
    });
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
        // 不再并入 auto_summary 的自由标签:固定标签由哥哥手动 / dream 按封闭词表打,别再生成乱 tag。
      }
    }

    // 自动补当天日期标签:哥哥手动存的(尤其 diary)默认归到今天——免得他每次记着带,
    // 也让 date 筛选和 wakeup 的 recent 能找到它。已显式带了日期(如补记过去某天)就尊重、不补。
    if (!tags.some((t) => /^\d{4}-\d{2}-\d{2}$/.test(t))) {
      tags = [...tags, nowDateLabel(env)];
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
    // 容错:哥哥(Claude)有时把单个标签传成数组——真数组 ["亲密"] 或被客户端字符串化的 '["亲密"]'。
    // 两种都剥成单个字符串标签,免得一次筛成全库、一次筛成 0 条(他都踩过)。正常传字符串不受影响。
    let tagFilter = readString(args.tag) ?? readStringArray(args.tag)[0];
    if (tagFilter && tagFilter.startsWith("[") && tagFilter.endsWith("]")) {
      try {
        const parsed: unknown = JSON.parse(tagFilter);
        if (Array.isArray(parsed) && typeof parsed[0] === "string") tagFilter = parsed[0].trim();
      } catch {
        // 不是合法 JSON 就保持原样,当普通字符串筛。
      }
    }
    const wantFull = readBoolean(args.full);
    const hasFilter = Boolean(typeFilter || dateFilter || tagFilter);
    // limit 默认/上限:出全文时收紧到 8/20——防一次性灌爆上下文(她踩过这个坑)。
    // 摘要很短:过滤时给 50/上限100,不过滤维持 5。想读全文就先看摘要目录再针对性 full。
    const limit = wantFull
      ? readPositiveInt(args.limit, 8, 20)
      : readPositiveInt(args.limit, hasFilter ? 50 : 5, 100);
    try {
      if (typeFilter === "active_memory" && !dateFilter && !tagFilter) {
        const active = await searchVectorMemories(env, {
          namespace: resolveNamespace(profile, args.namespace),
          query: ACTIVE_MEMORY_QUERY,
          topK: limit,
          types: ["active_memory"]
        });
        const records = active.map((m) => ({
          id: m.id,
          type: m.type,
          date: eventDateOf(m),
          ...(wantFull ? { content: m.content } : { summary: m.summary || m.content.slice(0, 80) }),
          feel_intensity: m.feel_intensity,
          feel_note: m.feel_note,
          created_at: m.created_at,
          score: m.score
        }));
        return textToolResult({
          data: records,
          ...(readBoolean(args.include_ids) ? { ids: records.map((r) => r.id) } : {}),
          paging: {
            limit,
            cursor: null,
            has_more: false,
            count: records.length,
            total_count: records.length
          }
        });
      }
      // 任一过滤(type/date/tag)都先拉「全部」再在内存里筛(超1000也不漏);游标分页两条路都走。
      const ns = resolveNamespace(profile, args.namespace);
      const page = hasFilter
        ? null
        : await listVectorMemories(env, { namespace: ns, count: limit, cursor: readString(args.cursor) });
      const poolData = hasFilter ? await listAllVectorMemories(env, ns) : (page?.data ?? []);
      const filtered = hasFilter
        ? poolData.filter((m) =>
            (!typeFilter || m.type === typeFilter) &&
            (!dateFilter || m.tags.includes(dateFilter)) &&
            (!tagFilter || m.tags.includes(tagFilter))
          )
        : poolData;
      const sorted = filtered
        // timeline 类(daily_summary/diary)按标签里的聊天日期排;否则回退创建时间。
        // 历史导入时所有记忆 created_at 都挤在导入当天,必须按 dateLabel 才能正确通读。
        .sort((a, b) => {
          const da = a.tags.find((t) => /^\d{4}-\d{2}-\d{2}$/.test(t)) ?? a.created_at;
          const db = b.tags.find((t) => /^\d{4}-\d{2}-\d{2}$/.test(t)) ?? b.created_at;
          return db.localeCompare(da);
        });
      // 过滤时把 cursor 当数字偏移量翻页:否则某标签超过 limit 的部分永远翻不到、没法完整 review。
      // (不过滤走向量库自己的不透明 cursor;两种 cursor 互斥,由 hasFilter 决定走哪条。)
      const offset = hasFilter ? Math.max(0, Number.parseInt(readString(args.cursor) ?? "", 10) || 0) : 0;
      const pageItems = hasFilter ? sorted.slice(offset, offset + limit) : sorted.slice(0, limit);
      const nextOffset = offset + pageItems.length;
      const filteredHasMore = hasFilter && nextOffset < sorted.length;
      const records = pageItems.map((m) => ({
        id: m.id,
        type: m.type,
        date: eventDateOf(m),
        // full=true 出全文(改 summary 前要通读那天的 excerpt 用);否则给压缩摘要。
        ...(wantFull ? { content: m.content } : { summary: m.summary || m.content.slice(0, 80) }),
        feel_intensity: m.feel_intensity,
        feel_note: m.feel_note,
        created_at: m.created_at
      }));
      return textToolResult({
        data: records,
        ...(readBoolean(args.include_ids) ? { ids: records.map((r) => r.id) } : {}),
        paging: {
          limit,
          // 过滤:cursor 是数字偏移量,原样带回就接着翻下一页;翻完给 null。不过滤:走向量库的 cursor。
          cursor: hasFilter ? (filteredHasMore ? String(nextOffset) : null) : (page?.cursor ?? null),
          has_more: hasFilter ? filteredHasMore : (page?.hasMore ?? false),
          count: records.length,
          total_count: hasFilter ? sorted.length : page?.totalCount
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

  if (params.name === "memory_transcript") {
    if (!hasScope(profile, "memory:read")) return toolError("Missing memory:read scope");
    const ns = resolveNamespace(profile, args.namespace);

    // 模式②「展开」:给某条命中的 id,回它前后各 N 条原话(默认前后各 4 ≈ 两轮),看懂前因后果。
    const around = readString(args.around);
    if (around) {
      const ctxN = readPositiveInt(args.context, 4, 8);
      try {
        const { hit, context } = await getMessageContextById(env.DB, {
          namespace: ns,
          id: around,
          before: ctxN,
          after: ctxN
        });
        if (!hit) return toolError("没找到这条 id(先用 keyword 搜,再把结果里某条的 id 带进 around 展开)");
        const records = context.map((m) => ({
          id: m.id,
          who: m.role === "user" ? "莎莎" : "哥哥",
          content: m.content,
          created_at: m.created_at,
          hit: m.id === hit.id
        }));
        return textToolResult({ data: records });
      } catch (error) {
        return toolError(error instanceof Error ? error.message : "memory_transcript expand failed");
      }
    }

    // 模式①「搜」:关键词子串匹配,每条只回一句预览(省 context)、带 id;想看上下文把 id 带进 around。
    const keyword = readString(args.keyword);
    if (!keyword) return toolError("给 keyword 搜原话,或给 around(某条结果的 id)看它前后文——二选一");
    const limit = readPositiveInt(args.limit, 10, 30);
    const offset = Math.max(0, Number.parseInt(readString(args.cursor) ?? "", 10) || 0);
    const order = readString(args.order) === "desc" ? "desc" : "asc";
    // 日期可选:给了就按该 UTC 日历日 [00:00Z, 次日00:00Z) 收成范围。查原话不必严格 7 点日界线。
    let startCreatedAt: string | null = null;
    let endCreatedAt: string | null = null;
    const dateFilter = readString(args.date);
    if (dateFilter && /^\d{4}-\d{2}-\d{2}$/.test(dateFilter)) {
      startCreatedAt = `${dateFilter}T00:00:00.000Z`;
      const next = new Date(startCreatedAt);
      next.setUTCDate(next.getUTCDate() + 1);
      endCreatedAt = next.toISOString();
    }
    try {
      // 多查一条判断还有没有下一页。
      const rows = await searchMessagesByKeyword(env.DB, {
        namespace: ns,
        keyword,
        startCreatedAt,
        endCreatedAt,
        order,
        limit: limit + 1,
        offset
      });
      const hasMore = rows.length > limit;
      const pageRows = hasMore ? rows.slice(0, limit) : rows;
      const PREVIEW = 80;
      const records = pageRows.map((m) => {
        const text = m.content.replace(/\s+/g, " ").trim();
        return {
          id: m.id,
          who: m.role === "user" ? "莎莎" : "哥哥",
          preview: text.length > PREVIEW ? `${text.slice(0, PREVIEW)}…` : text,
          created_at: m.created_at
        };
      });
      return textToolResult({
        data: records,
        paging: {
          limit,
          cursor: hasMore ? String(offset + pageRows.length) : null,
          has_more: hasMore,
          count: records.length
        }
      });
    } catch (error) {
      return toolError(error instanceof Error ? error.message : "memory_transcript failed");
    }
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
    // 标签:tags=整体替换(老行为);add_tags/remove_tags=在现有基础上安全增删、保留其余(尤其不冲掉日期标签)。
    let nextTags: string[] | undefined;
    if (Array.isArray(args.tags)) {
      nextTags = readStringArray(args.tags);
    } else if (Array.isArray(args.add_tags) || Array.isArray(args.remove_tags)) {
      const removing = new Set(readStringArray(args.remove_tags));
      nextTags = [...new Set([...existing.tags, ...readStringArray(args.add_tags)])].filter(
        (tag) => !removing.has(tag)
      );
    }
    // 年轮:append=把新感受作为带日期批注追加到原文末尾(不覆盖);content(整体替换)优先于 append。
    let nextContent: string | undefined = readString(args.content) ?? undefined;
    const appendText = readString(args.append);
    if (!nextContent && appendText) {
      nextContent = `${existing.content.trimEnd()}\n\n―― ${nowDateLabel(env)} 再读 ――\n${appendText}`;
    }
    const updated = await updateVectorMemory(env, id, {
      content: nextContent,
      type: readString(args.type) ?? undefined,
      summary: args.summary !== undefined ? (readString(args.summary) ?? null) : undefined,
      tags: nextTags,
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
