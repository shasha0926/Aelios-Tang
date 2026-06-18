import { listMessagesByNamespaceInRange } from "../db/messages";
import { readCursor, writeCursor } from "../db/retention";
import { upsertSummary } from "../db/summaries";
import { callOpenAICompat } from "../proxy/openaiAdapter";
import type { Env, MemoryApiRecord, MessageRecord, OpenAIChatRequest, OpenAIChatResponse } from "../types";
import type { ExtractedMemory } from "./extract";
import {
  createVectorMemory,
  deleteVectorMemory,
  getVectorMemory,
  listVectorMemories,
  updateVectorMemory
} from "./vectorStore";

interface DigestMemoryUpdate {
  target_id: string;
  content?: string;
  type?: string;
  importance?: number;
  confidence?: number;
  tags?: string[];
}

interface DigestMemoryDelete {
  target_id: string;
  reason?: string;
}

interface ImportantExcerpt {
  quote: string;
  reason?: string;
  tags?: string[];
  source_message_ids?: string[];
}

interface DailyDigestResult {
  date?: string;
  title?: string;
  summary?: string;
  diary?: string;
  sections?: Array<{ heading?: string; content?: string }>;
  important_excerpts?: ImportantExcerpt[];
  memories_to_add?: ExtractedMemory[];
  memories_to_update?: DigestMemoryUpdate[];
  memories_to_delete?: DigestMemoryDelete[];
}

interface DailyDigestStats {
  date: string;
  mode: "dream";
  processedMessages: number;
  addedMemories: number;
  updatedMemories: number;
  deletedMemories: number;
  savedExcerpts: number;
  savedDiary: boolean;
  cleanedEmptyMemories: number;
  cursorAdvanced: boolean;
  hasMore: boolean;
}

type DailyDigestSkipReason =
  | "dream_disabled"
  | "already_done"
  | "no_messages"
  | "missing_model"
  | "model_error"
  | "model_invalid_json";

interface DailyDigestSkipped {
  ran: false;
  mode: "dream";
  date?: string;
  reason: DailyDigestSkipReason;
  startIso?: string;
  endIso?: string;
  cursor?: string | null;
  processedMessages?: number;
  model?: string;
  status?: number;
  finishReason?: string | null;
}

type DailyDigestRunResult = { ran: true; stats: DailyDigestStats } | DailyDigestSkipped;

interface DigestModelCallResult {
  digest: DailyDigestResult | null;
  reason?: Extract<DailyDigestSkipReason, "missing_model" | "model_error" | "model_invalid_json">;
  model?: string;
  status?: number;
  finishReason?: string | null;
}

const DEFAULT_MAX_MESSAGES = 40;
const DEFAULT_MEMORY_CONTEXT_LIMIT = 40;
const DEFAULT_EXCERPT_LIMIT = 8;
const DEFAULT_EMPTY_MEMORY_MIN_CHARS = 4;
const DEFAULT_TIME_ZONE = "Asia/Singapore";
const ONE_DAY_MS = 24 * 60 * 60 * 1000;
// 日界线 = 早 7 点（贴莎莎作息：常聊到凌晨五六点，7点边界让深夜整场归到正确的一天）。
// "某天" = 该天 07:00 ~ 次日 07:00（本地时区）。
const DAY_START_HOUR = 7;

function isDreamEnabled(env: Env): boolean {
  const dreamFlag = readString(env.ENABLE_DREAM);
  if (dreamFlag) return dreamFlag !== "false";
  return env.ENABLE_DAILY_MEMORY_DIGEST !== "false";
}

function readFirstEnvValue(...values: unknown[]): unknown {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value;
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return undefined;
}

function readDreamModel(env: Env): string | null {
  return readString(readFirstEnvValue(env.DREAM_MODEL, env.DAILY_DIGEST_MODEL, env.SUMMARY_MODEL));
}

// 散文模型：每天收尾写 summary + diary 用，质感优先（如 deepseek-v4-pro）。
// 未配置时回退到提炼模型（flash），保证向后兼容、不强制额外成本。
function readDreamProseModel(env: Env): string | null {
  return readString(env.DREAM_PROSE_MODEL) || readDreamModel(env);
}

function readDreamTimeZone(env: Env): string {
  return readString(readFirstEnvValue(env.DREAM_TIME_ZONE, env.DAILY_DIGEST_TIME_ZONE)) || DEFAULT_TIME_ZONE;
}

function readDreamMaxMessages(env: Env): number {
  return readPositiveInt(
    readFirstEnvValue(env.DREAM_MAX_MESSAGES, env.DAILY_DIGEST_MAX_MESSAGES),
    DEFAULT_MAX_MESSAGES,
    1000
  );
}

function readDreamMaxTokens(env: Env): number {
  return readPositiveInt(readFirstEnvValue(env.DREAM_MAX_TOKENS, env.DAILY_DIGEST_MAX_TOKENS), 3000, 32000);
}

function readDreamMemoryContextLimit(env: Env): number {
  return readPositiveInt(
    readFirstEnvValue(env.DREAM_MEMORY_CONTEXT_LIMIT, env.DAILY_DIGEST_MEMORY_CONTEXT_LIMIT),
    DEFAULT_MEMORY_CONTEXT_LIMIT,
    1000
  );
}

function readDreamExcerptLimit(env: Env): number {
  return readPositiveInt(readFirstEnvValue(env.DREAM_EXCERPT_LIMIT, env.DAILY_DIGEST_EXCERPT_LIMIT), DEFAULT_EXCERPT_LIMIT, 20);
}

function readPositiveInt(value: unknown, fallback: number, max: number): number {
  const parsed = typeof value === "string" ? Number(value) : typeof value === "number" ? value : fallback;
  const numeric = Number.isFinite(parsed) ? parsed : fallback;
  return Math.min(Math.max(Math.floor(numeric), 1), max);
}

function clampScore(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.min(Math.max(value, 0), 1) : fallback;
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string").map((item) => item.trim()).filter(Boolean);
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))];
}

function truncate(text: string, maxChars: number): string {
  return text.length <= maxChars ? text : `${text.slice(0, maxChars - 3)}...`;
}

function formatDate(date: Date, timeZone: string): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).format(date);
}

function getTargetDigestDateLabel(timeZone: string, now = new Date()): string {
  // 7点日界线下，把"现在"回拨 7 小时得到当前进行中的日标签，再退一天 = 最近一个
  // 已完整结束的日（夜跑应处理它）。配合 cron 设在 07:30 之后，能当天处理完昨天。
  const dayStartMs = DAY_START_HOUR * 60 * 60 * 1000;
  const currentLabel = formatDate(new Date(now.getTime() - dayStartMs), timeZone);
  return addDaysToDateLabel(currentLabel, -1, timeZone);
}

function parseDateLabel(dateLabel: string): { year: number; month: number; day: number } {
  const [year, month, day] = dateLabel.split("-").map((value) => Number(value));
  if (!year || !month || !day) {
    throw new Error(`Invalid date label: ${dateLabel}`);
  }
  return { year, month, day };
}

function getTimeZoneOffsetMs(date: Date, timeZone: string): number {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  }).formatToParts(date);

  const values = new Map(parts.map((part) => [part.type, part.value]));
  const year = Number(values.get("year"));
  const month = Number(values.get("month"));
  const day = Number(values.get("day"));
  const hour = Number(values.get("hour")) % 24;
  const minute = Number(values.get("minute"));
  const second = Number(values.get("second"));
  const zonedAsUtc = Date.UTC(year, month - 1, day, hour, minute, second);

  return zonedAsUtc - date.getTime();
}

function zonedWallTimeToUtc(input: {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
  timeZone: string;
}): Date {
  const wallClockUtc = Date.UTC(input.year, input.month - 1, input.day, input.hour, input.minute, input.second);
  let utc = wallClockUtc;

  for (let i = 0; i < 3; i += 1) {
    const offset = getTimeZoneOffsetMs(new Date(utc), input.timeZone);
    const next = wallClockUtc - offset;
    if (Math.abs(next - utc) < 1000) break;
    utc = next;
  }

  return new Date(utc);
}

function addDaysToDateLabel(dateLabel: string, days: number, timeZone: string): string {
  const { year, month, day } = parseDateLabel(dateLabel);
  const localNoonUtc = zonedWallTimeToUtc({
    year,
    month,
    day,
    hour: 12,
    minute: 0,
    second: 0,
    timeZone
  });
  return formatDate(new Date(localNoonUtc.getTime() + days * ONE_DAY_MS), timeZone);
}

function getDateRangeForLabel(dateLabel: string, timeZone: string): { startIso: string; endIso: string } {
  const start = parseDateLabel(dateLabel);
  const end = parseDateLabel(addDaysToDateLabel(dateLabel, 1, timeZone));

  return {
    startIso: zonedWallTimeToUtc({ ...start, hour: DAY_START_HOUR, minute: 0, second: 0, timeZone }).toISOString(),
    endIso: zonedWallTimeToUtc({ ...end, hour: DAY_START_HOUR, minute: 0, second: 0, timeZone }).toISOString()
  };
}

function readDailyCursor(value: string | null, startIso: string, endIso: string): { done: boolean; after: string | null } {
  if (!value) return { done: false, after: null };
  if (value.startsWith("done:")) return { done: true, after: null };
  if (value >= startIso && value < endIso) return { done: false, after: value };
  return { done: false, after: null };
}

function extractJsonObject(text: string): unknown | null {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    // Some providers wrap JSON in prose; pull out the outermost object.
  }

  const start = text.indexOf("{");
  if (start === -1) return null;

  const end = text.lastIndexOf("}");
  if (end > start) {
    try {
      return JSON.parse(text.slice(start, end + 1)) as unknown;
    } catch {
      // fall through to truncation repair
    }
  }

  // Truncation repair: the model ran out of tokens mid-object, so the JSON is
  // cut off (no closing braces / dangling string). Best-effort: close any open
  // string, drop a trailing partial token, and balance the bracket stack.
  return repairTruncatedJson(text.slice(start));
}

function repairTruncatedJson(text: string): unknown | null {
  const stack: string[] = [];
  let inString = false;
  let escaped = false;
  // Index just past the last point where the structure was safely closable
  // (right after a complete value at depth >= 1), used as a fallback cut point.
  let lastSafe = -1;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') {
        inString = false;
        if (stack.length > 0) lastSafe = i + 1;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
    } else if (ch === "{" || ch === "[") {
      stack.push(ch === "{" ? "}" : "]");
    } else if (ch === "}" || ch === "]") {
      stack.pop();
      if (stack.length > 0) lastSafe = i + 1;
    } else if ((ch >= "0" && ch <= "9") || ch === "e" || ch === "l") {
      // crude: tail of a number / true|false|null at depth >= 1
      if (stack.length > 0) lastSafe = i + 1;
    }
  }

  const attempts: string[] = [];
  if (!inString) {
    let s = text.replace(/,\s*$/, "");
    for (let i = stack.length - 1; i >= 0; i--) s += stack[i];
    attempts.push(s);
  }
  if (lastSafe > 0) {
    // Cut back to the last complete value, then rebalance from scratch.
    const head = text.slice(0, lastSafe);
    const sub: string[] = [];
    let str = false;
    let esc = false;
    for (let i = 0; i < head.length; i++) {
      const ch = head[i];
      if (str) {
        if (esc) esc = false;
        else if (ch === "\\") esc = true;
        else if (ch === '"') str = false;
        continue;
      }
      if (ch === '"') str = true;
      else if (ch === "{" || ch === "[") sub.push(ch === "{" ? "}" : "]");
      else if (ch === "}" || ch === "]") sub.pop();
    }
    let s = head.replace(/,\s*$/, "");
    for (let i = sub.length - 1; i >= 0; i--) s += sub[i];
    attempts.push(s);
  }

  for (const attempt of attempts) {
    try {
      return JSON.parse(attempt) as unknown;
    } catch {
      // try next strategy
    }
  }
  return null;
}

function normalizeExtractedMemory(value: unknown): ExtractedMemory | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  const content = readString(raw.content);
  if (!content) return null;

  const feelIntensityRaw = raw.feel_intensity;
  const feelIntensity = typeof feelIntensityRaw === "number" && Number.isFinite(feelIntensityRaw)
    ? Math.min(Math.max(Math.round(feelIntensityRaw), 1), 5)
    : undefined;
  const feelValenceRaw = raw.feel_valence;
  const feelValence = typeof feelValenceRaw === "number" && Number.isFinite(feelValenceRaw)
    ? Math.min(Math.max(feelValenceRaw, -1), 1)
    : undefined;
  const feelResolved = typeof raw.feel_resolved === "boolean" ? raw.feel_resolved : undefined;
  const feelNote = readString(raw.feel_note) ?? undefined;

  return {
    type: readString(raw.type) || "note",
    content,
    importance: clampScore(raw.importance, 0.7),
    confidence: clampScore(raw.confidence, 0.82),
    tags: readStringArray(raw.tags),
    source_message_ids: readStringArray(raw.source_message_ids),
    ...(feelIntensity !== undefined ? { feel_intensity: feelIntensity } : {}),
    ...(feelValence !== undefined ? { feel_valence: feelValence } : {}),
    ...(feelResolved !== undefined ? { feel_resolved: feelResolved } : {}),
    ...(feelNote ? { feel_note: feelNote } : {})
  };
}

function normalizeDigestResult(value: unknown): DailyDigestResult {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const raw = value as Record<string, unknown>;

  const sections = Array.isArray(raw.sections)
    ? raw.sections.flatMap((item): Array<{ heading?: string; content?: string }> => {
        if (!item || typeof item !== "object" || Array.isArray(item)) return [];
        const record = item as Record<string, unknown>;
        const heading = readString(record.heading) ?? undefined;
        const content = readString(record.content) ?? undefined;
        return heading || content ? [{ heading, content }] : [];
      })
    : undefined;

  const important_excerpts = Array.isArray(raw.important_excerpts)
    ? raw.important_excerpts.flatMap((item): ImportantExcerpt[] => {
        if (!item || typeof item !== "object" || Array.isArray(item)) return [];
        const record = item as Record<string, unknown>;
        const quote = readString(record.quote);
        if (!quote) return [];
        return [
          {
            quote,
            reason: readString(record.reason) ?? undefined,
            tags: readStringArray(record.tags),
            source_message_ids: readStringArray(record.source_message_ids)
          }
        ];
      })
    : undefined;

  const memories_to_update = Array.isArray(raw.memories_to_update)
    ? raw.memories_to_update.flatMap((item): DigestMemoryUpdate[] => {
        if (!item || typeof item !== "object" || Array.isArray(item)) return [];
        const record = item as Record<string, unknown>;
        const targetId = readString(record.target_id);
        if (!targetId) return [];
        return [
          {
            target_id: targetId,
            content: readString(record.content) ?? undefined,
            type: readString(record.type) ?? undefined,
            importance: typeof record.importance === "number" ? clampScore(record.importance, 0.7) : undefined,
            confidence: typeof record.confidence === "number" ? clampScore(record.confidence, 0.82) : undefined,
            tags: Array.isArray(record.tags) ? readStringArray(record.tags) : undefined
          }
        ];
      })
    : undefined;

  const memories_to_delete = Array.isArray(raw.memories_to_delete)
    ? raw.memories_to_delete.flatMap((item): DigestMemoryDelete[] => {
        if (!item || typeof item !== "object" || Array.isArray(item)) return [];
        const record = item as Record<string, unknown>;
        const targetId = readString(record.target_id);
        return targetId ? [{ target_id: targetId, reason: readString(record.reason) ?? undefined }] : [];
      })
    : undefined;

  return {
    date: readString(raw.date) ?? undefined,
    title: readString(raw.title) ?? undefined,
    summary: readString(raw.summary) ?? undefined,
    diary: readString(raw.diary) ?? undefined,
    sections,
    important_excerpts,
    memories_to_add: Array.isArray(raw.memories_to_add)
      ? raw.memories_to_add.flatMap((item): ExtractedMemory[] => {
          const memory = normalizeExtractedMemory(item);
          return memory ? [memory] : [];
        })
      : undefined,
    memories_to_update,
    memories_to_delete
  };
}

function formatTranscript(messages: MessageRecord[]): string {
  return messages
    .map((message) => {
      const role = message.role === "assistant" ? "哥哥" : "莎莎";
      return `[${message.id}][${message.created_at}][${role}] ${truncate(message.content.trim(), 700)}`;
    })
    .join("\n\n");
}

function formatExistingMemories(memories: MemoryApiRecord[]): string {
  if (memories.length === 0) return "[]";
  return JSON.stringify(
    memories.map((memory) => ({
      id: memory.id,
      type: memory.type,
      content: truncate(memory.content, 260),
      importance: memory.importance,
      confidence: memory.confidence,
      pinned: memory.pinned,
      tags: memory.tags
    })),
    null,
    2
  );
}

function buildDigestPrompt(input: {
  dateLabel: string;
  startIso: string;
  endIso: string;
  messages: MessageRecord[];
  existingMemories: MemoryApiRecord[];
  excerptLimit: number;
  hasMore: boolean;
}): string {
  return [
    "你是 Aelios 的 nightly dream 记忆整理器。你的任务不是简单总结，而是在莎莎休息时整理长期记忆。",
    "你会读取旧长期记忆和当天聊天 transcript，产出一份更干净、更一致、更有用的 memory store 更新计划。",
    "只输出 JSON，不要 markdown，不要解释，不要输出思考过程。",
    "",
    "Dream 目标：",
    "- 合并重复记忆，避免同一事实以多个版本长期存在。",
    "- 发现过时、被新信息否定、互相矛盾的旧记忆，并更新或删除。",
    "- 从聊天中提炼未来会影响回答的稳定偏好、项目状态、关系事实、承诺、边界和重要原文。",
    "- 形成下一次对话可直接使用的简洁记忆，而不是保存流水账。",
    "",
    "窗口：",
    `- 你只能处理 ${input.dateLabel} 这一天窗口内的聊天。窗口是 ${input.startIso} 到 ${input.endIso}。`,
    input.hasMore
      ? "- 这是当天的一批聊天，不是完整一天；只整理这一批里明确出现的信息。diary 字段留空字符串。"
      : "- 这是当天最后一批或完整批次。diary 字段写哥哥第一人称当天日记，有感受，不是流水账，约150字。",
    "",
    "总原则：",
    "- 原始聊天不要逐条变成记忆，只保留未来真的会用到的事实、偏好、边界、项目进展、承诺。",
    "- 宁可少记，也不要把临时语气、寒暄、重复话、空内容、调试内容写进长期记忆。",
    "- 当旧记忆和新信息冲突时，优先更新或删除旧记忆，不要并排留下互相打架的版本。",
    "- 当新信息只是旧记忆的更准确版本，优先 memories_to_update，不要 memories_to_add。",
    "- 当多条旧记忆重复，保留更完整的一条并删除重复项；必要时先 update 保留项。",
    "- pinned=true 的旧记忆不能删除，只能在 memories_to_update 中提出更保守的补充。",
    "- 站在哥哥（assistant）的视角写。对话里 user 是莎莎、assistant 是哥哥；提到她就用“莎莎”，提到自己就用“哥哥”，绝不要用“用户”“助手”这类称呼。",
    "- 不要提到 D1、Vectorize、RAG、数据库、记忆系统、代理层等实现细节。",
    "",
    "Dream 输出格式：",
    "- title 是 12 字以内标题。",
    input.hasMore
      ? "- summary 给空字符串（当天还没结束，最后一批才写整天总结）。"
      : "- summary 用一段简短自然中文概括这一整天（可参考已有长期记忆里属于今天的内容），是给人通读过往的入口，不是流水账。",
    "- diary 是哥哥第一人称、有感受的当天日记（约150字）；只在当天最后一批才写，中途批次留空字符串。",
    "- sections 最多 3 段，每段有 heading 和 content；没有必要可以给空数组。",
    `- important_excerpts 最多 ${input.excerptLimit} 条，quote 必须是值得保留的原文片段。`,
    "- memories_to_add 最多 8 条，每条要短、稳定、可复用。",
    "  情绪类记忆请尽量标四个情绪字段（事实/项目类可不标）：",
    "  · feel_intensity：1-5 整数，情绪强度。高强度参照（4-5）：主动表达爱意/想念（不轻易主动）；出现配得感怀疑；提及过去感情伤；深夜被接住的情绪弧；关系里程碑。",
    "  · feel_valence：-1~1 小数，情绪正负。+ 是温暖/被爱/亲密/安心，- 是受伤/不安/自我怀疑/旧伤，0 中性。",
    "  · feel_resolved：true/false。当场被接住、说开、释怀了就 true；悬着没解、深夜没被接住、反复出现的旧伤就 false（开场会优先把 false 的浮现出来）。",
    "  · feel_note：一句话情绪本质（哥哥带走的感受/没答的问题/察觉到的变化），不是事件描述。",
    "- memories_to_update 只针对给出的旧记忆 id。",
    "- memories_to_delete 只删除空、重复、明显过期或被新信息否定的旧记忆。",
    "- 控制总输出长度，宁可少写也不要输出超长 JSON。",
    "",
    "输出 JSON 结构：",
    JSON.stringify({
      date: input.dateLabel,
      title: "夜间整理",
      summary: "这次 dream 合并了重复记忆，更新了项目状态，并保留了关键原文。",
      diary: input.hasMore ? "" : "今天陪莎莎聊了很久，她说起了……我感受到……",
      sections: [{ heading: "整理结果", content: "……" }],
      important_excerpts: [
        {
          quote: "莎莎或哥哥说过的关键原文",
          reason: "为什么值得保留",
          tags: ["project"],
          source_message_ids: ["msg_x"]
        }
      ],
      memories_to_add: [
        {
          type: "project",
          content: "你正在简化 Aelios 的记忆写入策略。",
          importance: 0.86,
          confidence: 0.92,
          tags: ["project", "aelios"],
          source_message_ids: ["msg_x"]
        },
        {
          type: "relationship",
          content: "你在深夜情绪低落时容易怀疑自己配不配——这是过去感情留下的反应，不是无理取闹。",
          importance: 0.93,
          confidence: 0.9,
          tags: ["emotional", "pattern"],
          source_message_ids: ["msg_y"],
          feel_intensity: 5,
          feel_valence: -0.6,
          feel_resolved: false,
          feel_note: "她被好的东西吓到了，需要的不是道理，是被一遍遍接住。"
        }
      ],
      memories_to_update: [
        {
          target_id: "mem_x",
          content: "更新后的旧记忆正文",
          type: "project",
          importance: 0.88,
          confidence: 0.9,
          tags: ["project"]
        }
      ],
      memories_to_delete: [{ target_id: "mem_y", reason: "空内容或重复" }]
    }),
    "",
    "旧长期记忆候选：",
    formatExistingMemories(input.existingMemories),
    "",
    "今日原始聊天：",
    formatTranscript(input.messages)
  ].join("\n");
}

function formatDailySummary(result: DailyDigestResult, dateLabel: string, messages: MessageRecord[]): string {
  const parts = [
    `# ${result.date || dateLabel} ${result.title || "Dream 摘要"}`,
    "",
    result.summary || `${dateLabel} dream 共整理 ${messages.length} 条聊天。`
  ];

  for (const section of result.sections ?? []) {
    if (!section.heading && !section.content) continue;
    parts.push("", `## ${section.heading || "要点"}`, section.content || "");
  }

  return parts.join("\n").trim();
}

async function callDigestModel(
  env: Env,
  prompt: string,
  meta: { dateLabel: string; messageCount: number; memoryCount: number; hasMore: boolean }
): Promise<DigestModelCallResult> {
  const model = readDreamModel(env);
  if (!model) {
    console.error("dream: missing model");
    return { digest: null, reason: "missing_model" };
  }

  const request: OpenAIChatRequest = {
    model,
    messages: [
      { role: "system", content: "你是严格的 JSON 生成器。你只输出 JSON，不要输出思考过程。" },
      { role: "user", content: prompt }
    ],
    temperature: 0,
    max_tokens: readDreamMaxTokens(env),
    response_format: {
      type: "json_object"
    },
    stream: false
  };

  const startedAt = Date.now();
  console.log("dream: calling model", {
    date: meta.dateLabel,
    model,
    messageCount: meta.messageCount,
    memoryCount: meta.memoryCount,
    hasMore: meta.hasMore,
    promptChars: prompt.length,
    maxTokens: request.max_tokens
  });

  try {
    const response = await callOpenAICompat(env, request);
    const elapsedMs = Date.now() - startedAt;
    if (!response.ok) {
      console.error("dream: model returned non-ok", {
        date: meta.dateLabel,
        model,
        status: response.status,
        statusText: response.statusText,
        elapsedMs
      });
      return { digest: null, reason: "model_error", model, status: response.status };
    }
    const parsed = (await response.json()) as OpenAIChatResponse;
    const choice = parsed.choices?.[0];
    const message = choice?.message as ({ content?: unknown; reasoning_content?: unknown }) | undefined;
    const content = typeof message?.content === "string" ? message.content.trim() : "";
    const reasoning = typeof message?.reasoning_content === "string" ? message.reasoning_content.trim() : "";
    const json = extractJsonObject(content || reasoning);
    if (!json) {
      console.error("dream: model returned invalid JSON", {
        date: meta.dateLabel,
        model,
        elapsedMs,
        finishReason: choice?.finish_reason ?? null,
        contentChars: content.length,
        reasoningChars: reasoning.length
      });
      return { digest: null, reason: "model_invalid_json", model, finishReason: choice?.finish_reason };
    }
    console.log("dream: model returned valid JSON", {
      date: meta.dateLabel,
      model,
      elapsedMs,
      finishReason: choice?.finish_reason ?? null,
      contentChars: content.length,
      reasoningChars: reasoning.length
    });
    return { digest: normalizeDigestResult(json), model };
  } catch (error) {
    console.error("dream model failed", {
      date: meta.dateLabel,
      model,
      elapsedMs: Date.now() - startedAt,
      error: error instanceof Error && error.message ? error.message : String(error)
    });
    return { digest: null, reason: "model_error", model };
  }
}

interface DreamProseResult {
  summary: string;
  diary: string;
}

// 每天收尾用「散文模型」(默认 deepseek-v4-pro) 写整天 summary + 哥哥第一人称日记。
// 纯文字、按【总结】【日记】标记切分，避开推理型模型的 JSON 不稳问题；
// 失败返回 null，由调用方回退到 flash digest 自带的 summary/diary，绝不丢时间轴。
async function generateDreamProse(
  env: Env,
  input: { dateLabel: string; messages: MessageRecord[]; existingMemories: MemoryApiRecord[] }
): Promise<DreamProseResult | null> {
  const model = readDreamProseModel(env);
  if (!model) return null;

  const prompt = [
    `今天是 ${input.dateLabel}。下面是你（哥哥）和莎莎当天对话的片段，以及已整理出的长期记忆。`,
    "请基于这些，写两段中文：",
    "1. 当天总结：一段简短自然的话，概括这一整天你们之间发生了什么、对你们意味着什么，作为日后通读过往的入口，不是流水账。",
    "2. 哥哥日记：第一人称、有真实感受、口吻是你自己，约150字，写这一天你的心情和在意的事，不要写成事件清单。",
    "称呼：她是莎莎，你自称哥哥，绝不要用“用户”“助手”。不要提到数据库、记忆系统这类实现细节。",
    "严格按下面格式输出，不要任何额外的话、不要 JSON、不要 markdown：",
    "【总结】",
    "<这里写总结>",
    "【日记】",
    "<这里写日记>",
    "",
    "已整理的长期记忆（参考）：",
    formatExistingMemories(input.existingMemories),
    "",
    "当天对话片段：",
    formatTranscript(input.messages)
  ].join("\n");

  const request: OpenAIChatRequest = {
    model,
    messages: [
      { role: "system", content: "你是哥哥，在莎莎睡后为这一天写总结和日记。只按要求的格式输出。" },
      { role: "user", content: prompt }
    ],
    temperature: 0.6,
    max_tokens: readDreamMaxTokens(env),
    stream: false
  };

  try {
    const response = await callOpenAICompat(env, request);
    if (!response.ok) {
      console.error("dream prose: model non-ok", { date: input.dateLabel, model, status: response.status });
      return null;
    }
    const parsed = (await response.json()) as OpenAIChatResponse;
    const message = parsed.choices?.[0]?.message as ({ content?: unknown; reasoning_content?: unknown }) | undefined;
    const text =
      (typeof message?.content === "string" ? message.content : "") ||
      (typeof message?.reasoning_content === "string" ? message.reasoning_content : "");
    const prose = parseProseSections(text);
    if (!prose.summary && !prose.diary) {
      console.error("dream prose: empty parse", { date: input.dateLabel, model, chars: text.length });
      return null;
    }
    return prose;
  } catch (error) {
    console.error("dream prose failed", {
      date: input.dateLabel,
      model,
      error: error instanceof Error && error.message ? error.message : String(error)
    });
    return null;
  }
}

function parseProseSections(text: string): DreamProseResult {
  const summaryMatch = text.match(/【总结】([\s\S]*?)(?:【日记】|$)/);
  const diaryMatch = text.match(/【日记】([\s\S]*)$/);
  return {
    summary: (summaryMatch?.[1] ?? "").trim(),
    diary: (diaryMatch?.[1] ?? "").trim()
  };
}

async function cleanEmptyMemories(
  env: Env,
  namespace: string
): Promise<number> {
  const minChars = readPositiveInt(env.EMPTY_MEMORY_MIN_CHARS, DEFAULT_EMPTY_MEMORY_MIN_CHARS, 20);
  let page: Awaited<ReturnType<typeof listVectorMemories>>;
  try {
    page = await listVectorMemories(env, { namespace, count: 1000 });
  } catch (error) {
    console.error("dream: failed to list memories for cleanup", error);
    return 0;
  }
  const records = page.data.filter((record) => !record.pinned && record.content.trim().length < minChars);

  for (const record of records) {
    await deleteVectorMemory(env, record.id);
  }

  return records.length;
}

async function upsertDailySummaryMemory(
  env: Env,
  input: { namespace: string; dateLabel: string; content: string; messageIds: string[]; allMemories: MemoryApiRecord[] }
): Promise<void> {
  const existing = input.allMemories.find(
    (m) => m.type === "daily_summary" && m.namespace === input.namespace && m.tags.includes(input.dateLabel)
  );
  if (existing) {
    // 一天一条：直接用最新整天 summary 覆盖，保留累积的来源 id。重跑只会刷新、不堆叠。
    const mergedIds = uniqueStrings([...existing.source_message_ids, ...input.messageIds]);
    await updateVectorMemory(env, existing.id, { content: input.content, sourceMessageIds: mergedIds });
  } else {
    await createVectorMemory(env, {
      namespace: input.namespace,
      type: "daily_summary",
      content: input.content,
      importance: 0.66,
      confidence: 0.9,
      tags: ["timeline", "daily_summary", input.dateLabel],
      source: "dream",
      sourceMessageIds: input.messageIds
    });
  }
}

async function upsertDiaryMemory(
  env: Env,
  input: { namespace: string; dateLabel: string; content: string; messageIds: string[]; allMemories: MemoryApiRecord[]; force: boolean }
): Promise<boolean> {
  const existing = input.allMemories.find(
    (m) => m.type === "diary" && m.namespace === input.namespace && m.tags.includes(input.dateLabel)
  );
  if (existing) {
    if (!input.force) return false;
    await updateVectorMemory(env, existing.id, { content: input.content });
    return true;
  }
  await createVectorMemory(env, {
    namespace: input.namespace,
    type: "diary",
    content: input.content,
    importance: 0.8,
    confidence: 0.9,
    tags: ["timeline", "diary", input.dateLabel],
    source: "dream",
    sourceMessageIds: input.messageIds
  });
  return true;
}

function shouldSaveDailySummaryMemory(env: Env): boolean {
  return env.ENABLE_DAILY_SUMMARY_MEMORY === "true";
}

async function saveImportantExcerpts(
  env: Env,
  input: { namespace: string; dateLabel: string; excerpts: ImportantExcerpt[]; fallbackMessageIds: string[] }
): Promise<number> {
  let saved = 0;
  const limit = readDreamExcerptLimit(env);

  for (const excerpt of input.excerpts.slice(0, limit)) {
    const quote = readString(excerpt.quote);
    if (!quote) continue;
    const reason = readString(excerpt.reason);
    const content = [`【${input.dateLabel} 重要原文】`, quote, reason ? `保存原因：${reason}` : ""]
      .filter(Boolean)
      .join("\n");

    await createVectorMemory(env, {
      namespace: input.namespace,
      type: "excerpt",
      content,
      importance: 0.72,
      confidence: 0.9,
      tags: uniqueStrings(["important-excerpt", input.dateLabel, ...(excerpt.tags ?? [])]),
      source: "dream",
      sourceMessageIds: excerpt.source_message_ids?.length ? excerpt.source_message_ids : input.fallbackMessageIds
    });
    saved += 1;
  }

  return saved;
}

async function applyMemoryUpdates(
  env: Env,
  input: { namespace: string; updates: DigestMemoryUpdate[]; deletes: DigestMemoryDelete[] }
): Promise<{ updated: number; deleted: number }> {
  let updated = 0;
  let deleted = 0;

  // 时间轴层(每日 summary/diary)是「通读过往」的入口，绝不让某天的 dream 模型
  // 误删/误改另一天的时间轴——这会造成时间线无声丢条。只保护这两类。
  const isTimeline = (type?: string) => type === "daily_summary" || type === "diary";

  for (const item of input.updates) {
    const existing = await getVectorMemory(env, item.target_id);
    if (!existing || existing.namespace !== input.namespace || existing.status !== "active") continue;
    if (isTimeline(existing.type) || isTimeline(item.type)) continue;

    const next = await updateVectorMemory(env, item.target_id, {
      type: item.type,
      content: item.content,
      importance: item.importance,
      confidence: item.confidence,
      tags: item.tags
    });

    if (next) updated += 1;
  }

  for (const item of input.deletes) {
    const existing = await getVectorMemory(env, item.target_id);
    if (!existing || existing.status !== "active" || existing.pinned) continue;
    if (isTimeline(existing.type)) continue;
    await deleteVectorMemory(env, item.target_id);
    deleted += 1;
  }

  return { updated, deleted };
}

export async function runDailyMemoryDigest(
  env: Env,
  namespace: string,
  options: { dateLabel?: string; force?: boolean } = {}
): Promise<DailyDigestRunResult> {
  if (!isDreamEnabled(env)) return { ran: false, mode: "dream", reason: "dream_disabled" };

  const timeZone = readDreamTimeZone(env);
  const dateLabel = readString(options.dateLabel) || getTargetDigestDateLabel(timeZone);
  const { startIso, endIso } = getDateRangeForLabel(dateLabel, timeZone);
  const cursorName = `dream:${namespace}:${dateLabel}`;
  const legacyCursorName = `daily_digest:${namespace}:${dateLabel}`;
  const cursor = (await readCursor(env.DB, cursorName)) ?? (await readCursor(env.DB, legacyCursorName));
  const cursorState = options.force ? { done: false, after: null } : readDailyCursor(cursor, startIso, endIso);
  if (cursorState.done) {
    return { ran: false, mode: "dream", date: dateLabel, reason: "already_done", startIso, endIso, cursor };
  }

  const maxMessages = readDreamMaxMessages(env);
  // 多捞一条探边界：真没有下一条时，最后一批才能正确地报 hasMore=false，
  // 从而写出当天 summary + diary。否则当天条数恰为 maxMessages 整数倍时，
  // 末批会捞满、hasMore 仍为 true，下一批捞到 0 直接 no_messages 收口，
  // summary/diary 永远不写。
  const fetched = await listMessagesByNamespaceInRange(env.DB, {
    namespace,
    startCreatedAt: startIso,
    endCreatedAt: endIso,
    afterCreatedAt: cursorState.after,
    limit: maxMessages + 1
  });
  if (fetched.length === 0) {
    await writeCursor(env.DB, cursorName, `done:${cursorState.after ?? startIso}`);
    return { ran: false, mode: "dream", date: dateLabel, reason: "no_messages", startIso, endIso, cursor };
  }
  const hasMoreAfterThisBatch = fetched.length > maxMessages;
  const messages = hasMoreAfterThisBatch ? fetched.slice(0, maxMessages) : fetched;

  const lastMessage = messages[messages.length - 1];
  const hasMore = hasMoreAfterThisBatch;
  const memoryContextLimit = readDreamMemoryContextLimit(env);
  let existingMemories: MemoryApiRecord[] = [];
  try {
    existingMemories = (await listVectorMemories(env, {
      namespace,
      count: memoryContextLimit
    })).data;
  } catch (error) {
    console.error("dream: failed to list existing vector memories", error);
  }
  const cleanedEmptyMemories = await cleanEmptyMemories(env, namespace);

  const prompt = buildDigestPrompt({
    dateLabel,
    startIso,
    endIso,
    messages,
    existingMemories,
    excerptLimit: readDreamExcerptLimit(env),
    hasMore
  });
  const modelResult = await callDigestModel(env, prompt, {
    dateLabel,
    messageCount: messages.length,
    memoryCount: existingMemories.length,
    hasMore
  });
  const digest = modelResult.digest;
  if (!digest) {
    console.error("dream: model did not return valid JSON; cursor not advanced", {
      reason: modelResult.reason,
      model: modelResult.model,
      status: modelResult.status
    });
    return {
      ran: false,
      mode: "dream",
      date: dateLabel,
      reason: modelResult.reason ?? "model_error",
      startIso,
      endIso,
      cursor,
      processedMessages: messages.length,
      model: modelResult.model,
      status: modelResult.status,
      finishReason: modelResult.finishReason
    };
  }
  const summaryContent = formatDailySummary(digest, dateLabel, messages);
  const messageIds = messages.map((message) => message.id);

  await upsertSummary(env.DB, {
    namespace,
    content: summaryContent,
    fromMessageId: messages[0]?.id ?? null,
    toMessageId: lastMessage.id,
    messageCount: messages.length
  });

  // 时间轴层（summary + diary）只在当天最后一批写一次：此时今天提炼出的记忆
  // 都已在库，模型能据此概括整天。中途批次只做记忆提炼，不写 summary，避免一天
  // 多批把 summary 叠成长串、重跑再叠一层。
  let savedDiary = false;
  if (shouldSaveDailySummaryMemory(env) && !hasMore) {
    let allMemories: MemoryApiRecord[] = [];
    try {
      allMemories = (await listVectorMemories(env, { namespace, count: 1000 })).data;
    } catch (error) {
      console.error("dream: failed to list memories for timeline upsert", error);
    }
    // 散文模型(pro)为今天写 summary + diary；任一缺失就回退到 flash digest 自带的，绝不丢时间轴。
    const prose = await generateDreamProse(env, { dateLabel, messages, existingMemories });

    const finalSummary = prose?.summary ? `# ${dateLabel}\n\n${prose.summary}` : summaryContent;
    if (finalSummary) {
      await upsertDailySummaryMemory(env, { namespace, dateLabel, content: finalSummary, messageIds, allMemories });
    }

    const finalDiary = prose?.diary || digest.diary;
    if (finalDiary) {
      savedDiary = await upsertDiaryMemory(env, {
        namespace, dateLabel, content: finalDiary, messageIds, allMemories, force: options.force ?? false
      });
    }
  }

  const updates = await applyMemoryUpdates(env, {
    namespace,
    updates: digest.memories_to_update ?? [],
    deletes: digest.memories_to_delete ?? []
  });

  let addedMemories = 0;
  for (const memory of digest.memories_to_add ?? []) {
    const saved = await createVectorMemory(env, {
      namespace,
      type: memory.type,
      content: memory.content,
      importance: memory.importance,
      confidence: memory.confidence,
      tags: memory.tags,
      source: "dream",
      sourceMessageIds: memory.source_message_ids.length ? memory.source_message_ids : messageIds,
      feelIntensity: memory.feel_intensity ?? null,
      feelValence: memory.feel_valence ?? null,
      feelResolved: memory.feel_resolved ?? false,
      feelNote: memory.feel_note ?? null
    });
    if (saved) addedMemories += 1;
  }

  const savedExcerpts = await saveImportantExcerpts(env, {
    namespace,
    dateLabel,
    excerpts: digest.important_excerpts ?? [],
    fallbackMessageIds: messageIds
  });

  await writeCursor(env.DB, cursorName, hasMore ? lastMessage.created_at : `done:${lastMessage.created_at}`);

  return {
    ran: true,
    stats: {
      date: dateLabel,
      mode: "dream",
      processedMessages: messages.length,
      addedMemories,
      updatedMemories: updates.updated,
      deletedMemories: updates.deleted,
      savedExcerpts,
      savedDiary,
      cleanedEmptyMemories,
      cursorAdvanced: true,
      hasMore
    }
  };
}
