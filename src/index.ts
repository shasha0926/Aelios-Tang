import { handleAdmin } from "./api/admin";
import { handleHealth } from "./api/health";
import { handleCache } from "./api/cache";
import { handleCacheHealth, handleVectorHealth, handleVectorReindex } from "./api/debug";
import { handleChatCompletions } from "./api/chatCompletions";
import { handleGuideDogChatCompletions } from "./api/guideDog";
import { handleIngestMessagesApi, handleMemories, handleSearchMemoriesApi } from "./api/memories";
import { handleMcp } from "./api/mcp";
import { handleMessagesAdmin } from "./api/messagesAdmin";
import { handleModels } from "./api/models";
import { runDailyMemoryDigest } from "./memory/dailyDigest";
import { runMemoryRetention } from "./memory/retention";
import { handleQueueMessage } from "./queue/consumer";
import type { Env, QueueMessage } from "./types";
import { openAiError } from "./utils/json";

function getDailyDigestNamespace(env: Env): string {
  return env.DREAM_NAMESPACE?.trim() || "default";
}

function getDailyDigestMaxRuns(env: Env): number {
  const parsed = Number(env.DREAM_MAX_RUNS || env.DAILY_DIGEST_MAX_RUNS || 10);
  if (!Number.isFinite(parsed)) return 10;
  return Math.min(Math.max(Math.floor(parsed), 1), 10);
}

// 一天可能要分多批读完才轮到「收口批」(hasMore=false) 写 summary+diary。
// 旧逻辑 `if (!result.ran || ...) break` 把任何 ran=false 都当终止——一旦遇到
// 模型瞬时错误(503/坏JSON)就半途中断，当天 summary+diary 永远不写。这里区分
// 「正常终止(没消息/已完成)」与「瞬时错误」：前者收工，后者重试，直到真正收口。
const TERMINAL_DIGEST_REASONS = new Set(["no_messages", "already_done", "dream_disabled"]);
const MAX_TRANSIENT_RETRIES = 3;

async function runDailyMemoryDigestBatches(env: Env, namespace: string): Promise<unknown[]> {
  const results: unknown[] = [];
  const maxRuns = getDailyDigestMaxRuns(env);
  let processed = 0;
  let transientRetries = 0;

  while (processed < maxRuns) {
    const result = await runDailyMemoryDigest(env, namespace);
    results.push(result);

    if (result.ran) {
      processed += 1;
      transientRetries = 0;
      if (!result.stats?.hasMore) break; // 收口完成（summary+diary 已写）
      continue;
    }

    // ran === false
    if (TERMINAL_DIGEST_REASONS.has(result.reason ?? "")) break; // 正常终止
    // 瞬时错误：不放弃收口，重试同一批；连续超限则留给下次 cron 续跑（游标未 done）
    transientRetries += 1;
    if (transientRetries >= MAX_TRANSIENT_RETRIES) break;
  }

  return results;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "GET" && (url.pathname === "/admin" || url.pathname === "/memory-admin")) {
      return handleAdmin();
    }

    if (request.method === "GET" && url.pathname === "/health") {
      return handleHealth(env);
    }

    if (request.method === "GET" && url.pathname === "/v1/models") {
      return handleModels(request, env);
    }

    if (request.method === "POST" && url.pathname === "/v1/chat/completions") {
      return handleChatCompletions(request, env, ctx);
    }

    if (
      request.method === "POST" &&
      (url.pathname === "/v1/guide-dog/chat/completions" || url.pathname === "/guide-dog/v1/chat/completions")
    ) {
      return handleGuideDogChatCompletions(request, env);
    }

    if (url.pathname === "/mcp" || url.pathname === "/memory-mcp") {
      return handleMcp(request, env, ctx);
    }

    if (url.pathname === "/v1/admin/messages") {
      return handleMessagesAdmin(request, env);
    }

    if (url.pathname.startsWith("/v1/memories")) {
      return handleMemories(request, env, ctx);
    }

    if (url.pathname === "/v1/memory" || url.pathname.startsWith("/v1/memory/")) {
      return handleMemories(request, env, ctx);
    }

    if (
      request.method === "POST" &&
      (url.pathname === "/v1/ingest/messages" || url.pathname === "/v1/messages/ingest")
    ) {
      return handleIngestMessagesApi(request, env, ctx);
    }

    if (request.method === "POST" && url.pathname === "/v1/search/memories") {
      return handleSearchMemoriesApi(request, env);
    }

    if (url.pathname.startsWith("/v1/cache/")) {
      return handleCache(request, env);
    }

    if (request.method === "GET" && url.pathname === "/v1/debug/cache_health") {
      return handleCacheHealth(request, env);
    }

    if (request.method === "GET" && url.pathname === "/v1/debug/vector_health") {
      return handleVectorHealth(request, env);
    }

    if (request.method === "POST" && url.pathname === "/v1/debug/vector_reindex") {
      return handleVectorReindex(request, env);
    }

    return openAiError("Not found", 404);
  },

  async queue(batch: MessageBatch<QueueMessage>, env: Env): Promise<void> {
    for (const message of batch.messages) {
      try {
        await handleQueueMessage(message.body, env);
        message.ack();
      } catch (error) {
        console.error("queue message failed", error);
        message.retry();
      }
    }
  },

  async scheduled(_controller: ScheduledController, env: Env, ctx: ExecutionContext): Promise<void> {
    const namespace = getDailyDigestNamespace(env);
    ctx.waitUntil(
      Promise.all([
        runDailyMemoryDigestBatches(env, namespace),
        runMemoryRetention(env, namespace)
      ]).then(([digest, retention]) => {
        console.log("scheduled daily memory maintenance", { namespace, digest, retention });
      })
    );
  }
};
