import { authenticate } from "../auth/apiKey";
import { requireScope } from "../auth/scopes";
import {
  countMessagesBySource,
  deleteMessagesBySource,
  deleteAllMessagesInNamespace
} from "../db/messages";
import type { Env } from "../types";
import { json, openAiError } from "../utils/json";

// ---------------------------------------------------------------------------
// /v1/admin/messages —— 原文台账(④)运维端点。
//
// 存在的唯一目的：干净重建。官方导出会把 regenerate/编辑的废弃分支一并导出，
// 历史导入平铺全取后，④ 混入了大量莎莎未采用的草稿。本机无 wrangler/D1 直连，
// 所以用这个受 Bearer 鉴权保护的端点做"按 source 统计 + 定向删除"。
//
//   GET  /v1/admin/messages?namespace=default
//        → 按 source 统计条数与时间范围（REST 即时一致，可信）
//
//   DELETE /v1/admin/messages?namespace=default&source=历史导入v3&confirm=yes
//        → 删除某 source 的全部原文
//
//   DELETE /v1/admin/messages?namespace=default&all=yes&confirm=yes
//        → 删除该 namespace 的全部原文（慎用）
//
// confirm=yes 是防手滑的二次确认；缺了直接 400。
// ---------------------------------------------------------------------------
export async function handleMessagesAdmin(request: Request, env: Env): Promise<Response> {
  const auth = await authenticate(request, env);
  if (!auth.ok) return openAiError("Unauthorized", 401, "authentication_error");

  const scopeError = requireScope(auth.profile, "memory:write");
  if (scopeError) return scopeError;

  const url = new URL(request.url);
  const namespace = url.searchParams.get("namespace") || "default";

  if (request.method === "GET") {
    const bySource = await countMessagesBySource(env.DB, namespace);
    const total = bySource.reduce((sum, row) => sum + (row.count ?? 0), 0);
    return json({ data: { namespace, total, by_source: bySource } });
  }

  if (request.method === "DELETE") {
    if (url.searchParams.get("confirm") !== "yes") {
      return openAiError("Refusing to delete without confirm=yes", 400);
    }
    const source = url.searchParams.get("source");
    const all = url.searchParams.get("all") === "yes";

    if (source) {
      const deleted = await deleteMessagesBySource(env.DB, namespace, source);
      return json({ data: { namespace, source, deleted } });
    }
    if (all) {
      const deleted = await deleteAllMessagesInNamespace(env.DB, namespace);
      return json({ data: { namespace, all: true, deleted } });
    }
    return openAiError("Specify either source=... or all=yes", 400);
  }

  return openAiError("Method not allowed", 405);
}
