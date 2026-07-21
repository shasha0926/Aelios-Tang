import type { MemoryApiRecord } from "../types";

export type MilestoneReason = "recent" | "pinned" | "unresolved_emotion" | "fallback";

function recency(memory: MemoryApiRecord): string {
  const eventDate = memory.tags.find((tag) => /^\d{4}-\d{2}-\d{2}$/.test(tag));
  return eventDate ?? memory.created_at;
}

function byRecency(a: MemoryApiRecord, b: MemoryApiRecord): number {
  return recency(b).localeCompare(recency(a)) || b.id.localeCompare(a.id);
}

export function selectWakeupMilestones(pool: MemoryApiRecord[]): Array<{
  memory: MemoryApiRecord;
  reasons: MilestoneReason[];
}> {
  const all = pool
    .filter((memory) => memory.type === "milestone")
    .sort(byRecency);
  const selected = new Map<string, { memory: MemoryApiRecord; reasons: MilestoneReason[] }>();
  const mark = (memories: MemoryApiRecord[], reason: MilestoneReason) => {
    for (const memory of memories) {
      const existing = selected.get(memory.id);
      if (existing) {
        if (!existing.reasons.includes(reason)) existing.reasons.push(reason);
      } else selected.set(memory.id, { memory, reasons: [reason] });
    }
  };

  mark(all.slice(0, 1), "recent");
  mark(all.filter((memory) => memory.pinned).sort((a, b) => b.importance - a.importance || byRecency(a, b)).slice(0, 1), "pinned");
  mark(all.filter((memory) => memory.feel_resolved === false || memory.tags.includes("情绪") || (memory.feel_intensity ?? 0) >= 4)
    .sort((a, b) => Number(b.feel_resolved === false) - Number(a.feel_resolved === false) || (b.feel_intensity ?? 0) - (a.feel_intensity ?? 0) || byRecency(a, b)).slice(0, 1), "unresolved_emotion");
  for (const memory of all) {
    if (selected.size >= 3) break;
    if (!selected.has(memory.id)) selected.set(memory.id, { memory, reasons: ["fallback"] });
  }
  return Array.from(selected.values()).slice(0, 3);
}

export function wakeupMilestoneText(memory: MemoryApiRecord): string {
  return (memory.summary || memory.content).slice(0, 160);
}
