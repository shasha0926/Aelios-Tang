import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const out = fs.mkdtempSync(path.join(os.tmpdir(), "aelios-wakeup-"));
fs.writeFileSync(path.join(out, "package.json"), '{"type":"module"}\n');
execFileSync("./node_modules/.bin/tsc", ["--noEmit", "false", "--outDir", out, "--target", "ES2022", "--lib", "ES2022", "--types", "@cloudflare/workers-types", "--module", "ES2022", "--moduleResolution", "Bundler", "src/memory/wakeupMilestones.ts"], { stdio: "inherit" });
function findCompiledModule(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const entryPath = path.join(dir, entry.name);
    if (entry.isFile() && entry.name === "wakeupMilestones.js") return entryPath;
    if (entry.isDirectory()) {
      const found = findCompiledModule(entryPath);
      if (found) return found;
    }
  }
  return null;
}

const compiledModule = findCompiledModule(out);
if (!compiledModule) throw new Error("compiled wakeupMilestones.js was not found");
const { selectWakeupMilestones, wakeupMilestoneText } = await import(pathToFileURL(compiledModule).href);

const m = (id, patch = {}) => ({ id, type: "milestone", content: `正文-${id}`, summary: null, importance: 0.5, pinned: false, tags: [], created_at: `2026-07-0${id}T00:00:00.000Z`, feel_resolved: true, feel_intensity: null, ...patch });
const pool = [m("1", { pinned: true, importance: 0.9 }), m("2", { feel_resolved: false, feel_intensity: 5 }), m("3"), m("4")];
const selected = selectWakeupMilestones(pool);
assert.equal(selected.length, 3);
assert.equal(new Set(selected.map((x) => x.memory.id)).size, 3);
assert.ok(selected.some((x) => x.reasons.includes("recent")));
assert.ok(selected.some((x) => x.reasons.includes("pinned")));
assert.ok(selected.some((x) => x.reasons.includes("unresolved_emotion")));
assert.equal(selectWakeupMilestones([m("1", { pinned: true, feel_resolved: false, feel_intensity: 5 })])[0].reasons.join("+"), "recent+pinned+unresolved_emotion");
assert.deepEqual(selectWakeupMilestones([m("1"), m("2")]).map((x) => x.reasons), [["recent"], ["fallback"]]);
assert.ok(selectWakeupMilestones([m("1"), m("2", { tags: ["情绪"] })]).some((x) => x.memory.id === "2" && x.reasons.includes("unresolved_emotion")));
const sameTime = [m("1", { created_at: "2026-07-01T00:00:00.000Z" }), m("2", { created_at: "2026-07-01T00:00:00.000Z" })];
assert.deepEqual(selectWakeupMilestones(sameTime).map((x) => x.memory.id), selectWakeupMilestones([...sameTime].reverse()).map((x) => x.memory.id));
assert.equal(selectWakeupMilestones([m("1", { created_at: "2026-07-04T00:00:00.000Z" }), m("2", { created_at: "2026-07-01T00:00:00.000Z", tags: ["2026-07-05"] })])[0].memory.id, "2");
assert.equal(wakeupMilestoneText(m("1", { summary: "摘要".repeat(100) })).length, 160);
assert.equal(wakeupMilestoneText(m("1", { summary: null, content: "正文".repeat(100) })).length, 160);
console.log("wakeup milestone contract: PASS");
