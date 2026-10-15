"use client";

import clsx from "clsx";
import {
  AlertCircle,
  Check,
  Clapperboard,
  CloudOff,
  Info,
  Plus,
  Sparkles,
  UserRound,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  api,
  ApiError,
  narrationSeconds,
  newScene,
  type Character,
  type Project,
  type Scene,
  type Script,
} from "@/lib/api";
import { SceneCard, SCENE_HUES } from "./SceneCard";
import { Spinner, useToast } from "./ui";

type SaveState = "saved" | "dirty" | "saving" | "error";
let keySeed = 0;
const freshKey = () => `s${++keySeed}`;

interface Props {
  project: Project;
  readOnly: boolean;
  onProjectChange: (project: Project) => void;
}

export function ScriptEditor({ project, readOnly, onProjectChange }: Props) {
  const toast = useToast();
  const [script, setScript] = useState<Script>(() => project.script!);
  const [keys, setKeys] = useState<string[]>(() => project.script!.scenes.map(freshKey));
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [saveState, setSaveState] = useState<SaveState>("saved");
  const [rendering, setRendering] = useState(false);

  // Refs let the debounced save see the latest state without re-subscribing.
  const edits = useRef(0);
  const revision = useRef(project.script_revision);
  const latest = useRef({ script, keys });
  latest.current = { script, keys };
  const inflight = useRef<Promise<Project | null> | null>(null);

  // Adopt server-side script changes (e.g. after re-running the conversion) when clean.
  useEffect(() => {
    if (project.script_revision !== revision.current && saveState === "saved" && project.script) {
      revision.current = project.script_revision;
      setScript(project.script);
      setKeys(project.script.scenes.map(freshKey));
      setExcluded(new Set());
    }
  }, [project.script_revision, project.script, saveState]);

  const edit = useCallback((next: (current: Script) => Script, nextKeys?: (keys: string[]) => string[]) => {
    edits.current++;
    setScript(next);
    if (nextKeys) setKeys(nextKeys);
    setSaveState("dirty");
  }, []);

  const save = useCallback(async (): Promise<Project | null> => {
    if (inflight.current) await inflight.current;
    const startedAt = edits.current;
    const { script: snapshot } = latest.current;
    setSaveState("saving");
    const task = (async () => {
      try {
        const saved = await api.saveScript(project.id, snapshot, revision.current);
        revision.current = saved.script_revision;
        if (edits.current === startedAt && saved.script) {
          // Nothing typed meanwhile: take server-derived prompts and timings.
          setScript(saved.script);
          setSaveState("saved");
        } else {
          setSaveState("dirty");
        }
        onProjectChange(saved);
        return saved;
      } catch (error) {
        setSaveState("error");
        const conflict = error instanceof ApiError && error.status === 409;
        toast("error", conflict ? `${(error as Error).message}.` : `Couldn't save: ${(error as Error).message}`);
        return null;
      }
    })();
    inflight.current = task;
    const result = await task;
    inflight.current = null;
    return result;
  }, [project.id, onProjectChange, toast]);

  // Debounced autosave.
  useEffect(() => {
    if (saveState !== "dirty" || readOnly) return;
    const timer = setTimeout(save, 1200);
    return () => clearTimeout(timer);
  }, [script, saveState, save, readOnly]);

  // Warn before leaving with unsaved edits; Cmd/Ctrl+S saves immediately.
  useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (saveState === "dirty" || saveState === "saving") event.preventDefault();
    };
    const keydown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "s") {
        event.preventDefault();
        if (saveState === "dirty" || saveState === "error") save();
      }
    };
    window.addEventListener("beforeunload", beforeUnload);
    window.addEventListener("keydown", keydown);
    return () => {
      window.removeEventListener("beforeunload", beforeUnload);
      window.removeEventListener("keydown", keydown);
    };
  }, [saveState, save]);

  const updateScene = (index: number, patch: Partial<Scene>) =>
    edit((current) => ({
      ...current,
      scenes: current.scenes.map((scene, i) => (i === index ? { ...scene, ...patch } : scene)),
    }));

  const move = (index: number, delta: -1 | 1) => {
    const swap = <T,>(items: T[]) => {
      const copy = [...items];
      [copy[index], copy[index + delta]] = [copy[index + delta], copy[index]];
      return copy;
    };
    edit((current) => ({ ...current, scenes: swap(current.scenes) }), swap);
  };

  const insertAt = (index: number, scene: Scene) =>
    edit(
      (current) => ({
        ...current,
        scenes: [...current.scenes.slice(0, index), scene, ...current.scenes.slice(index)],
      }),
      (current) => [...current.slice(0, index), freshKey(), ...current.slice(index)],
    );

  const remove = (index: number) =>
    edit(
      (current) => ({ ...current, scenes: current.scenes.filter((_, i) => i !== index) }),
      (current) => current.filter((_, i) => i !== index),
    );

  const selected = keys.flatMap((key, index) => (excluded.has(key) ? [] : [index + 1]));
  const totalSeconds = script.scenes.reduce(
    (sum, scene) => sum + narrationSeconds(scene.narration || scene.text || "x"),
    0,
  );

  const startRender = async () => {
    setRendering(true);
    try {
      if (saveState !== "saved") {
        const saved = await save();
        if (!saved) return;
      }
      const all = selected.length === script.scenes.length;
      const started = await api.render(project.id, all ? undefined : selected);
      onProjectChange(started);
    } catch (error) {
      toast("error", (error as Error).message);
    } finally {
      setRendering(false);
    }
  };

  const issuesByScene = useMemo(() => {
    const map = new Map<number, typeof project.issues>();
    for (const issue of project.issues) {
      if (issue.scene !== null) map.set(issue.scene, [...(map.get(issue.scene) ?? []), issue]);
    }
    return map;
  }, [project.issues]);
  const globalIssues = project.issues.filter((issue) => issue.scene === null);
  // Issues are computed server-side for the last saved version only.
  const issuesFresh = saveState === "saved";

  return (
    <>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)] xl:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        <aside className="space-y-4 lg:sticky lg:top-24 lg:max-h-[calc(100dvh-8rem)] lg:self-start lg:overflow-y-auto lg:pb-4 lg:pr-1">
          <Panel title="Story" icon={<Sparkles className="size-4" />}>
            <label className="block">
              <span className="field-label">Title</span>
              <input
                className="input font-display text-lg"
                value={script.title}
                readOnly={readOnly}
                onChange={(event) => edit((current) => ({ ...current, title: event.target.value }))}
              />
            </label>
            <label className="mt-3 block">
              <span className="field-label">Summary</span>
              <textarea
                className="input min-h-[4.5rem] resize-none leading-relaxed"
                value={script.summary}
                readOnly={readOnly}
                placeholder="A one-line summary of the story arc"
                onChange={(event) => edit((current) => ({ ...current, summary: event.target.value }))}
              />
            </label>
            <div className="mt-3 flex flex-wrap gap-1.5 text-xs">
              <span className="chip">{script.planner_backend} planner</span>
              <span className="chip">{script.scenes.length} scenes</span>
              <span className="chip">{script.source_sentences.length || project.source.sentence_count} sentences</span>
            </div>
          </Panel>

          {script.character && (
            <CharacterPanel
              character={script.character}
              readOnly={readOnly}
              onChange={(character) => edit((current) => ({ ...current, character }))}
            />
          )}

          {script.style && Object.keys(script.style).length > 0 && (
            <Panel title="Visual style" icon={<Clapperboard className="size-4" />}>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(script.style).map(([key, value]) => (
                  <label key={key} className={clsx("block", key === "type" && "col-span-2")}>
                    <span className="field-label">{key}</span>
                    <input
                      className="input"
                      value={value ?? ""}
                      readOnly={readOnly}
                      onChange={(event) =>
                        edit((current) => ({
                          ...current,
                          style: { ...current.style, [key]: event.target.value },
                        }))
                      }
                    />
                  </label>
                ))}
              </div>
            </Panel>
          )}

          {globalIssues.length > 0 && issuesFresh && (
            <Panel title="Review notes" icon={<Info className="size-4" />}>
              <ul className="space-y-2 text-sm text-ink-soft">
                {globalIssues.map((issue) => (
                  <li key={issue.message} className="flex gap-2">
                    <AlertCircle className={clsx("mt-0.5 size-4 shrink-0", issue.level === "warning" ? "text-warn" : "text-ink-faint")} />
                    <span className="break-words">{issue.message}</span>
                  </li>
                ))}
              </ul>
              <p className="mt-3 text-xs text-ink-faint">These are hints only; rendering is never blocked.</p>
            </Panel>
          )}
        </aside>

        <section className="min-w-0 space-y-4">
          <Timeline
            scenes={script.scenes}
            keys={keys}
            excluded={excluded}
            target={script.duration_seconds}
            totalSeconds={totalSeconds}
          />
          {script.scenes.map((scene, index) => (
            <div key={keys[index]} className="rise" style={{ animationDelay: `${Math.min(index, 8) * 70}ms` }}>
              <SceneCard
                scene={scene}
                index={index}
                total={script.scenes.length}
                anchorId={`scene-${keys[index]}`}
                character={script.character}
                issues={issuesFresh ? issuesByScene.get(index + 1) ?? [] : []}
                included={!excluded.has(keys[index])}
                readOnly={readOnly}
                onChange={(patch) => updateScene(index, patch)}
                onMove={(delta) => move(index, delta)}
                onDuplicate={() => insertAt(index + 1, { ...scene, source_sentence_ids: [] })}
                onDelete={() => remove(index)}
                onToggleIncluded={() =>
                  setExcluded((current) => {
                    const next = new Set(current);
                    if (next.has(keys[index])) next.delete(keys[index]);
                    else next.add(keys[index]);
                    return next;
                  })
                }
              />
            </div>
          ))}
          {!readOnly && script.scenes.length < 12 && (
            <button
              className="flex w-full items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-line py-5 text-sm font-medium text-ink-soft transition-colors hover:border-accent/50 hover:text-ink"
              onClick={() => insertAt(script.scenes.length, newScene(script.scenes.length + 1))}
            >
              <Plus className="size-4" /> Add scene
            </button>
          )}
        </section>
      </div>

      {!readOnly && (
        <div className="fixed inset-x-0 bottom-[calc(env(safe-area-inset-bottom)+3.75rem)] z-30 px-3 sm:bottom-5 sm:px-6">
          <div className="mx-auto flex max-w-3xl items-center gap-3 rounded-2xl border border-line bg-elevated/90 p-2 pl-4 shadow-float backdrop-blur-xl sm:rounded-full">
            <SaveIndicator state={saveState} onRetry={save} />
            <span className="hidden text-xs text-ink-faint md:inline">
              {selected.length}/{script.scenes.length} scenes · ~{Math.round(totalSeconds)}s narration
            </span>
            <div className="ml-auto flex items-center gap-2">
              {(saveState === "dirty" || saveState === "error") && (
                <button className="btn-quiet hidden h-9 sm:inline-flex" onClick={save}>
                  Save now
                </button>
              )}
              <button
                className="btn-primary h-10"
                disabled={rendering || selected.length === 0}
                onClick={startRender}
              >
                {rendering ? <Spinner /> : <Clapperboard className="size-4" />}
                {selected.length === script.scenes.length
                  ? "Generate video"
                  : `Render ${selected.length} scene${selected.length === 1 ? "" : "s"}`}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function SaveIndicator({ state, onRetry }: { state: SaveState; onRetry: () => void }) {
  if (state === "saving")
    return (
      <span className="flex items-center gap-1.5 text-xs text-ink-soft">
        <Spinner className="size-3.5" /> Saving…
      </span>
    );
  if (state === "dirty")
    return (
      <span className="flex items-center gap-1.5 text-xs text-ink-soft">
        <span className="size-1.5 rounded-full bg-warm" /> Unsaved
      </span>
    );
  if (state === "error")
    return (
      <button className="flex items-center gap-1.5 text-xs text-bad" onClick={onRetry}>
        <CloudOff className="size-3.5" /> Retry save
      </button>
    );
  return (
    <span className="flex items-center gap-1.5 text-xs text-ink-faint">
      <Check className="size-3.5 text-ok" /> Saved
    </span>
  );
}

function Panel({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <div className="card p-4 sm:p-5">
      <div className="mb-3 flex items-center gap-2 text-sm font-medium">
        <span className="text-ink-faint">{icon}</span>
        {title}
      </div>
      {children}
    </div>
  );
}

function CharacterPanel({
  character,
  readOnly,
  onChange,
}: {
  character: Character;
  readOnly: boolean;
  onChange: (character: Character) => void;
}) {
  const inferred = new Set(character.inferred_fields ?? []);
  const fields: [keyof Character & string, string][] = [
    ["name", "Name"],
    ["kind", "Kind"],
    ["age", "Age"],
    ["hair", "Hair"],
    ["eyes", "Eyes"],
    ["clothes", "Clothes"],
  ];
  const set = (key: string, value: string) => onChange({ ...character, [key]: value });
  const dictionary = (key: "outfits" | "states", label: string) => {
    const entries = Object.entries(character[key] ?? {});
    if (!entries.length) return null;
    return (
      <div className="mt-3">
        <span className="field-label">
          {label}
          {inferred.has(key) && <InferredTag />}
        </span>
        <div className="space-y-2">
          {entries.map(([id, value]) => (
            <div key={id} className="flex items-center gap-2">
              <span className="w-20 shrink-0 truncate rounded-lg bg-sunken px-2 py-1 font-mono text-[11px] text-ink-soft" title={id}>
                {id}
              </span>
              <input
                className="input py-1.5"
                value={value}
                readOnly={readOnly}
                onChange={(event) => onChange({ ...character, [key]: { ...character[key], [id]: event.target.value } })}
              />
            </div>
          ))}
        </div>
      </div>
    );
  };
  return (
    <Panel title="Main character" icon={<UserRound className="size-4" />}>
      <div className="grid grid-cols-2 gap-3">
        {fields
          .filter(([key]) => key === "name" || key === "kind" || character[key] != null)
          .map(([key, label]) => (
            <label key={key} className="block">
              <span className="field-label">
                {label}
                {inferred.has(key) && <InferredTag />}
              </span>
              <input
                className="input"
                value={String(character[key] ?? "")}
                readOnly={readOnly}
                onChange={(event) => set(key, event.target.value)}
              />
            </label>
          ))}
      </div>
      <label className="mt-3 block">
        <span className="field-label">
          Visual identity
          {inferred.has("visual_identity") && <InferredTag />}
        </span>
        <textarea
          className="input min-h-[3rem] resize-none"
          value={character.visual_identity}
          readOnly={readOnly}
          onChange={(event) => set("visual_identity", event.target.value)}
        />
      </label>
      {dictionary("outfits", "Outfits")}
      {dictionary("states", "States")}
    </Panel>
  );
}

function InferredTag() {
  return (
    <span
      className="rounded-full bg-warn-soft px-1.5 py-px text-[10px] normal-case tracking-normal text-warn"
      title="Not stated in the PDF; the planner chose it. Check it."
    >
      inferred
    </span>
  );
}

function Timeline({
  scenes,
  keys,
  excluded,
  target,
  totalSeconds,
}: {
  scenes: Scene[];
  keys: string[];
  excluded: Set<string>;
  target: number;
  totalSeconds: number;
}) {
  const span = Math.max(target, totalSeconds);
  return (
    <div className="card p-4 sm:p-5">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <p className="text-sm font-medium">Timeline</p>
        <p className={clsx("text-xs tabular-nums", totalSeconds > target * 1.2 ? "text-warn" : "text-ink-faint")}>
          ~{totalSeconds.toFixed(1)}s narration · {target}s target
        </p>
      </div>
      <div className="relative">
        <div className="flex h-12 gap-1">
          {scenes.map((scene, index) => {
            const seconds = narrationSeconds(scene.narration || scene.text || "x");
            return (
              <button
                key={keys[index]}
                onClick={() =>
                  document.getElementById(`scene-${keys[index]}`)?.scrollIntoView({ behavior: "smooth", block: "start" })
                }
                className={clsx(
                  "group relative min-w-0 overflow-hidden rounded-lg px-2 text-left text-[11px] font-medium text-white transition-[filter,opacity] hover:brightness-110",
                  excluded.has(keys[index]) && "opacity-30",
                )}
                style={{ flexGrow: seconds, flexBasis: 0, background: SCENE_HUES[index % SCENE_HUES.length] }}
                title={`Scene ${index + 1} · ~${seconds}s`}
              >
                <span className="block truncate">{index + 1}</span>
                <span className="block truncate opacity-75">{seconds}s</span>
              </button>
            );
          })}
        </div>
        <div
          className="pointer-events-none absolute -bottom-1 -top-1 w-px bg-ink"
          style={{ left: `${(target / span) * 100}%`, display: totalSeconds > target ? "block" : "none" }}
          title={`${target}s target`}
        />
      </div>
    </div>
  );
}
