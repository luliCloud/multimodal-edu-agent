"use client";

import clsx from "clsx";
import {
  ArrowDown,
  ArrowUp,
  BookOpen,
  ChevronDown,
  Copy,
  RotateCcw,
  Trash2,
  Wand2,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import type { Character, Issue, Scene } from "@/lib/api";
import { narrationSeconds, wordCount } from "@/lib/api";

export const SCENE_HUES = ["#5b47e0", "#e8763a", "#1f9e8a", "#d0478f", "#3f7fd8", "#b58a1a", "#7c5cd6", "#3c9a52"];

/** Mirrors backend narration_word_range: per-scene word budget for a 15s short. */
export function narrationRange(sceneCount: number): [number, number] {
  if (sceneCount <= 4) return [6, 9];
  if (sceneCount === 5) return [5, 7];
  if (sceneCount === 6) return [4, 6];
  return [3, 6];
}

interface Props {
  scene: Scene;
  index: number;
  total: number;
  anchorId: string;
  character: Character | null;
  issues: Issue[];
  included: boolean;
  readOnly: boolean;
  onChange: (patch: Partial<Scene>) => void;
  onMove: (delta: -1 | 1) => void;
  onDuplicate: () => void;
  onDelete: () => void;
  onToggleIncluded: () => void;
}

export function SceneCard({
  scene,
  index,
  total,
  anchorId,
  character,
  issues,
  included,
  readOnly,
  onChange,
  onMove,
  onDuplicate,
  onDelete,
  onToggleIncluded,
}: Props) {
  const [showSource, setShowSource] = useState(false);
  const [showPrompt, setShowPrompt] = useState(false);
  const hue = SCENE_HUES[index % SCENE_HUES.length];
  const words = wordCount(scene.narration);
  const seconds = narrationSeconds(scene.narration || scene.text || "x");
  const [minWords, maxWords] = narrationRange(total);
  const budgeted = character !== null;
  const offBudget = budgeted && (words < minWords || words > maxWords);
  const outfits = Object.entries(character?.outfits ?? {});
  const states = Object.entries(character?.states ?? {});
  const heading =
    scene.action.trim() || scene.narration.trim().split(/[.!?]/)[0] || `Untitled scene`;

  return (
    <article
      id={anchorId}
      className={clsx(
        "card scroll-mt-28 overflow-hidden transition-[opacity,box-shadow] duration-200",
        !included && "opacity-60",
      )}
    >
      <div className="h-1" style={{ background: hue }} />
      <header className="flex items-start gap-3 px-4 pt-4 sm:px-5">
        <span
          className="font-display text-4xl leading-none tabular-nums"
          style={{ color: hue }}
          aria-hidden
        >
          {String(index + 1).padStart(2, "0")}
        </span>
        <div className="min-w-0 flex-1 pt-0.5">
          <p className="eyebrow">Scene {index + 1}</p>
          <h3 className="mt-0.5 truncate font-medium">{heading}</h3>
          {scene.pages.length > 0 && (
            <p className="mt-0.5 text-xs text-ink-faint">
              Source page{scene.pages.length > 1 ? "s" : ""} {scene.pages.join(", ")}
            </p>
          )}
        </div>
        {!readOnly && (
          <div className="-mr-2 flex shrink-0 items-center">
            <label
              className="mr-1 flex cursor-pointer items-center gap-1.5 rounded-full px-2 py-1 text-xs text-ink-soft hover:bg-sunken"
              title="Include this scene when rendering"
            >
              <input
                type="checkbox"
                checked={included}
                onChange={onToggleIncluded}
                className="size-3.5 accent-[var(--accent)]"
              />
              <span className="hidden sm:inline">Render</span>
            </label>
            <button className="icon-btn" onClick={() => onMove(-1)} disabled={index === 0} aria-label="Move scene up">
              <ArrowUp className="size-4" />
            </button>
            <button className="icon-btn" onClick={() => onMove(1)} disabled={index === total - 1} aria-label="Move scene down">
              <ArrowDown className="size-4" />
            </button>
            <button className="icon-btn hidden sm:inline-flex" onClick={onDuplicate} aria-label="Duplicate scene">
              <Copy className="size-4" />
            </button>
            <button className="icon-btn hover:text-bad" onClick={onDelete} disabled={total <= 1} aria-label="Delete scene">
              <Trash2 className="size-4" />
            </button>
          </div>
        )}
      </header>

      {issues.length > 0 && (
        <ul className="mx-4 mt-3 space-y-1 rounded-xl bg-warn-soft px-3 py-2 text-xs text-warn sm:mx-5">
          {issues.map((issue) => (
            <li key={issue.message}>{issue.message}</li>
          ))}
        </ul>
      )}

      <div className="space-y-4 p-4 sm:p-5">
        <div>
          <label className="field-label" htmlFor={`${anchorId}-narration`}>
            Narration
            <span className={clsx("normal-case tracking-normal", offBudget ? "text-warn" : "text-ink-faint")}>
              {words} words{budgeted ? ` (aim ${minWords}–${maxWords})` : ""} · ~{seconds}s
            </span>
          </label>
          <textarea
            id={`${anchorId}-narration`}
            className="input min-h-[3.5rem] resize-none font-display text-lg leading-snug"
            value={scene.narration}
            placeholder="What the narrator says during this scene…"
            readOnly={readOnly}
            onChange={(event) => onChange({ narration: event.target.value })}
          />
        </div>

        <Field label="Action" hint="What happens on screen">
          <textarea
            className="input min-h-[2.5rem] resize-none"
            value={scene.action}
            placeholder="e.g. Mia jumps into a puddle"
            readOnly={readOnly}
            onChange={(event) => onChange({ action: event.target.value })}
          />
        </Field>

        <div className="grid gap-3 sm:grid-cols-2">
          {(
            [
              ["location", "Location", "sidewalk outside the house"],
              ["weather", "Weather / light", "light rain, soft grey light"],
              ["camera", "Camera", "medium shot, eye level"],
              ["motion", "Motion", "splash rises as she lands"],
            ] as const
          ).map(([key, label, placeholder]) => (
            <Field key={key} label={label}>
              <input
                className="input"
                value={scene[key]}
                placeholder={placeholder}
                readOnly={readOnly}
                onChange={(event) => onChange({ [key]: event.target.value })}
              />
            </Field>
          ))}
          {outfits.length > 1 && (
            <Field label="Outfit">
              <select
                className="input"
                value={scene.outfit_id}
                disabled={readOnly}
                onChange={(event) => onChange({ outfit_id: event.target.value })}
              >
                {outfits.map(([id, description]) => (
                  <option key={id} value={id}>
                    {id}: {description}
                  </option>
                ))}
              </select>
            </Field>
          )}
          {states.length > 1 && (
            <Field label="Character state">
              <select
                className="input"
                value={scene.state_id}
                disabled={readOnly}
                onChange={(event) => onChange({ state_id: event.target.value })}
              >
                {states.map(([id, description]) => (
                  <option key={id} value={id}>
                    {id}: {description}
                  </option>
                ))}
              </select>
            </Field>
          )}
        </div>

        <div className="divide-y divide-line rounded-xl border border-line">
          <Disclosure
            open={showSource}
            onToggle={() => setShowSource((value) => !value)}
            icon={<BookOpen className="size-3.5" />}
            title="Source text"
            meta={scene.keywords.length ? scene.keywords.slice(0, 3).join(" · ") : undefined}
          >
            {scene.text ? (
              <p className="font-display text-[15px] leading-relaxed text-ink-soft">{scene.text}</p>
            ) : (
              <p className="text-sm text-ink-faint">This scene was added by hand and has no source text.</p>
            )}
            {scene.keywords.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {scene.keywords.map((keyword) => (
                  <span key={keyword} className="chip">
                    {keyword}
                  </span>
                ))}
              </div>
            )}
          </Disclosure>
          <Disclosure
            open={showPrompt}
            onToggle={() => setShowPrompt((value) => !value)}
            icon={<Wand2 className="size-3.5" />}
            title="Visual prompt"
            meta={scene.custom_prompt ? "custom" : "auto"}
          >
            <textarea
              className="input min-h-[5rem] resize-none font-mono text-xs leading-relaxed"
              value={scene.visual_prompt ?? ""}
              readOnly={readOnly}
              placeholder="Generated from the fields above after saving. Type here to override."
              onChange={(event) =>
                onChange({ visual_prompt: event.target.value || null, custom_prompt: event.target.value.length > 0 })
              }
            />
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-ink-faint">
              <span>
                {scene.custom_prompt
                  ? "Hand-written: edits to the fields above won't change it."
                  : "Rebuilt from character, style and the fields above each time you save."}
              </span>
              {scene.custom_prompt && !readOnly && (
                <button
                  className="inline-flex items-center gap-1 font-medium text-accent-text hover:underline"
                  onClick={() => onChange({ custom_prompt: false })}
                >
                  <RotateCcw className="size-3" /> Use automatic prompt
                </button>
              )}
            </div>
          </Disclosure>
        </div>
      </div>
    </article>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="field-label">
        {label}
        {hint && <span className="normal-case tracking-normal font-normal">{hint}</span>}
      </span>
      {children}
    </label>
  );
}

function Disclosure({
  open,
  onToggle,
  icon,
  title,
  meta,
  children,
}: {
  open: boolean;
  onToggle: () => void;
  icon: ReactNode;
  title: string;
  meta?: string;
  children: ReactNode;
}) {
  return (
    <div>
      <button
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left text-xs font-medium text-ink-soft hover:text-ink"
        onClick={onToggle}
        aria-expanded={open}
      >
        {icon}
        {title}
        {meta && <span className="truncate font-normal text-ink-faint">· {meta}</span>}
        <ChevronDown className={clsx("ml-auto size-4 shrink-0 transition-transform", open && "rotate-180")} />
      </button>
      {open && <div className="px-3 pb-3">{children}</div>}
    </div>
  );
}
