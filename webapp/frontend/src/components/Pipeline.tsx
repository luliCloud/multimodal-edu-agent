"use client";

import clsx from "clsx";
import { Check, FileSearch, ScrollText, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Project, ProjectEvent } from "@/lib/api";
import { formatClock } from "@/lib/hooks";

const STEPS = ["Upload", "Extract", "Plan", "Review", "Render", "Watch"] as const;

function stepIndex(project: Project) {
  switch (project.status) {
    case "extracting":
      return 1;
    case "planning":
      return 2;
    case "draft":
      return 3;
    case "rendering":
      return 4;
    case "rendered":
      return 5;
    case "failed":
      return project.source.page_count ? 2 : 1;
  }
}

export function Stepper({ project }: { project: Project }) {
  const active = stepIndex(project);
  const failed = project.status === "failed";
  const finished = project.status === "rendered";
  return (
    <ol className="flex items-center gap-1 overflow-x-auto pb-1 [scrollbar-width:none]" aria-label="Progress">
      {STEPS.map((label, index) => {
        const done = index < active || (finished && index === active);
        const current = index === active && !finished;
        return (
          <li key={label} className="flex shrink-0 items-center gap-1">
            <span
              className={clsx(
                "flex items-center gap-1.5 rounded-full py-1 pl-1 text-xs font-medium transition-colors",
                current ? "pr-3" : "pr-1 sm:pr-3",
                done && "text-ink",
                current && !failed && "bg-accent-soft text-accent-text",
                current && failed && "bg-bad-soft text-bad",
                !done && !current && "text-ink-faint",
              )}
              aria-current={current ? "step" : undefined}
            >
              <span
                className={clsx(
                  "flex size-5 items-center justify-center rounded-full text-[10px]",
                  done ? "bg-ink text-bg" : current ? "bg-current/15" : "border border-line",
                )}
              >
                {done ? (
                  <Check className="size-3" strokeWidth={3} />
                ) : current && !failed && index !== 3 ? (
                  <span className="pulse-dot size-1.5 rounded-full bg-current" />
                ) : (
                  index + 1
                )}
              </span>
              <span className={clsx(!current && "hidden sm:inline")}>{label}</span>
            </span>
            {index < STEPS.length - 1 && (
              <span className={clsx("h-px w-4 sm:w-6", index < active ? "bg-ink/40" : "bg-line")} />
            )}
          </li>
        );
      })}
    </ol>
  );
}

/** Live view while the PDF is being read and the planner is drafting scenes. */
export function ProcessingView({ project }: { project: Project }) {
  const extracting = project.status === "extracting";
  const { source } = project;
  return (
    <div className="grid gap-6 lg:grid-cols-[1.15fr_1fr]">
      <div className="card overflow-hidden">
        <div className="flex items-center gap-3 border-b border-line px-5 py-4">
          <span className="flex size-9 items-center justify-center rounded-xl bg-accent-soft text-accent-text">
            {extracting ? <FileSearch className="size-4" /> : <Sparkles className="size-4" />}
          </span>
          <div className="min-w-0">
            <p className="font-medium">{extracting ? "Reading your PDF" : "Drafting the storyboard"}</p>
            <p className="truncate text-sm text-ink-faint">{project.stage_message}</p>
          </div>
        </div>

        <div className="space-y-5 p-5">
          <div className="grid grid-cols-3 gap-3">
            <Stat label="Pages" value={source.page_count || null} />
            <Stat label="Sentences" value={source.sentence_count || null} />
            <Stat label="Topics" value={source.keywords.length || null} />
          </div>

          <div>
            <p className="field-label">Key topics</p>
            {source.keywords.length ? (
              <div className="flex flex-wrap gap-1.5">
                {source.keywords.map((item, index) => (
                  <span
                    key={item.keyword}
                    className="chip rise"
                    style={{ animationDelay: `${index * 60}ms` }}
                  >
                    {item.keyword}
                    <span className="text-ink-faint">p.{item.pages.join(",")}</span>
                  </span>
                ))}
              </div>
            ) : (
              <div className="flex gap-1.5">
                {[64, 88, 52, 76, 60].map((width) => (
                  <span key={width} className="skeleton h-6 rounded-full" style={{ width }} />
                ))}
              </div>
            )}
          </div>

          <div>
            <p className="field-label">Extracted text</p>
            {source.excerpt ? (
              <Typewriter text={source.excerpt} />
            ) : (
              <div className="space-y-2">
                {[100, 94, 97, 70].map((width) => (
                  <div key={width} className="skeleton h-3 rounded" style={{ width: `${width}%` }} />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="space-y-6">
        <div className="card p-5">
          <div className="mb-4 flex items-center gap-2">
            <ScrollText className="size-4 text-ink-faint" />
            <p className="text-sm font-medium">Scenes</p>
            <span className="ml-auto text-xs text-ink-faint">
              {extracting ? "waiting for text" : "being written"}
            </span>
          </div>
          <div className="space-y-3">
            {[0, 1, 2, 3].map((index) => (
              <div
                key={index}
                className="flex gap-3 rounded-xl border border-line p-3"
                style={{ opacity: 1 - index * 0.18 }}
              >
                <div className="skeleton h-14 w-10 shrink-0 rounded-lg" />
                <div className="flex-1 space-y-2 pt-1">
                  <div className="skeleton h-3 w-1/3 rounded" />
                  <div className="skeleton h-3 w-11/12 rounded" />
                  <div className="skeleton h-3 w-2/3 rounded" />
                </div>
              </div>
            ))}
          </div>
        </div>
        <EventLog events={project.events} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="rounded-xl border border-line bg-bg/50 px-3 py-2.5">
      <p className="text-[11px] uppercase tracking-wider text-ink-faint">{label}</p>
      {value === null ? (
        <div className="skeleton mt-1.5 h-5 w-8 rounded" />
      ) : (
        <p className="rise mt-0.5 font-display text-2xl leading-tight">{value}</p>
      )}
    </div>
  );
}

function Typewriter({ text }: { text: string }) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setShown(text.length);
      return;
    }
    setShown(0);
    const timer = setInterval(() => {
      setShown((count) => {
        if (count >= text.length) {
          clearInterval(timer);
          return count;
        }
        return count + 6;
      });
    }, 16);
    return () => clearInterval(timer);
  }, [text]);
  return (
    <p className="max-h-56 overflow-y-auto rounded-xl bg-sunken/70 p-4 font-display text-[17px] leading-relaxed text-ink-soft">
      {text.slice(0, shown)}
      {shown < text.length && <span className="ml-0.5 inline-block h-4 w-0.5 translate-y-0.5 animate-pulse bg-accent" />}
    </p>
  );
}

export function EventLog({ events, className }: { events: ProjectEvent[]; className?: string }) {
  const list = useRef<HTMLOListElement>(null);
  useEffect(() => {
    list.current?.scrollTo({ top: list.current.scrollHeight, behavior: "smooth" });
  }, [events.length]);
  return (
    <div className={clsx("card p-5", className)}>
      <p className="mb-3 text-sm font-medium">Activity</p>
      <ol ref={list} className="max-h-64 space-y-2.5 overflow-y-auto pr-1">
        {events.length === 0 && <li className="text-sm text-ink-faint">Nothing yet.</li>}
        {events.map((event, index) => (
          <li key={`${event.at}-${index}`} className="flex gap-3 text-sm">
            <span
              className={clsx(
                "mt-1.5 size-1.5 shrink-0 rounded-full",
                event.level === "error" ? "bg-bad" : index === events.length - 1 ? "bg-accent" : "bg-line-strong",
              )}
            />
            <span className={clsx("flex-1 break-words", event.level === "error" ? "text-bad" : "text-ink-soft")}>
              {event.message}
            </span>
            <time className="shrink-0 text-xs tabular-nums text-ink-faint">{formatClock(event.at)}</time>
          </li>
        ))}
      </ol>
    </div>
  );
}
