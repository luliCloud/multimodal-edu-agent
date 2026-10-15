"use client";

import clsx from "clsx";
import { AlertTriangle, Check, Download, Film, PenLine, Play, RefreshCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { apiUrl, type Project, type ProjectVideo } from "@/lib/api";
import { formatBytes } from "@/lib/hooks";
import { EventLog } from "./Pipeline";
import { SCENE_HUES } from "./SceneCard";
import { Spinner } from "./ui";

interface Props {
  project: Project;
  onEditScript: () => void;
  onRerender: () => void;
  rerendering: boolean;
}

export function VideoStage({ project, onEditScript, onRerender, rerendering }: Props) {
  if (project.status === "rendering") return <RenderProgress project={project} />;
  const final = project.videos.find((video) => video.kind === "final");
  const scenes = project.videos.filter((video) => video.kind === "scene");
  const stale =
    project.rendered_revision !== null && project.script_revision > project.rendered_revision;

  if (!final)
    return (
      <div className="card flex flex-col items-center px-6 py-16 text-center">
        <Film className="size-6 text-ink-faint" />
        <p className="mt-3 font-medium">No video yet</p>
        <p className="mt-1 max-w-sm text-sm text-ink-faint">
          Review the script, then choose Generate video. Clips are written to the server&apos;s
          video directory and streamed from there.
        </p>
        <button className="btn-ghost mt-5" onClick={onEditScript}>
          <PenLine className="size-4" /> Go to script
        </button>
      </div>
    );

  return (
    <div className="space-y-6">
      {stale && (
        <div className="flex flex-col gap-3 rounded-2xl border border-warn/30 bg-warn-soft px-4 py-3 text-sm text-warn sm:flex-row sm:items-center">
          <AlertTriangle className="size-4 shrink-0" />
          <p className="flex-1">The script has changed since this video was rendered.</p>
          <button className="btn h-8 bg-warn px-3 text-white hover:brightness-110" onClick={onRerender} disabled={rerendering}>
            {rerendering ? <Spinner className="size-3.5" /> : <RefreshCw className="size-3.5" />}
            Re-render
          </button>
        </div>
      )}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="card overflow-hidden">
          <div className="flex justify-center bg-[#0b0a0f]">
            <video
              key={final.url}
              src={apiUrl(final.url)}
              controls
              playsInline
              preload="metadata"
              className="max-h-[70dvh] w-full object-contain"
            />
          </div>
          <div className="flex flex-wrap items-center gap-3 p-4 sm:px-5">
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium">{project.title}</p>
              <p className="text-xs text-ink-faint">
                {[
                  project.render?.backend && `${project.render.backend} renderer`,
                  final.duration_seconds && `${final.duration_seconds}s`,
                  final.size_bytes && formatBytes(final.size_bytes),
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
            </div>
            <button className="btn-ghost h-9" onClick={onEditScript}>
              <PenLine className="size-4" /> Edit script
            </button>
            <a className="btn-dark h-9" href={apiUrl(`${final.url}?download=true`)}>
              <Download className="size-4" /> Download
            </a>
          </div>
        </div>
        <div className="space-y-6">
          {scenes.length > 0 && (
            <div className="card p-4 sm:p-5">
              <p className="mb-3 text-sm font-medium">Scene clips</p>
              <div className="grid grid-cols-2 gap-3">
                {scenes.map((video) => (
                  <ClipThumb key={video.key} video={video} />
                ))}
              </div>
            </div>
          )}
          <EventLog events={project.events.slice(-30)} />
        </div>
      </div>
    </div>
  );
}

function ClipThumb({ video }: { video: ProjectVideo }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const hue = SCENE_HUES[((video.scene ?? 1) - 1) % SCENE_HUES.length];
  const toggle = () => {
    const element = ref.current;
    if (!element) return;
    if (element.paused) element.play().catch(() => {});
    else element.pause();
  };
  return (
    <button
      onClick={toggle}
      onMouseEnter={() => ref.current?.play().catch(() => {})}
      onMouseLeave={() => {
        ref.current?.pause();
      }}
      className="group relative aspect-video overflow-hidden rounded-xl border border-line bg-sunken text-left"
    >
      <video
        ref={ref}
        src={`${apiUrl(video.url)}#t=0.3`}
        muted
        loop
        playsInline
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        className="size-full object-cover"
      />
      <span
        className="absolute left-1.5 top-1.5 rounded-md px-1.5 py-0.5 text-[10px] font-semibold text-white"
        style={{ background: hue }}
      >
        Scene {video.scene}
      </span>
      {!playing && (
        <span className="absolute inset-0 flex items-center justify-center bg-black/10 opacity-100 transition-opacity group-hover:opacity-0">
          <span className="flex size-8 items-center justify-center rounded-full bg-white/90 text-black shadow">
            <Play className="size-3.5 translate-x-px fill-current" />
          </span>
        </span>
      )}
    </button>
  );
}

function RenderProgress({ project }: { project: Project }) {
  const render = project.render;
  const progress = render?.progress ?? 0;
  const scenes = render?.scenes ?? [];
  const done = Math.floor(progress * scenes.length + 1e-6);
  const elapsed = useElapsed(render?.started_at);
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
      <div className="card flex flex-col items-center px-6 py-12 text-center sm:py-16">
        <ProgressRing value={progress} />
        <h2 className="mt-6 font-display text-3xl">Rendering your video</h2>
        <p className="mt-2 max-w-md text-sm text-ink-soft">{project.stage_message}</p>
        <p className="mt-1 text-xs tabular-nums text-ink-faint">
          {render?.backend} renderer · {elapsed}
          {render?.backend === "wan" && " · real model inference can take several minutes"}
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-2">
          {scenes.map((number, index) => {
            const state = index < done ? "done" : index === done ? "active" : "waiting";
            return (
              <span
                key={number}
                className={clsx(
                  "flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors",
                  state === "done" && "border-transparent bg-ok-soft text-ok",
                  state === "active" && "border-accent/40 bg-accent-soft text-accent-text",
                  state === "waiting" && "border-line text-ink-faint",
                )}
              >
                {state === "done" ? (
                  <Check className="size-3" strokeWidth={3} />
                ) : state === "active" ? (
                  <Spinner className="size-3" />
                ) : (
                  <span className="size-1.5 rounded-full bg-current opacity-50" />
                )}
                Scene {number}
              </span>
            );
          })}
          {scenes.length > 1 && (
            <span
              className={clsx(
                "flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium",
                done >= scenes.length ? "border-accent/40 bg-accent-soft text-accent-text" : "border-line text-ink-faint",
              )}
            >
              {done >= scenes.length ? <Spinner className="size-3" /> : <Film className="size-3" />}
              Combine
            </span>
          )}
        </div>
      </div>
      <EventLog events={project.events.slice(-40)} />
    </div>
  );
}

function ProgressRing({ value }: { value: number }) {
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  return (
    <div className="relative size-36">
      <svg viewBox="0 0 120 120" className="size-full -rotate-90">
        <circle cx="60" cy="60" r={radius} fill="none" stroke="var(--line)" strokeWidth="8" />
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - Math.max(0.03, value))}
          className="transition-[stroke-dashoffset] duration-700 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-display text-4xl tabular-nums">{Math.round(value * 100)}%</span>
      </div>
    </div>
  );
}

function useElapsed(since?: string) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  if (!since) return "";
  const seconds = Math.max(0, Math.floor((now - new Date(since).getTime()) / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")} elapsed`;
}
