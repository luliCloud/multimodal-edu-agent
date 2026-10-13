"use client";

import clsx from "clsx";
import { ArrowUpRight, Download, Film, FolderOpen, Play, Search, X } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, apiUrl, type LibraryVideo, type ServerInfo } from "@/lib/api";
import { formatBytes, timeAgo, usePolling } from "@/lib/hooks";

const FILTERS = [
  { id: "final", label: "Final videos" },
  { id: "scene", label: "Scene clips" },
  { id: "external", label: "Imported" },
  { id: "all", label: "All" },
] as const;
type Filter = (typeof FILTERS)[number]["id"];

export default function LibraryPage() {
  const library = usePolling(api.library, () => 20000);
  const [server, setServer] = useState<ServerInfo | null>(null);
  const [filter, setFilter] = useState<Filter>("final");
  const [query, setQuery] = useState("");
  const [playing, setPlaying] = useState<LibraryVideo | null>(null);

  useEffect(() => {
    api.health().then(setServer).catch(() => setServer(null));
  }, []);

  const counts = useMemo(() => {
    const items = library.data ?? [];
    return {
      all: items.length,
      final: items.filter((item) => item.kind === "final").length,
      scene: items.filter((item) => item.kind === "scene").length,
      external: items.filter((item) => item.kind === "external").length,
    };
  }, [library.data]);

  const visible = (library.data ?? []).filter((item) => {
    if (filter !== "all" && item.kind !== filter) return false;
    const haystack = `${item.project_title ?? ""} ${item.name}`.toLowerCase();
    return haystack.includes(query.trim().toLowerCase());
  });

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="eyebrow">Library</p>
          <h1 className="mt-1 font-display text-4xl tracking-tight sm:text-5xl">Rendered videos</h1>
          {server && (
            <p className="mt-2 text-sm text-ink-faint">
              <FolderOpen className="mr-1.5 inline size-4 -translate-y-px" />
              Loaded from <code className="break-all rounded bg-sunken px-1.5 py-0.5 font-mono text-xs">{server.video_dir}</code>
            </p>
          )}
        </div>
        <label className="relative w-full sm:w-72">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-ink-faint" />
          <input
            className="input rounded-full pl-9"
            placeholder="Search videos"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
      </div>

      <div className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-1 [scrollbar-width:none] sm:mx-0 sm:px-0">
        {FILTERS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setFilter(id)}
            className={clsx(
              "flex shrink-0 items-center gap-2 rounded-full border px-3.5 py-1.5 text-sm transition-colors",
              filter === id ? "border-ink bg-ink text-bg" : "border-line bg-elevated text-ink-soft hover:text-ink",
            )}
          >
            {label}
            <span className={clsx("text-xs tabular-nums", filter === id ? "opacity-70" : "text-ink-faint")}>{counts[id]}</span>
          </button>
        ))}
      </div>

      {library.loading && !library.data ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {Array.from({ length: 8 }).map((_, index) => (
            <div key={index} className="skeleton aspect-video rounded-2xl" />
          ))}
        </div>
      ) : library.error && !library.data ? (
        <div className="card p-6 text-sm text-ink-soft">{library.error.message}</div>
      ) : visible.length === 0 ? (
        <div className="card flex flex-col items-center px-6 py-16 text-center">
          <Film className="size-6 text-ink-faint" />
          <p className="mt-3 font-medium">{counts.all ? "No videos match" : "No videos yet"}</p>
          <p className="mt-1 max-w-md text-sm text-ink-faint">
            {counts.all
              ? "Try another filter or search term."
              : "Rendered videos appear here. Any .mp4 copied into the video directory shows up under Imported."}
          </p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {visible.map((video, index) => (
            <VideoCard key={video.key} video={video} index={index} onPlay={() => setPlaying(video)} />
          ))}
        </div>
      )}

      <PlayerModal video={playing} onClose={() => setPlaying(null)} />
    </div>
  );
}

function VideoCard({ video, index, onPlay }: { video: LibraryVideo; index: number; onPlay: () => void }) {
  const title = video.project_title ?? video.name.replace(/\.[^.]+$/, "");
  const scene = video.kind === "scene" ? video.name.match(/scene-(\d+)/)?.[1] : null;
  return (
    <div className="rise group card overflow-hidden" style={{ animationDelay: `${Math.min(index, 12) * 40}ms` }}>
      <button onClick={onPlay} className="relative block aspect-video w-full overflow-hidden bg-sunken" aria-label={`Play ${title}`}>
        <video
          src={`${apiUrl(video.url)}#t=0.5`}
          muted
          playsInline
          preload="metadata"
          className="pointer-events-none size-full object-cover transition-transform duration-500 group-hover:scale-[1.03]"
        />
        <span className="absolute inset-0 flex items-center justify-center bg-gradient-to-t from-black/40 via-transparent to-transparent opacity-90">
          <span className="flex size-11 items-center justify-center rounded-full bg-white/95 text-black shadow-lg transition-transform group-hover:scale-110">
            <Play className="size-4 translate-x-px fill-current" />
          </span>
        </span>
        {scene && (
          <span className="absolute left-2 top-2 rounded-md bg-black/60 px-1.5 py-0.5 text-[10px] font-semibold text-white backdrop-blur">
            Scene {Number(scene)}
          </span>
        )}
      </button>
      <div className="flex items-start gap-2 p-3.5">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{title}</p>
          <p className="truncate text-xs text-ink-faint">
            {formatBytes(video.size_bytes)} · {timeAgo(video.modified_at)}
          </p>
        </div>
        {video.project_id && (
          <Link href={`/projects/${video.project_id}`} className="icon-btn size-8" aria-label="Open project">
            <ArrowUpRight className="size-4" />
          </Link>
        )}
        <a href={apiUrl(`${video.url}?download=true`)} className="icon-btn size-8" aria-label="Download">
          <Download className="size-4" />
        </a>
      </div>
    </div>
  );
}

function PlayerModal({ video, onClose }: { video: LibraryVideo | null; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (video && !dialog.open) dialog.showModal();
    if (!video && dialog.open) dialog.close();
  }, [video]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(event) => event.target === ref.current && onClose()}
      className="m-auto w-[min(64rem,calc(100vw-1.5rem))] overflow-hidden rounded-3xl border border-line bg-[#0b0a0f] p-0 text-white shadow-float backdrop:bg-black/70 backdrop:backdrop-blur-sm"
    >
      {video && (
        <>
          <div className="flex items-center gap-3 px-4 py-3">
            <p className="min-w-0 flex-1 truncate text-sm font-medium">{video.project_title ?? video.name}</p>
            <button onClick={onClose} className="flex size-8 items-center justify-center rounded-full hover:bg-white/10" aria-label="Close">
              <X className="size-4" />
            </button>
          </div>
          <video key={video.url} src={apiUrl(video.url)} controls autoPlay playsInline className="max-h-[75dvh] w-full bg-black object-contain" />
        </>
      )}
    </dialog>
  );
}
