"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Fetches with `load`, then re-polls every `interval(data)` ms (null = stop polling).
 * Polling pauses while the tab is hidden and resumes immediately when it is shown.
 */
export function usePolling<T>(
  load: () => Promise<T>,
  interval: (data: T | undefined) => number | null,
  deps: unknown[] = [],
) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<Error>();
  const [loading, setLoading] = useState(true);
  const loadRef = useRef(load);
  const intervalRef = useRef(interval);
  loadRef.current = load;
  intervalRef.current = interval;
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const generation = useRef(0);

  const refresh = useCallback(async () => {
    const current = ++generation.current;
    clearTimeout(timer.current);
    let next: T | undefined;
    try {
      next = await loadRef.current();
      if (current !== generation.current) return;
      setData(next);
      setError(undefined);
    } catch (err) {
      if (current !== generation.current) return;
      setError(err as Error);
    } finally {
      if (current === generation.current) {
        setLoading(false);
        const wait = intervalRef.current(next);
        // Back off while the server is unreachable instead of hammering it.
        const delay = next === undefined ? 5000 : wait;
        if (delay !== null && document.visibilityState === "visible") {
          timer.current = setTimeout(refresh, delay);
        }
      }
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    refresh();
    const onVisible = () => document.visibilityState === "visible" && refresh();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      generation.current++;
      clearTimeout(timer.current);
      document.removeEventListener("visibilitychange", onVisible);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, setData, error, loading, refresh };
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function timeAgo(iso: string) {
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 45) return "just now";
  const minutes = seconds / 60;
  if (minutes < 60) return `${Math.round(minutes)} min ago`;
  const hours = minutes / 60;
  if (hours < 24) return `${Math.round(hours)} h ago`;
  const days = hours / 24;
  if (days < 7) return `${Math.round(days)} d ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function formatClock(iso: string) {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}
