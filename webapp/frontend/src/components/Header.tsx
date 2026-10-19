"use client";

import clsx from "clsx";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Clapperboard, LayoutGrid } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type ServerInfo } from "@/lib/api";
import { Logo } from "./ui";

const NAV = [
  { href: "/", label: "Studio", icon: LayoutGrid },
  { href: "/library", label: "Library", icon: Clapperboard },
];

export function Header() {
  const pathname = usePathname();
  const [server, setServer] = useState<ServerInfo | null | undefined>(undefined);

  useEffect(() => {
    let alive = true;
    const check = () =>
      api
        .health()
        .then((info) => alive && setServer(info))
        .catch(() => alive && setServer(null));
    check();
    const timer = setInterval(check, 15000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <>
      <header className="sticky top-0 z-40 border-b border-line/70 bg-bg/75 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center gap-3 px-4 sm:px-6">
          <Link href="/" className="flex items-center gap-2.5">
            <Logo className="size-8" />
            <span className="font-display text-xl leading-none tracking-tight">
              EduTok <span className="text-ink-faint">Studio</span>
            </span>
          </Link>
          <nav className="ml-2 hidden items-center gap-1 sm:flex">
            {NAV.map(({ href, label }) => {
              const active = href === "/" ? pathname === "/" || pathname.startsWith("/projects") : pathname.startsWith(href);
              return (
                <Link
                  key={href}
                  href={href}
                  className={clsx(
                    "rounded-full px-3.5 py-1.5 text-sm transition-colors",
                    active ? "bg-ink text-bg" : "text-ink-soft hover:bg-sunken hover:text-ink",
                  )}
                >
                  {label}
                </Link>
              );
            })}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <ServerPill server={server} />
          </div>
        </div>
    </header>
      {/* Bottom tab bar on phones. Outside <header>: its backdrop-filter would trap position:fixed. */}
      <nav className="fixed inset-x-0 bottom-0 z-40 flex border-t border-line bg-elevated/90 pb-[env(safe-area-inset-bottom)] backdrop-blur-xl sm:hidden">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" || pathname.startsWith("/projects") : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex flex-1 flex-col items-center gap-0.5 py-2.5 text-[11px] font-medium",
                active ? "text-ink" : "text-ink-faint",
              )}
            >
              <Icon className="size-5" />
              {label}
            </Link>
          );
        })}
      </nav>
    </>
  );
}

function ServerPill({ server }: { server: ServerInfo | null | undefined }) {
  if (server === undefined) return null;
  const online = server !== null;
  return (
    <div
      title={online ? `Planner: ${server.planner_backend} · Renderer: ${server.render_backend} · Narration: ${server.tts_enabled ? "on" : "off"} · Storage: ${server.video_dir}` : "API server unreachable"}
      className="flex items-center gap-2 rounded-full border border-line bg-elevated px-3 py-1.5 text-xs text-ink-soft"
    >
      <span className={clsx("size-1.5 rounded-full", online ? "bg-ok" : "bg-bad")} />
      {online ? (
        <span className="hidden sm:inline">
          {server.planner_backend} planner · {server.render_backend} renderer
          {server.tts_enabled ? " · narration on" : ""}
        </span>
      ) : (
        <span>API offline</span>
      )}
      {online && <span className="sm:hidden">Online</span>}
    </div>
  );
}
