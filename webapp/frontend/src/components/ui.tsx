"use client";

import clsx from "clsx";
import { AlertTriangle, CheckCircle2, Info, Loader2, X } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { ProjectStatus } from "@/lib/api";

const STATUS: Record<ProjectStatus, { label: string; tone: string; busy?: boolean }> = {
  extracting: { label: "Reading PDF", tone: "bg-accent-soft text-accent-text", busy: true },
  planning: { label: "Planning scenes", tone: "bg-accent-soft text-accent-text", busy: true },
  draft: { label: "Ready to review", tone: "bg-warm-soft text-warm" },
  rendering: { label: "Rendering", tone: "bg-accent-soft text-accent-text", busy: true },
  rendered: { label: "Video ready", tone: "bg-ok-soft text-ok" },
  failed: { label: "Failed", tone: "bg-bad-soft text-bad" },
};

export function StatusBadge({ status, className }: { status: ProjectStatus; className?: string }) {
  const { label, tone, busy } = STATUS[status];
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
        tone,
        className,
      )}
    >
      <span className="relative flex size-1.5">
        {busy && <span className="pulse-dot absolute inset-0 rounded-full bg-current" />}
        <span className="relative size-1.5 rounded-full bg-current" />
      </span>
      {label}
    </span>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={clsx("animate-spin", className ?? "size-4")} aria-hidden />;
}

export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect width="32" height="32" rx="9" fill="var(--ink)" />
      <path d="M9 9.5h9.5a4.5 4.5 0 0 1 0 9H13" stroke="var(--bg)" strokeWidth="2.4" fill="none" strokeLinecap="round" />
      <path d="M13.5 15.2v8.3l6.8-4.15z" fill="var(--warm)" />
    </svg>
  );
}

// ------------------------------------------------------------------ toasts

type Toast = { id: number; tone: "ok" | "error" | "info"; message: string };
const ToastContext = createContext<(tone: Toast["tone"], message: string) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(0);
  const push = useCallback((tone: Toast["tone"], message: string) => {
    const id = ++nextId.current;
    setToasts((items) => [...items.slice(-3), { id, tone, message }]);
    setTimeout(() => setToasts((items) => items.filter((item) => item.id !== id)), 5000);
  }, []);
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div
        aria-live="polite"
        className="pointer-events-none fixed inset-x-0 bottom-[calc(env(safe-area-inset-bottom)+5.5rem)] z-50 flex flex-col items-center gap-2 px-4 sm:bottom-6 sm:items-end sm:pr-6"
      >
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className="rise pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-2xl border border-line bg-elevated px-4 py-3 text-sm shadow-float"
          >
            {toast.tone === "ok" ? (
              <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-ok" />
            ) : toast.tone === "error" ? (
              <AlertTriangle className="mt-0.5 size-4 shrink-0 text-bad" />
            ) : (
              <Info className="mt-0.5 size-4 shrink-0 text-accent" />
            )}
            <p className="flex-1 text-ink">{toast.message}</p>
            <button
              className="text-ink-faint hover:text-ink"
              aria-label="Dismiss"
              onClick={() => setToasts((items) => items.filter((item) => item.id !== toast.id))}
            >
              <X className="size-4" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export const useToast = () => useContext(ToastContext);

// ------------------------------------------------------------------ confirm dialog

export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel,
  danger,
  onConfirm,
  onClose,
}: {
  open: boolean;
  title: string;
  body: ReactNode;
  confirmLabel: string;
  danger?: boolean;
  onConfirm: () => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(event) => event.target === ref.current && onClose()}
      className="m-auto w-[min(28rem,calc(100vw-2rem))] rounded-3xl border border-line bg-elevated p-0 text-ink shadow-float backdrop:bg-black/40 backdrop:backdrop-blur-sm"
    >
      <div className="p-6">
        <h2 className="font-display text-2xl">{title}</h2>
        <div className="mt-2 text-sm leading-relaxed text-ink-soft">{body}</div>
        <div className="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <button className="btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button
            className={clsx(danger ? "btn bg-bad text-white hover:brightness-110" : "btn-primary")}
            onClick={() => {
              onConfirm();
              onClose();
            }}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </dialog>
  );
}

export function ErrorPanel({
  title,
  message,
  action,
}: {
  title: string;
  message: string;
  action?: ReactNode;
}) {
  return (
    <div className="card mx-auto max-w-xl p-8 text-center">
      <div className="mx-auto flex size-12 items-center justify-center rounded-2xl bg-bad-soft text-bad">
        <AlertTriangle className="size-5" />
      </div>
      <h2 className="mt-4 font-display text-2xl">{title}</h2>
      <p className="mt-2 text-sm leading-relaxed text-ink-soft">{message}</p>
      {action && <div className="mt-6 flex flex-wrap justify-center gap-2">{action}</div>}
    </div>
  );
}
