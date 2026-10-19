"use client";

import clsx from "clsx";
import {
  ArrowLeft,
  FileDown,
  FileText,
  Film,
  MoreHorizontal,
  PenLine,
  RefreshCw,
  Trash2,
} from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ProcessingView, Stepper } from "@/components/Pipeline";
import { AssetLibrary } from "@/components/AssetLibrary";
import { ScriptEditor } from "@/components/ScriptEditor";
import { ConfirmDialog, ErrorPanel, Spinner, StatusBadge, useToast } from "@/components/ui";
import { VideoStage } from "@/components/VideoStage";
import { api, apiUrl, ApiError, BUSY, type Project } from "@/lib/api";
import { formatBytes, timeAgo, usePolling } from "@/lib/hooks";

type Tab = "script" | "video";

export default function ProjectPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const toast = useToast();
  const { data: project, setData, error, refresh } = usePolling(
    () => api.project(id),
    (data) => (data && BUSY.includes(data.status) ? 1200 : null),
    [id],
  );
  const [tab, setTab] = useState<Tab>("script");
  const [confirm, setConfirm] = useState<"delete" | "replan" | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [rerendering, setRerendering] = useState(false);
  const previousStatus = useRef<string | undefined>(undefined);

  // Follow the pipeline: jump to the video tab when a render starts or finishes.
  useEffect(() => {
    if (!project) return;
    const before = previousStatus.current;
    previousStatus.current = project.status;
    if (project.status === before) return;
    if (project.status === "rendering" || (project.status === "rendered" && (before === undefined || before === "rendering"))) {
      setTab("video");
    } else if (project.status === "draft") {
      setTab("script");
    }
    if (before === "rendering" && project.status === "rendered") toast("ok", "Your video is ready.");
    if (before === "rendering" && project.status === "draft" && project.error) toast("error", project.error);
    if (before === "planning" && project.status === "draft") toast("ok", "Storyboard ready. Review and edit the scenes.");
  }, [project, toast]);

  const onProjectChange = useCallback(
    (next: Project) => {
      setData(next);
      if (BUSY.includes(next.status)) refresh();
    },
    [setData, refresh],
  );

  if (error && !project) {
    const missing = error instanceof ApiError && error.status === 404;
    return (
      <ErrorPanel
        title={missing ? "Project not found" : "Couldn't load project"}
        message={missing ? "It may have been deleted." : error.message}
        action={
          <Link href="/" className="btn-ghost">
            <ArrowLeft className="size-4" /> Back to studio
          </Link>
        }
      />
    );
  }
  if (!project) return <PageSkeleton />;

  const replan = async () => {
    try {
      onProjectChange(await api.replan(project.id));
    } catch (err) {
      toast("error", (err as Error).message);
    }
  };
  const remove = async () => {
    try {
      await api.deleteProject(project.id);
      router.push("/");
    } catch (err) {
      toast("error", (err as Error).message);
    }
  };
  const rerender = async () => {
    setRerendering(true);
    try {
      onProjectChange(await api.render(project.id, project.render?.scenes));
    } catch (err) {
      toast("error", (err as Error).message);
    } finally {
      setRerendering(false);
    }
  };

  const busy = BUSY.includes(project.status);
  const hasScript = project.script !== null && !["extracting", "planning"].includes(project.status);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4">
        <Link href="/" className="inline-flex w-fit items-center gap-1.5 text-sm text-ink-faint hover:text-ink">
          <ArrowLeft className="size-4" /> All projects
        </Link>
        <div className="flex flex-wrap items-start gap-x-4 gap-y-3">
          <div className="min-w-0 flex-1 basis-72">
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge status={project.status} />
              <span className="text-xs text-ink-faint">
                {project.filename} · {formatBytes(project.size_bytes)} · updated {timeAgo(project.updated_at)}
              </span>
            </div>
            <h1 className="mt-2 font-display text-4xl leading-tight tracking-tight sm:text-5xl">
              {project.title}
            </h1>
          </div>
          <div className="relative flex items-center gap-2">
            <a className="btn-ghost h-9" href={apiUrl(`/api/projects/${project.id}/pdf`)} target="_blank" rel="noreferrer">
              <FileText className="size-4" /> <span className="hidden sm:inline">View PDF</span>
            </a>
            <button className="icon-btn border border-line bg-elevated" onClick={() => setMenuOpen((open) => !open)} aria-label="More actions" aria-expanded={menuOpen}>
              <MoreHorizontal className="size-4" />
            </button>
            {menuOpen && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setMenuOpen(false)} />
                <div className="rise absolute right-0 top-11 z-50 w-56 overflow-hidden rounded-2xl border border-line bg-elevated p-1.5 shadow-float">
                  {project.script && (
                    <MenuItem
                      icon={<FileDown className="size-4" />}
                      label="Export script.json"
                      href={apiUrl(`/api/projects/${project.id}/script.json`)}
                      onClick={() => setMenuOpen(false)}
                    />
                  )}
                  <MenuItem
                    icon={<RefreshCw className="size-4" />}
                    label="Re-run conversion"
                    disabled={busy}
                    onClick={() => {
                      setMenuOpen(false);
                      setConfirm("replan");
                    }}
                  />
                  <MenuItem
                    icon={<Trash2 className="size-4" />}
                    label="Delete project"
                    danger
                    onClick={() => {
                      setMenuOpen(false);
                      setConfirm("delete");
                    }}
                  />
                </div>
              </>
            )}
          </div>
        </div>
        <Stepper project={project} />
      </div>

      {project.status === "failed" ? (
        <ErrorPanel
          title="Conversion failed"
          message={project.error ?? "Something went wrong while reading the PDF."}
          action={
            <>
              <button className="btn-primary" onClick={replan}>
                <RefreshCw className="size-4" /> Try again
              </button>
              <Link href="/" className="btn-ghost">
                Upload a different PDF
              </Link>
            </>
          }
        />
      ) : !hasScript ? (
        <ProcessingView project={project} />
      ) : (
        <>
          {project.error && project.status !== "rendering" && (
            <div className="flex items-start gap-3 rounded-2xl border border-bad/30 bg-bad-soft px-4 py-3 text-sm text-bad">
              <span className="font-medium">Last render failed:</span>
              <span className="flex-1 break-words">{project.error}</span>
            </div>
          )}
          <div className="flex w-fit gap-1 rounded-full border border-line bg-elevated p-1" role="tablist">
            <TabButton active={tab === "script"} onClick={() => setTab("script")} icon={<PenLine className="size-4" />}>
              Script
            </TabButton>
            <TabButton
              active={tab === "video"}
              onClick={() => setTab("video")}
              icon={project.status === "rendering" ? <Spinner /> : <Film className="size-4" />}
            >
              Video
              {project.videos.some((video) => video.kind === "final") && (
                <span className="size-1.5 rounded-full bg-ok" />
              )}
            </TabButton>
          </div>
          {/* The editor stays mounted while hidden so unsaved edits survive tab switches. */}
          <div hidden={tab !== "script"}>
            <AssetLibrary projectId={project.id} disabled={project.status === "rendering"} />
            <ScriptEditor
              key={project.id}
              project={project}
              readOnly={project.status === "rendering"}
              onProjectChange={onProjectChange}
            />
          </div>
          {tab === "video" && (
            <VideoStage
              project={project}
              onEditScript={() => setTab("script")}
              onRerender={rerender}
              rerendering={rerendering}
            />
          )}
          {tab === "script" && project.status === "rendering" && (
            <p className="text-center text-sm text-ink-faint">Editing is paused while the video renders.</p>
          )}
        </>
      )}

      <ConfirmDialog
        open={confirm === "replan"}
        title="Re-run conversion?"
        body="The PDF will be read and planned again. Your current script edits will be replaced by a fresh draft."
        confirmLabel="Re-run"
        onConfirm={replan}
        onClose={() => setConfirm(null)}
      />
      <ConfirmDialog
        open={confirm === "delete"}
        title="Delete project?"
        body="The PDF, script and every rendered video for this project will be removed from storage."
        confirmLabel="Delete"
        danger
        onConfirm={remove}
        onClose={() => setConfirm(null)}
      />
    </div>
  );
}

function TabButton({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <button
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={clsx(
        "flex items-center gap-2 rounded-full px-4 py-1.5 text-sm font-medium transition-colors",
        active ? "bg-ink text-bg" : "text-ink-soft hover:text-ink",
      )}
    >
      {icon}
      {children}
    </button>
  );
}

function MenuItem({
  icon,
  label,
  onClick,
  href,
  danger,
  disabled,
}: {
  icon: React.ReactNode;
  label: string;
  onClick?: () => void;
  href?: string;
  danger?: boolean;
  disabled?: boolean;
}) {
  const className = clsx(
    "flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-sm transition-colors disabled:opacity-40",
    danger ? "text-bad hover:bg-bad-soft" : "text-ink hover:bg-sunken",
  );
  return href ? (
    <a className={className} href={href} onClick={onClick}>
      {icon}
      {label}
    </a>
  ) : (
    <button className={className} onClick={onClick} disabled={disabled}>
      {icon}
      {label}
    </button>
  );
}

function PageSkeleton() {
  return (
    <div className="space-y-6">
      <div className="skeleton h-4 w-24 rounded" />
      <div className="skeleton h-12 w-2/3 rounded-xl" />
      <div className="skeleton h-7 w-full max-w-xl rounded-full" />
      <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
        <div className="skeleton h-72 rounded-2xl" />
        <div className="space-y-4">
          <div className="skeleton h-24 rounded-2xl" />
          <div className="skeleton h-64 rounded-2xl" />
        </div>
      </div>
    </div>
  );
}
