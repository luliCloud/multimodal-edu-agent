"use client";

import { ArrowRight, Clapperboard, FileText, PenLine, Sparkles, Trash2 } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { Dropzone } from "@/components/Dropzone";
import { ConfirmDialog, StatusBadge, useToast } from "@/components/ui";
import { api, apiUrl, BUSY, type ProjectSummary } from "@/lib/api";
import { timeAgo, usePolling } from "@/lib/hooks";

const STEPS = [
  { icon: FileText, title: "Drop", text: "Upload one story PDF. Text is extracted page by page." },
  { icon: Sparkles, title: "Draft", text: "The planner groups sentences into scenes with narration and shots." },
  { icon: PenLine, title: "Refine", text: "Edit narration, action, setting and camera, then reorder scenes." },
  { icon: Clapperboard, title: "Render", text: "Generate clips and a combined short, served from storage." },
];

export default function StudioPage() {
  const projects = usePolling(
    api.projects,
    (data) => (data?.some((project) => BUSY.includes(project.status)) ? 2000 : 15000),
  );
  const toast = useToast();
  const [pendingDelete, setPendingDelete] = useState<ProjectSummary | null>(null);

  const remove = async (project: ProjectSummary) => {
    try {
      await api.deleteProject(project.id);
      projects.setData((items) => items?.filter((item) => item.id !== project.id));
      toast("ok", `Deleted “${project.title}”`);
    } catch (error) {
      toast("error", (error as Error).message);
    }
  };

  return (
    <div className="space-y-16">
      <section className="grid items-center gap-10 lg:grid-cols-[1fr_1.1fr] lg:gap-14">
        <div className="rise">
          <p className="eyebrow">PDF → Script → Video</p>
          <h1 className="mt-4 font-display text-[2.75rem] leading-[1.02] tracking-tight sm:text-6xl lg:text-7xl">
            Turn a story into a <em className="text-accent-text">short</em>, scene by scene.
          </h1>
          <p className="mt-5 max-w-lg text-base leading-relaxed text-ink-soft sm:text-lg">
            Review every scene before any GPU time is spent. Edit the narration and shot
            directions, then render when the script reads right.
          </p>
          <ol className="mt-8 grid grid-cols-2 gap-x-6 gap-y-5">
            {STEPS.map(({ icon: Icon, title, text }, index) => (
              <li key={title} className="flex gap-3">
                <span className="flex size-8 shrink-0 items-center justify-center rounded-xl border border-line bg-elevated text-ink-soft">
                  <Icon className="size-4" />
                </span>
                <div>
                  <p className="text-sm font-medium">
                    <span className="mr-1.5 text-ink-faint">0{index + 1}</span>
                    {title}
                  </p>
                  <p className="mt-0.5 text-xs leading-relaxed text-ink-faint">{text}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
        <div className="rise order-first [animation-delay:120ms] lg:order-none">
          <Dropzone />
        </div>
      </section>

      <section>
        <div className="mb-5 flex items-end justify-between gap-4">
          <div>
            <p className="eyebrow">Your work</p>
            <h2 className="mt-1 font-display text-3xl">Recent projects</h2>
          </div>
          {projects.data && projects.data.length > 0 && (
            <span className="text-sm text-ink-faint">{projects.data.length} total</span>
          )}
        </div>

        {projects.loading && !projects.data ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[0, 1, 2].map((key) => (
              <div key={key} className="skeleton h-44 rounded-2xl" />
            ))}
          </div>
        ) : projects.error && !projects.data ? (
          <div className="card p-6 text-sm text-ink-soft">{projects.error.message}</div>
        ) : projects.data?.length ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {projects.data.map((project, index) => (
              <ProjectCard
                key={project.id}
                project={project}
                index={index}
                onDelete={() => setPendingDelete(project)}
              />
            ))}
          </div>
        ) : (
          <div className="card flex flex-col items-center px-6 py-14 text-center">
            <FileText className="size-6 text-ink-faint" />
            <p className="mt-3 font-medium">No projects yet</p>
            <p className="mt-1 text-sm text-ink-faint">Your first upload will show up here.</p>
          </div>
        )}
      </section>

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Delete project?"
        body={
          <>
            “{pendingDelete?.title}”, its script and all rendered videos will be removed from
            storage. This can&apos;t be undone.
          </>
        }
        confirmLabel="Delete"
        danger
        onConfirm={() => pendingDelete && remove(pendingDelete)}
        onClose={() => setPendingDelete(null)}
      />
    </div>
  );
}

function ProjectCard({
  project,
  index,
  onDelete,
}: {
  project: ProjectSummary;
  index: number;
  onDelete: () => void;
}) {
  return (
    <div
      className="rise group card relative flex flex-col overflow-hidden transition-all duration-200 hover:-translate-y-0.5 hover:shadow-float"
      style={{ animationDelay: `${Math.min(index, 8) * 50}ms` }}
    >
      <Link
        href={`/projects/${project.id}`}
        className="absolute inset-0 z-0"
        aria-label={`Open ${project.title}`}
      />
      <div className="relative aspect-[16/7] overflow-hidden border-b border-line bg-sunken">
        {project.final_video_url ? (
          <video
            src={`${apiUrl(project.final_video_url)}#t=0.5`}
            muted
            playsInline
            preload="metadata"
            className="pointer-events-none size-full object-cover"
          />
        ) : (
          <div className="flex size-full items-center justify-center">
            <div className="flex gap-1.5">
              {Array.from({ length: Math.max(project.scene_count, 4) }).map((_, i) => (
                <span
                  key={i}
                  className={`h-10 w-6 rounded-md border border-line ${
                    project.scene_count ? "bg-elevated" : "skeleton"
                  }`}
                />
              ))}
            </div>
          </div>
        )}
      </div>
      <div className="relative z-10 flex flex-1 flex-col p-4 [pointer-events:none]">
        <div className="flex items-start justify-between gap-3">
          <h3 className="line-clamp-2 font-medium leading-snug">{project.title}</h3>
          <button
            onClick={onDelete}
            className="icon-btn pointer-events-auto -mr-2 -mt-1.5 size-8 opacity-100 sm:opacity-0 sm:group-hover:opacity-100"
            aria-label={`Delete ${project.title}`}
          >
            <Trash2 className="size-4" />
          </button>
        </div>
        <p className="mt-1 truncate text-xs text-ink-faint">{project.filename}</p>
        <div className="mt-auto flex items-center justify-between gap-2 pt-4">
          <StatusBadge status={project.status} />
          <span className="flex items-center gap-1 text-xs text-ink-faint">
            {project.scene_count ? `${project.scene_count} scenes · ` : ""}
            {timeAgo(project.updated_at)}
            <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" />
          </span>
        </div>
      </div>
    </div>
  );
}
