"use client";

import clsx from "clsx";
import { ArrowUpRight, FileText, UploadCloud } from "lucide-react";
import { useRouter } from "next/navigation";
import { useRef, useState, type DragEvent } from "react";
import { uploadPdf } from "@/lib/api";
import { formatBytes } from "@/lib/hooks";
import { useToast } from "./ui";

const MAX_BYTES = 10 * 1024 * 1024;

export function Dropzone() {
  const router = useRouter();
  const toast = useToast();
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const depth = useRef(0);

  const accept = async (files: FileList | null) => {
    if (!files || files.length === 0 || file) return;
    if (files.length > 1) {
      toast("info", "One PDF at a time. Drop a single file to start.");
      return;
    }
    const picked = files[0];
    if (picked.type !== "application/pdf" && !picked.name.toLowerCase().endsWith(".pdf")) {
      toast("error", "That doesn't look like a PDF.");
      return;
    }
    if (picked.size > MAX_BYTES) {
      toast("error", `PDFs are limited to 10 MB (this one is ${formatBytes(picked.size)}).`);
      return;
    }
    setFile(picked);
    setProgress(0);
    try {
      const project = await uploadPdf(picked, setProgress);
      router.push(`/projects/${project.id}`);
    } catch (error) {
      toast("error", (error as Error).message);
      setFile(null);
    } finally {
      if (input.current) input.current.value = "";
    }
  };

  const onDrag = (event: DragEvent, delta: number) => {
    event.preventDefault();
    depth.current = Math.max(0, depth.current + delta);
    setDragging(depth.current > 0);
  };

  return (
    <div
      onDragEnter={(event) => onDrag(event, 1)}
      onDragLeave={(event) => onDrag(event, -1)}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        depth.current = 0;
        setDragging(false);
        accept(event.dataTransfer.files);
      }}
      className={clsx(
        "group relative overflow-hidden rounded-[28px] border-2 border-dashed p-1.5 transition-all duration-300",
        dragging
          ? "scale-[1.01] border-accent bg-accent-soft/60"
          : "border-line-strong/80 bg-elevated/60 hover:border-accent/50",
      )}
    >
      <input
        ref={input}
        type="file"
        accept="application/pdf,.pdf"
        className="sr-only"
        onChange={(event) => accept(event.target.files)}
        aria-label="Choose a PDF"
      />
      <div className="relative flex min-h-[260px] flex-col items-center justify-center rounded-[22px] px-6 py-10 text-center sm:min-h-[300px]">
        {file ? (
          <div className="rise flex w-full max-w-sm flex-col items-center">
            <div className="relative flex h-20 w-16 items-center justify-center rounded-lg border border-line bg-elevated shadow-card">
              <FileText className="size-7 text-accent" />
              <div className="absolute inset-x-1 top-1 h-0.5 rounded bg-accent/70 scanline" />
            </div>
            <p className="mt-5 max-w-full truncate font-medium">{file.name}</p>
            <p className="mt-1 text-sm text-ink-faint">
              {progress < 1 ? `Uploading · ${Math.round(progress * 100)}%` : "Starting conversion…"}
            </p>
            <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-sunken">
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-300"
                style={{ width: `${Math.max(4, progress * 100)}%` }}
              />
            </div>
          </div>
        ) : (
          <>
            <div
              className={clsx(
                "flex size-16 items-center justify-center rounded-2xl bg-ink text-bg shadow-float transition-transform duration-300",
                dragging ? "-translate-y-1 rotate-[-6deg] scale-110" : "group-hover:-translate-y-0.5",
              )}
            >
              <UploadCloud className="size-7" />
            </div>
            <h2 className="mt-6 font-display text-3xl sm:text-4xl">
              {dragging ? "Release to upload" : "Drop a story PDF"}
            </h2>
            <p className="mt-2 max-w-md text-sm leading-relaxed text-ink-soft">
              We&apos;ll read it, split it into scenes, and draft narration and shot
              directions you can edit before rendering.
            </p>
            <button className="btn-primary mt-6" onClick={() => input.current?.click()}>
              Choose a file
              <ArrowUpRight className="size-4" />
            </button>
            <p className="mt-4 text-xs text-ink-faint">One text-based PDF at a time · up to 10 MB</p>
          </>
        )}
      </div>
    </div>
  );
}
