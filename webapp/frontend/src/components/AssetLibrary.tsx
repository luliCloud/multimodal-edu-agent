"use client";

import { ImageIcon, Upload } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, apiUrl, type ProjectAsset } from "@/lib/api";
import { formatBytes } from "@/lib/hooks";

export function AssetLibrary({ projectId, disabled }: { projectId: string; disabled: boolean }) {
  const [assets, setAssets] = useState<ProjectAsset[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.assets(projectId).then(setAssets).catch((reason: Error) => setError(reason.message));
  }, [projectId]);

  const preview = useMemo(
    () => assets.find((asset) => asset.name === "character_reference.png")
      ?? assets.find((asset) => asset.name === "full_pose_sheet.png")
      ?? assets.find((asset) => /\.(png|jpe?g)$/i.test(asset.name)),
    [assets],
  );

  const upload = async () => {
    if (!files.length) return;
    setUploading(true);
    setError(null);
    try {
      setAssets(await api.uploadAssets(projectId, files));
      setFiles([]);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <section className="mb-5 overflow-hidden rounded-2xl border border-line bg-elevated">
      <div className="flex flex-col gap-4 p-5 sm:flex-row sm:items-start">
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element -- arbitrary user-uploaded asset URL
          <img
            src={apiUrl(preview.url)}
            alt="Character reference"
            className="h-28 w-24 shrink-0 rounded-xl border border-line bg-bg object-contain"
          />
        ) : (
          <div className="flex h-28 w-24 shrink-0 items-center justify-center rounded-xl border border-dashed border-line bg-bg text-ink-faint">
            <ImageIcon className="size-7" />
          </div>
        )}
        <div className="min-w-0 flex-1">
          <h2 className="font-display text-xl">Visual bible</h2>
          <p className="mt-1 text-sm text-ink-soft">
            These references are reused in every scene. Upload canonical files such as
            {" "}<code>character_reference.png</code>, <code>full_pose_sheet.png</code>, or
            {" "}<code>scene_01_keyframe.png</code>.
          </p>
          {assets.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {assets.map((asset) => (
                <span key={asset.name} className="rounded-full bg-bg px-2.5 py-1 text-xs text-ink-soft" title={formatBytes(asset.size_bytes)}>
                  {asset.name}
                </span>
              ))}
            </div>
          )}
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <label className="btn-ghost h-9 cursor-pointer">
              <Upload className="size-4" /> Choose assets
              <input
                className="sr-only"
                type="file"
                multiple
                accept=".png,.jpg,.jpeg,.json"
                disabled={disabled || uploading}
                onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
              />
            </label>
            {files.length > 0 && (
              <button className="btn-primary h-9" disabled={disabled || uploading} onClick={upload}>
                {uploading ? "Uploading…" : `Upload ${files.length} file${files.length === 1 ? "" : "s"}`}
              </button>
            )}
            <span className="text-xs text-ink-faint">
              {assets.length ? `${assets.length} assets ready` : "No visual references yet"}
            </span>
          </div>
          {error && <p className="mt-2 text-sm text-bad">{error}</p>}
        </div>
      </div>
    </section>
  );
}
