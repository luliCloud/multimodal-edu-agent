// Typed client for the FastAPI wrapper (webapp/backend/app/models.py).

export type ProjectStatus =
  | "extracting"
  | "planning"
  | "draft"
  | "rendering"
  | "rendered"
  | "failed";

export const BUSY: ProjectStatus[] = ["extracting", "planning", "rendering"];

export interface Scene {
  scene: number;
  action: string;
  location: string;
  weather: string;
  camera: string;
  motion: string;
  narration: string;
  narration_seconds: number;
  outfit_id: string;
  state_id: string;
  text: string;
  pages: number[];
  keywords: string[];
  source_sentence_ids: number[];
  visual_prompt: string | null;
  keyframe_prompt: string | null;
  custom_prompt: boolean;
}

export interface Character {
  name: string;
  kind: string;
  visual_identity: string;
  age?: number | string | null;
  hair?: string | null;
  eyes?: string | null;
  clothes?: string | null;
  outfits?: Record<string, string>;
  states?: Record<string, string>;
  inferred_fields?: string[];
  [key: string]: unknown;
}

export interface Script {
  title: string;
  summary: string;
  character: Character | null;
  style: Record<string, string> | null;
  scenes: Scene[];
  source_sentences: { id: number; page: number; text: string }[];
  duration_seconds: number;
  planner_backend: string;
}

export interface Issue {
  level: "warning" | "info";
  message: string;
  scene: number | null;
}

export interface ProjectEvent {
  at: string;
  message: string;
  level: "info" | "error";
}

export interface ProjectVideo {
  key: string;
  kind: "final" | "scene";
  scene: number | null;
  url: string;
  duration_seconds: number | null;
  size_bytes: number | null;
}

export interface RenderState {
  render_id: string;
  backend: string;
  scenes: number[];
  progress: number;
  started_at: string;
  finished_at: string | null;
  error: string | null;
}

export interface Project {
  id: string;
  title: string;
  filename: string;
  size_bytes: number;
  status: ProjectStatus;
  stage_message: string;
  error: string | null;
  created_at: string;
  updated_at: string;
  source: {
    page_count: number;
    sentence_count: number;
    keywords: { keyword: string; occurrences: number; pages: number[] }[];
    excerpt: string;
  };
  script: Script | null;
  script_revision: number;
  rendered_revision: number | null;
  issues: Issue[];
  render: RenderState | null;
  videos: ProjectVideo[];
  events: ProjectEvent[];
}

export interface ProjectSummary {
  id: string;
  title: string;
  filename: string;
  status: ProjectStatus;
  stage_message: string;
  scene_count: number;
  created_at: string;
  updated_at: string;
  final_video_url: string | null;
}

export interface LibraryVideo {
  key: string;
  url: string;
  name: string;
  size_bytes: number;
  modified_at: string;
  project_id: string | null;
  project_title: string | null;
  kind: "final" | "scene" | "external";
}

export interface ServerInfo {
  status: string;
  planner_backend: string;
  render_backend: string;
  video_dir: string;
  storage: string;
}

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");

/** Absolute URL for an API path or a media URL returned by the API. */
export const apiUrl = (path: string) => `${BASE}${path}`;

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), { cache: "no-store", ...init });
  } catch {
    throw new ApiError("Can't reach the API server. Is FastAPI running on port 8000?", 0);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    if (response.status >= 500 && response.status !== 502 && !detail) {
      detail = "The API server returned an error";
    }
    throw new ApiError(detail || `Request failed (${response.status})`, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<ServerInfo>("/api/health"),
  projects: () => request<ProjectSummary[]>("/api/projects"),
  project: (id: string) => request<Project>(`/api/projects/${id}`),
  deleteProject: (id: string) => request<void>(`/api/projects/${id}`, { method: "DELETE" }),
  saveScript: (id: string, script: Script, baseRevision: number) =>
    request<Project>(`/api/projects/${id}/script`, {
      method: "PUT",
      ...json({ script, base_revision: baseRevision }),
    }),
  replan: (id: string) => request<Project>(`/api/projects/${id}/replan`, { method: "POST" }),
  render: (id: string, scenes?: number[]) =>
    request<Project>(`/api/projects/${id}/render`, {
      method: "POST",
      ...json({ scenes: scenes ?? null }),
    }),
  library: () => request<LibraryVideo[]>("/api/library"),
};

/** Upload with progress events (fetch has no upload progress). */
export function uploadPdf(file: File, onProgress: (fraction: number) => void): Promise<Project> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl("/api/projects"));
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as Project);
      else
        reject(
          new ApiError(
            (xhr.response && xhr.response.detail) || `Upload failed (${xhr.status})`,
            xhr.status,
          ),
        );
    };
    xhr.onerror = () =>
      reject(new ApiError("Can't reach the API server. Is FastAPI running on port 8000?", 0));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export function newScene(number: number): Scene {
  return {
    scene: number,
    action: "",
    location: "",
    weather: "",
    camera: "",
    motion: "",
    narration: "",
    narration_seconds: 3,
    outfit_id: "default",
    state_id: "default",
    text: "",
    pages: [],
    keywords: [],
    source_sentence_ids: [],
    visual_prompt: null,
    keyframe_prompt: null,
    custom_prompt: false,
  };
}

/** Mirrors storyboard_schema.narration_seconds (150 wpm, 3s minimum). */
export const narrationSeconds = (text: string) =>
  Math.round(Math.max(3, text.trim().split(/\s+/).filter(Boolean).length / 2.5) * 10) / 10;

export const wordCount = (text: string) => text.trim().split(/\s+/).filter(Boolean).length;
