# Project Status and GPU Resource Request

## What We Are Building

We are developing a multimodal educational-content pipeline that converts short
reading materials into concise, accessible videos. The current target workflow is:

```text
Reading material (PDF)
        ↓
Grounded LLM analysis
        ↓
Four-scene storyboard
        ↓
Global Character JSON + Global Style JSON
        ↓
One character reference image
        ↓
Four consistent scene keyframes
        ↓
2–4 second image-to-video animation per keyframe
        ↓
TTS + captions + transitions
        ↓
15-second vertical educational video
```

The architectural goal is to separate stable visual identity from scene-specific
content. Character appearance and visual style are defined once. Each scene describes
only the action, location, weather, camera, narration, and a reference to a globally
defined outfit state. The program then constructs every generation prompt from:

```text
GLOBAL_CHARACTER + GLOBAL_STYLE + SCENE_DESCRIPTION
```

This structure is intended to reduce character drift, style changes, and contradictory
scene prompts. It also makes intermediate decisions reviewable before expensive video
inference begins.

## Work Completed

We have implemented and tested the following components:

- A FastAPI backend with PDF upload, planning, job-status, video-listing, and media
  endpoints.
- Text extraction from English, text-based PDFs using Poppler with a bundled
  `pdfplumber` fallback, plus validation for file type, file size, and missing text.
- Source-grounded story parsing that retains sentence IDs and page references.
- A local Qwen3-4B-Instruct planning stage that runs on the GPU, produces a summary and
  storyboard, and releases GPU memory before video generation begins.
- Validation that assigns every source sentence to an ordered scene group so short
  actions and story endings are not silently omitted.
- A single-GPU scheduler abstraction with multiple CPU workers and one serialized GPU
  slot. Its interface is designed so that additional GPU IDs can be added later.
- Real local video inference with Wan2.1 T2V 1.3B, MP4 generation, ordered scene
  concatenation, and API playback.
- End-to-end tests using two different stories, *The Little Seed* and *After the Rain*.
- Regression tests for PDF extraction, source coverage, prompt construction, scheduling,
  job execution, and video assembly. The current feature branch passes 41 tests.

The experiments also identified the main limitation of independent text-to-video scene
generation: the story events may be correct while a recurring character's face, body,
clothes, or illustration style changes between clips. In *After the Rain*, for example,
the system generated the puddle jump, clearing sky, and rainbow, but Mia and her father
did not remain visually consistent across all scenes.

## Current Development

We have replaced the eight-scene independent text-to-video planner with a controlled
four-scene, 15-second planning contract. The current implementation includes:

- A typed `CharacterSpec` that stores one global identity and explicitly marks visual
  details inferred by the model rather than stated in the reading material.
- A typed `StyleSpec` shared by all four scenes.
- Named global outfit states for legitimate story changes, such as indoor clothing and
  a raincoat. Scenes reference an `outfit_id` instead of redescribing the character.
- Named lifecycle states for subjects that visibly transform, such as seed, young plant,
  and flowering plant. Scenes reference a `state_id`.
- A four-scene schema that separates action, location, weather, camera, and narration
  from character appearance.
- Programmatic construction of character-reference and keyframe prompts.
- Grounding checks that reject scene actions unsupported by the corresponding source
  section.

The next implementation steps are to generate one character reference, derive four
keyframes from that reference, animate each keyframe with an image-to-video model, and
assemble the results with TTS, captions, and transitions. Stable Video Diffusion is one
candidate for this stage because it is designed to condition short video generation on
an input image. We will benchmark it against other image-to-video options instead of
assuming one model will provide the best quality and latency.

## Why We Need at Least Two GPUs

One GPU is sufficient to prove that each component works, but it is not sufficient to
develop and evaluate the multi-GPU core engine described in our design specification.
Two GPUs are the minimum meaningful configuration for the following reasons.

### 1. The project explicitly requires multi-GPU scheduling

The design specification requires concurrent processing of script segments, dynamic
assignment to available GPUs, parallel clip generation, scalability to N GPUs, and GPU
utilization above 80%. A one-GPU experiment can measure a baseline, but it cannot test
GPU assignment, contention, load balancing, concurrent execution, or recovery when one
worker or device fails. At least two physical GPUs are required to compare the same
pipeline at N=1 and N=2.

### 2. The four scenes can be generated independently after planning

After the character reference and keyframes are ready, the four image-to-video jobs are
independent. With one GPU they must run sequentially. With two GPUs the scheduler can
generate two scenes concurrently and then dispatch the remaining two. This does not
guarantee a two-times speedup because model loading, decoding, storage, and assembly add
overhead, but it creates the parallelism needed to reduce end-to-end wall-clock time and
to measure the actual scaling efficiency.

### 3. Two GPUs enable representative end-to-end concurrency tests

The pipeline contains several inference workloads: LLM planning, reference and keyframe
generation, image-to-video generation, and potentially GPU-accelerated TTS or visual
quality evaluation. On one GPU these stages must be serialized or repeatedly unloaded
and reloaded. With two GPUs we can test a practical schedule, such as reserving one GPU
for the active video diffusion worker while the second processes another scene or runs
an evaluation task. This is necessary for measuring throughput, GPU utilization, queue
delay, average latency, and P95/P99 latency under realistic load.

### 4. Stronger models may exceed the memory of one development GPU

Our current workstation has one NVIDIA GeForce RTX 4060 Ti with 16 GB of VRAM. It can
run the 1.3B Wan prototype with CPU offload, but quality and speed are limited. Higher
quality image and video models have larger weights and much larger activation memory,
especially at higher resolution, with more frames or more inference steps. Two GPUs
allow us to evaluate model sharding or distributed inference when a model cannot fit on
one device. They also let us distinguish two optimization strategies: running separate
scenes on separate GPUs for throughput, and splitting one larger model across GPUs for
memory capacity. These strategies answer different research questions and both require
more than one GPU.

### 5. The 15-second goal requires measurement, not extrapolation

The design document targets less than 15 seconds of generation latency per clip, while
the product workflow targets a 15-second final video. These are separate requirements.
Our current Wan prototype takes tens of seconds for a single short scene and therefore
does not meet the latency target. We need controlled benchmarks on one and two GPUs to
measure whether parallel scene generation, mixed precision, model offload, fewer
inference steps, compilation, and a stronger GPU/model combination can close the gap.
Without at least two GPUs, any claim about multi-GPU speedup or scalability would be an
unsupported projection.

## Requested Development and Benchmark Environment

The minimum useful allocation is two CUDA-capable NVIDIA GPUs available to the same job
or node, with enough VRAM to run two scene workers concurrently. GPUs with at least
24 GB each would broaden the models we can test; 40–80 GB GPUs such as A100-class devices
would also allow experiments with stronger models and model parallelism. For valid
benchmark results, the two GPUs should be available concurrently and, during recorded
runs, should not be shared with unrelated workloads.

We also need:

- A supported NVIDIA driver and CUDA runtime supplied by the cluster administrator.
- A writable home, project, or scratch directory for Python environments, model weights,
  generated frames, videos, and benchmark logs.
- Sufficient storage for model caches; the current local Wan checkpoint occupies about
  27 GB, and additional image-to-video models will require more.
- Outbound HTTPS access to GitHub, Python package indexes, and approved model repositories,
  or an administrator-supported offline mirror.
- Permission to run long GPU jobs through the scheduler and request two GPUs in one job.
- Access to utilization and memory metrics through `nvidia-smi`, the scheduler, or the
  cluster's monitoring service.

## Questions for the School GPU/HPC Administrator

1. Are students expected to work without `sudo` on the GPU nodes? If so, which supported
   mechanism should we use to install project software in user space: Python virtual
   environments, Conda, environment modules, Apptainer/Singularity, or a prebuilt
   container image?
2. Are the NVIDIA driver and CUDA runtime managed centrally? Which CUDA and PyTorch
   versions are currently supported?
3. Can one scheduled job request at least two GPUs on the same node? What GPU models,
   VRAM capacities, time limits, quotas, and queue policies are available?
4. Can we obtain exclusive access to two GPUs for short, repeatable benchmark windows so
   that utilization, throughput, and latency results are not distorted by other users?
5. May users install Python packages and download model weights into a home or scratch
   directory without administrator privileges? What storage quota and retention policy
   apply to model caches and generated videos?
6. Do compute nodes allow outbound HTTPS access to GitHub, PyPI, and Hugging Face? If
   outbound access is blocked, is there an approved package/model mirror or transfer
   procedure?
7. Are AI-assisted development tools permitted? Specifically, may we use an AI coding
   agent through SSH or a remote IDE to inspect and modify our project workspace? Such a
   tool would need shell and Git access, write access only to our project directory, and
   outbound HTTPS access to its service.
8. If an external AI agent is not permitted on compute nodes, may it be used on a login
   or development node while GPU jobs are submitted separately through the scheduler?
9. Are Docker containers prohibited? If so, is Apptainer/Singularity the approved method
   for packaging reproducible CUDA, PyTorch, FFmpeg, and model dependencies?
10. Are background services such as FastAPI, Redis, or Celery allowed for development,
    or must all components run inside scheduled batch or interactive jobs?

## Short Message for the Administrator

**Subject: Questions About Two-GPU Access and User-Space Development Tools**

Hello,

Our team is developing a multimodal educational-content system that converts reading
materials into four short animated scenes and assembles them into a 15-second vertical
video. The project includes LLM-based grounded story planning, character-reference and
keyframe generation, image-to-video inference, TTS, captions, and a GPU task scheduler.

We can develop the functional pipeline on one GPU, but our approved core-engine design
requires segment-level parallelism, dynamic GPU assignment, and benchmarking at N=1 and
N=2. We therefore need at least two GPUs available concurrently. This will allow us to
run independent scene-generation jobs in parallel, measure utilization and latency, and
evaluate stronger models that may require more aggregate GPU memory.

Could you please clarify the following?

- Can a student job request two GPUs on the same node, and can they be reserved
  exclusively for short benchmarking windows?
- Since students may not have `sudo`, what is the approved way to install Python, CUDA,
  PyTorch, FFmpeg, and model dependencies: Conda/venv, environment modules, or
  Apptainer/Singularity?
- Are users allowed to download packages and model weights from GitHub, PyPI, and Hugging
  Face, or is an internal mirror required?
- Are AI coding agents or remote IDE assistants allowed to access a user's project
  workspace through SSH? If they are not allowed on compute nodes, may they be used on a
  login/development node while GPU jobs are submitted through the scheduler?
- What GPU models, VRAM, storage quotas, job time limits, and monitoring tools are
  available?

Thank you for helping us identify a supported and reproducible development workflow.

## Technical References

- Project design specification: `docs/design/Design_Spec_for_Multimodal_AI_Agent.pdf`
- [Hugging Face Diffusers: Stable Video Diffusion image-to-video pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/svd)
- [Hugging Face Diffusers: Wan video pipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/wan)
