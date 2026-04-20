# Llama - Local AI Platform

A self-hosted AI platform combining chat and image generation through a unified ChatGPT-style web interface, with full system monitoring.

## Architecture

<div align="center">
<table>
<tr>
<td colspan="4" align="center">
<b>Web Browser</b><br/>
<code>https://&lt;server-ip&gt;</code>
</td>
</tr>
<tr><td colspan="4" align="center">&#11167; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &#11167;</td></tr>
<tr>
<td colspan="2" align="center">
<b>Unified Chat (Flask)</b><br/>
Port 443 (HTTPS)<br/><br/>
&#8226; Chat routing<br/>
&#8226; Image routing<br/>
&#8226; Music composition<br/>
&#8226; Model management<br/>
&#8226; Auto-tuning<br/>
&#8226; Parameter control<br/>
&#8226; Perf logging
</td>
<td colspan="2" align="center">
<b>Grafana</b><br/>
Port 3000<br/><br/>
System + Model Metrics<br/><br/>
&#11167;<br/>
<b>InfluxDB</b> :8086<br/>
1-day retention<br/>
&#11165;<br/>
<b>Telegraf</b><br/>
CPU, Mem, Disk, Net<br/>
Service process stats
</td>
</tr>
<tr><td colspan="2" align="center">&#11167; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &#11167;</td><td colspan="2"></td></tr>
<tr>
<td align="center">
<b>Ollama</b><br/>
:11434<br/>
Chat + Music
</td>
<td align="center">
<b>SD WebUI</b><br/>
:7860<br/>
Images<br/>
CPU-only, 3 cores
</td>
<td align="center">
<b>LilyPond + Timidity</b><br/>
Sheet Music &rarr; MP3
</td>
</tr>
</table>
</div>

## Quick Install

```bash
git clone https://github.com/ystanbl/llama.git
cd llama
sudo bash install.sh
```

This single command:
- Creates a `llama` service user
- Clones the repo to `/home/llama/llama`
- Installs Ollama and pulls the default model
- Installs Stable Diffusion WebUI (CPU-only) with shared model storage at `/opt/sd-models`
- Sets up the Unified Chat web interface with SSL
- Installs music tools (LilyPond, Timidity, ffmpeg)
- Deploys Grafana + InfluxDB + Telegraf monitoring
- Configures all systemd services and sudoers

### Prerequisites
- Ubuntu 22.04+
- Python 3.10+
- systemd

## Deploying Updates

After pushing changes to the repo:
```bash
sudo bash /home/llama/llama/install.sh --update
```

This pulls the latest code, re-installs service files, and restarts the app.

Or from the status page at `https://<server-ip>/status`, click **Restart Service** for unified-chat.

## Components

### Unified Chat (`unified-chat/`)
Flask web application providing a ChatGPT-style interface that automatically routes requests:
- **Natural language detection** — "draw me a cat" goes to Stable Diffusion, everything else goes to Ollama
- **Self-correcting routing** — say "that was for chat" to correct misrouted messages; the system learns and remembers via localStorage
- **Streaming chat** via SSE with any Ollama model
- **Image generation** with auto-optimized parameters for CPU
- **Request queuing** — concurrent image requests queue up instead of canceling each other
- **Music composition** — "compose a fugue in the style of Bach" generates sheet music and MP3 via Ollama, LilyPond, Timidity, and ffmpeg
- **Non-blocking concurrency** — chat while images or music generate, separate abort controllers
- **Model management** — search, browse, pull/download, switch, and delete models for both Ollama and Stable Diffusion (HuggingFace integration)
- **Auto-tuning** — benchmarks samplers, step counts, resolutions, and thread counts to find optimal SD settings for the hardware
- **Parameter tweaking** — expose all model parameters (temperature, top_p, top_k, context length, steps, CFG scale, sampler, seed, etc.) with per-model awareness
- **Performance logging** — writes generation time, tokens/sec, and other metrics to InfluxDB for Grafana dashboards
- **Live image preview** — watch images form in real time as each sampling step completes
- **Generation controls** — cancel or skip remaining steps during image generation
- **Prompt history** — press up/down arrow to cycle through previous prompts in the current chat
- **Admin status page** (`/status`) — real-time view of all services, active tasks, queue, with cancel and restart controls

### Stable Diffusion WebUI
AUTOMATIC1111's stable-diffusion-webui configured for CPU-only operation:
- Pinned to **3 CPU cores** (0-2) via `CPUAffinity` to leave remaining cores available for other services
- Thread counts locked to 3 (OMP, MKL, OpenBLAS, NUMEXPR, PyTorch)
- Runs with `--use-cpu all --no-half --api` flags
- Models stored in `/opt/sd-models` (shared, user-agnostic)

### Ollama
Local LLM inference server:
- Runs as system user `ollama`
- Default model: `mistral:latest`
- Models manageable through the web interface or `ollama` CLI

### Monitoring Stack
- **Grafana** (port 3000) — anonymous access enabled, no login required
  - System panels: CPU per core, CPU total gauge, memory gauge, memory breakdown, disk, network, load average, swap, disk I/O
  - Service panels: per-service CPU and memory (RSS) for ollama, sd-webui, unified-chat
  - **Model performance panels**: image generation time by model, chat response time by model, tokens/sec by model, average comparisons, steps vs time analysis
- **InfluxDB** — 1-day data retention (`autogen` policy)
- **Telegraf** — 10s collection interval, procstat monitoring for all three services
- Monitoring stack runs at **lowest CPU priority** (`nice 19`, `CPUSchedulingPolicy=idle`)

## URLs

| Service | URL |
|---------|-----|
| Chat Interface | https://&lt;server-ip&gt; (self-signed cert) |
| Task Status | https://&lt;server-ip&gt;/status |
| Grafana Dashboard | http://&lt;server-ip&gt;:3000 |
| Ollama API | http://&lt;server-ip&gt;:11434 |
| SD WebUI API | http://&lt;server-ip&gt;:7860 |

## API Endpoints

### Chat & Image
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Stream chat response (SSE). Body: `{model, messages, options}` |
| POST | `/api/image` | Generate image (queued). Body: `{prompt, negative_prompt, steps, width, height, cfg_scale, sampler_name, seed}` |
| GET | `/api/image/progress` | Poll generation progress (step, %, ETA, live preview, queue position) |
| POST | `/api/image/cancel` | Interrupt and discard current generation |
| POST | `/api/image/skip` | Skip remaining steps, return partial result |

### Music
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/music` | Generate sheet music + MP3. Body: `{prompt, model}` |
| GET | `/api/music/progress` | Poll generation phase (composing/engraving/synthesizing/encoding) |

### Model Management — Ollama
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List installed model names |
| GET | `/api/ollama/models` | List installed models with details |
| POST | `/api/ollama/pull` | Pull a model (SSE progress). Body: `{name}` |
| POST | `/api/ollama/delete` | Delete a model. Body: `{name}` |
| GET | `/api/ollama/search?q=` | Search Ollama library |
| POST | `/api/ollama/show` | Get model info and default params. Body: `{name}` |

### Model Management — Stable Diffusion
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/sd/models` | List installed SD models |
| POST | `/api/sd/switch` | Switch active model. Body: `{title}` |
| POST | `/api/sd/delete` | Delete a model file. Body: `{filename}` |
| GET | `/api/sd/search?q=` | Search HuggingFace for compatible checkpoint models |
| POST | `/api/sd/browse` | List files in a HuggingFace repo with variant labels. Body: `{repo_id}` |
| POST | `/api/sd/download` | Download a model. Body: `{url, filename}` |
| GET | `/api/sd/download` | Poll download progress |
| GET | `/api/sd/samplers` | List available samplers |

### Admin & Status
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/status` | Admin status page (HTML) |
| GET | `/api/status` | Service status, active tasks, and queue info (JSON) |
| POST | `/api/status/cancel/sd` | Cancel running SD generation and clear queue |
| POST | `/api/status/cancel/ollama` | Cancel running Ollama tasks (unloads models) |
| POST | `/api/status/queue/clear` | Clear queued SD requests (active generation continues) |
| POST | `/api/status/restart/sd` | Restart SD WebUI systemd service |
| POST | `/api/status/restart/ollama` | Restart Ollama systemd service |
| POST | `/api/status/restart/unified-chat` | Restart unified-chat (self-restart) |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/resources` | System resources, SD params, tuning state |
| POST | `/api/tune` | Start auto-tuning |
| GET | `/api/tune` | Poll tuning progress |
| DELETE | `/api/tune` | Reset tuned params to defaults |

## File Structure

```
.
├── README.md
├── install.sh              # One-step installer (also supports --update)
├── unified-chat/
│   ├── app.py              # Flask backend
│   ├── requirements.txt    # Python dependencies
│   └── templates/
│       ├── index.html      # ChatGPT-style frontend
│       └── status.html     # Admin status page
└── config/
    ├── systemd/
    │   ├── unified-chat.service
    │   ├── ollama.service
    │   └── sd-webui.service
    ├── telegraf/
    │   └── telegraf.conf
    └── grafana/
        └── dashboard.json
```

## Installed Paths

| Path | Contents |
|------|----------|
| `/home/llama/llama/` | Git repo (this project) |
| `/home/llama/llama/unified-chat/` | Chat app (run directly from repo) |
| `/home/llama/stable-diffusion-webui/` | AUTOMATIC1111 SD WebUI |
| `/opt/sd-models/` | Shared SD model storage |

## CPU Allocation

| Cores | Service |
|-------|---------|
| 0, 1, 2 | Stable Diffusion (pinned via CPUAffinity) |
| 3+ | Ollama, Unified Chat, system tasks |
| All (idle priority) | Grafana, InfluxDB, Telegraf |

## Logging

All services log to **syslog** as the single source of truth:
```bash
# View unified-chat logs
journalctl -u unified-chat -f

# View all AI service logs
journalctl -t unified-chat -t ollama -t sd-webui -f
```
