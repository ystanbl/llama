import base64
import glob as globmod
import json
import logging
import logging.handlers
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import requests
from flask import Flask, render_template, request, Response, stream_with_context, jsonify

app = Flask(__name__)

# Configure syslog logging
syslog_handler = logging.handlers.SysLogHandler(
    address="/dev/log",
    facility=logging.handlers.SysLogHandler.LOG_LOCAL0,
)
syslog_handler.setFormatter(logging.Formatter("unified-chat: %(levelname)s %(message)s"))
app.logger.addHandler(syslog_handler)
app.logger.setLevel(logging.INFO)
logging.getLogger().addHandler(syslog_handler)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
SD_URL = os.environ.get("SD_URL", "http://localhost:7860")
TUNE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned_params.json")

# Global tuning state
tuning_state = {"running": False, "progress": "", "results": None}

# Active task tracking
active_tasks = {
    "sd": {"active": False, "prompt": "", "started": None, "model": ""},
    "ollama": {"active": False, "prompt": "", "started": None, "model": "", "count": 0},
    "music": {
        "active": False, "prompt": "", "started": None,
        "model": "", "phase": "", "cancel": False,
    },
    "code": {"active": False, "prompt": "", "started": None, "phase": ""},
}

# Request queues — serialise access to single-threaded backends
sd_lock = threading.Lock()
sd_queue_prompts = []  # prompts of requests waiting for the lock

# Music generation working directory
MUSIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "music_output")
os.makedirs(MUSIC_DIR, exist_ok=True)

# InfluxDB for performance metrics
INFLUXDB_URL = os.environ.get("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_DB = "telegraf"

# ── Code workspace management ──────────────────────────────────────────────

WORKSPACES_DIR = os.environ.get(
    "WORKSPACES_DIR", "/home/llama/workspaces"
)
os.makedirs(WORKSPACES_DIR, exist_ok=True)

MAX_FILE_SIZE = 100 * 1024  # 100KB per file for context injection
MAX_WORKSPACE_FILES = 5000  # cap file listing for very large repos
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB upload

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".wav", ".ogg", ".mp4", ".avi", ".mov", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".msi",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pyc", ".pyo", ".class", ".o", ".obj", ".wasm",
    ".woff", ".woff2", ".ttf", ".eot",
    ".sqlite", ".db", ".lock", ".jar", ".war",
    ".min.js", ".min.css", ".map",
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".env", ".idea", ".vscode", "dist", "build",
    ".next", ".nuxt", "target", "vendor", ".tox", ".mypy_cache",
    ".pytest_cache", "coverage", ".coverage", "htmlcov",
    ".eggs", "*.egg-info", ".sass-cache", "bower_components",
}

# Clone progress tracking
clone_state = {"active": False, "progress": "", "name": ""}


PREFS_FILE = ".llama-prefs.json"


def load_workspace_prefs(ws_path):
    """Load learned preferences for a workspace."""
    prefs_path = os.path.join(ws_path, PREFS_FILE)
    try:
        with open(prefs_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"excluded": [], "boosted": []}


def save_workspace_prefs(ws_path, prefs):
    """Save learned preferences for a workspace."""
    prefs_path = os.path.join(ws_path, PREFS_FILE)
    with open(prefs_path, "w") as f:
        json.dump(prefs, f, indent=2)


def sanitize_workspace_name(name):
    """Strip to safe chars for directory name."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "-", name.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned[:80] or "workspace"


def build_file_tree(workspace_path):
    """Walk workspace, skip ignored dirs/extensions, return file list + tree."""
    files = []
    for root, dirs, filenames in os.walk(workspace_path):
        # Filter out skip dirs in-place
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not d.endswith(".egg-info")
        ]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SKIP_EXTENSIONS:
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, workspace_path).replace("\\", "/")
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            files.append({"path": rel, "size": size, "ext": ext})
            if len(files) >= MAX_WORKSPACE_FILES:
                break
        if len(files) >= MAX_WORKSPACE_FILES:
            break

    # Build indented tree string for system prompt
    tree_lines = []
    prev_parts = []
    for f in sorted(files, key=lambda x: x["path"]):
        parts = f["path"].split("/")
        # Add directory headers for new directories
        for i, part in enumerate(parts[:-1]):
            if i >= len(prev_parts) or prev_parts[i] != part:
                tree_lines.append("  " * i + part + "/")
        tree_lines.append("  " * (len(parts) - 1) + parts[-1])
        prev_parts = parts

    tree_str = "\n".join(tree_lines)
    if len(files) >= MAX_WORKSPACE_FILES:
        tree_str += f"\n... (truncated at {MAX_WORKSPACE_FILES} files)"
    return files, tree_str


def estimate_tokens(text):
    """Rough token estimate: ~4 chars per token for code."""
    return len(text) // 4


def extract_relevant_sections(lines, keywords, max_lines):
    """Extract sections of a file most relevant to keywords.

    Instead of sending the first N lines, find functions/blocks
    containing keywords and return those with context.
    """
    # Score each line by keyword hits
    scored = []
    for i, line in enumerate(lines):
        line_lower = line.lower()
        hits = sum(1 for kw in keywords if kw in line_lower)
        scored.append((i, hits))

    # Find clusters of relevant lines (expand to surrounding context)
    relevant_ranges = []
    context_lines = 5  # lines before/after each hit
    for i, hits in scored:
        if hits > 0:
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            relevant_ranges.append((start, end))

    if not relevant_ranges:
        # No keyword hits — return the top of the file
        result = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            result += f"\n[...showing first {max_lines}"
            result += f" of {len(lines)} lines...]"
        return result

    # Merge overlapping ranges
    relevant_ranges.sort()
    merged = [relevant_ranges[0]]
    for start, end in relevant_ranges[1:]:
        if start <= merged[-1][1] + 2:  # merge if gap <= 2 lines
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Also always include the first ~10 lines (imports/header)
    header_end = min(10, len(lines))
    sections = []
    total = 0
    # Add header if not already covered
    if merged[0][0] > header_end:
        sections.append(
            f"[Lines 1-{header_end}]\n"
            + "\n".join(lines[:header_end])
        )
        total += header_end

    for start, end in merged:
        chunk_size = end - start
        if total + chunk_size > max_lines:
            end = start + (max_lines - total)
            if end <= start:
                break
        sections.append(
            f"\n[Lines {start + 1}-{end}]\n"
            + "\n".join(lines[start:end])
        )
        total += end - start
        if total >= max_lines:
            break

    result = "\n".join(sections)
    if total < len(lines):
        result += (
            f"\n[...showing {total} of {len(lines)} lines"
            " (relevant sections)...]"
        )
    return result


def find_cross_references(content, workspace_files):
    """Scan file content for references to other workspace files.

    Looks for imports, render_template, require, open(), url_for,
    and string literals matching workspace filenames.
    Returns list of referenced file paths found in the workspace.
    """
    basenames = {}
    for f in workspace_files:
        bn = os.path.basename(f["path"]).lower()
        basenames.setdefault(bn, []).append(f["path"])

    referenced = set()

    # Python imports: import foo, from foo import bar, from foo.bar import baz
    for m in re.finditer(
        r"(?:from|import)\s+([\w.]+)", content
    ):
        mod = m.group(1).replace(".", "/")
        for ext in (".py", ".pyx", ""):
            candidate = (mod + ext).lower()
            for f in workspace_files:
                if f["path"].lower() == candidate or \
                   f["path"].lower().endswith("/" + candidate):
                    referenced.add(f["path"])

    # String literals that match workspace filenames
    # Catches render_template("x.html"), open("x.cfg"), url_for, etc.
    for m in re.finditer(r"""['"]([^'"]{2,80})['"]""", content):
        literal = m.group(1).strip().replace("\\", "/")
        lit_lower = literal.lower()
        lit_base = os.path.basename(lit_lower)

        # Direct basename match
        if lit_base in basenames:
            for path in basenames[lit_base]:
                referenced.add(path)
            continue

        # Path match (e.g. "templates/index.html")
        for f in workspace_files:
            if f["path"].lower() == lit_lower or \
               f["path"].lower().endswith("/" + lit_lower):
                referenced.add(f["path"])
                break

    # JS/TS imports: require("./foo"), import x from "./foo"
    for m in re.finditer(
        r"""(?:require|from)\s*\(\s*['"]([^'"]+)['"]\s*\)"""
        r"""|from\s+['"]([^'"]+)['"]""",
        content,
    ):
        ref = m.group(1) or m.group(2)
        ref = ref.lstrip("./").replace("\\", "/")
        ref_lower = ref.lower()
        for f in workspace_files:
            path_lower = f["path"].lower()
            if path_lower == ref_lower or \
               path_lower.endswith("/" + ref_lower):
                referenced.add(f["path"])
            # Try with common extensions
            for ext in (".js", ".ts", ".jsx", ".tsx"):
                if path_lower == ref_lower + ext or \
                   path_lower.endswith("/" + ref_lower + ext):
                    referenced.add(f["path"])

    return list(referenced)


def find_relevant_files(query, file_list, prefs=None):
    """Score files by keyword relevance to query. Returns top 5."""
    prefs = prefs or {"excluded": [], "boosted": []}
    query_lower = query.lower()
    stop_words = {
        "the", "and", "for", "how", "does", "what", "this", "that",
        "with", "from", "can", "you", "are", "is", "it", "in", "to",
        "of", "a", "an", "my", "me", "do", "not", "but", "or",
    }
    words = [
        w for w in re.findall(r"\w+", query_lower)
        if len(w) >= 3 and w not in stop_words
    ]

    ext_keywords = {
        ".py": ["python", "flask", "django", "function", "class",
                "import", "def", "pip", "requirements",
                "code", "logic", "backend", "api", "route"],
        ".js": ["javascript", "frontend", "react", "vue", "function",
                "const", "node", "npm", "code", "logic"],
        ".ts": ["typescript", "frontend", "react", "angular",
                "interface", "code", "logic"],
        ".html": ["template", "page", "html", "frontend", "view",
                  "code", "ui"],
        ".css": ["style", "css", "layout", "design", "theme"],
        ".sql": ["database", "query", "table", "sql", "schema",
                 "migration"],
        ".yml": ["config", "yaml", "deploy", "docker", "pipeline",
                 "ci"],
        ".yaml": ["config", "yaml", "deploy", "docker", "pipeline",
                  "ci"],
        ".json": ["config", "package", "settings", "data",
                  "manifest"],
        ".md": ["readme", "docs", "documentation", "markdown"],
        ".sh": ["script", "bash", "shell", "install", "deploy",
                "code"],
        ".go": ["golang", "handler", "server", "middleware", "code"],
        ".rs": ["rust", "cargo", "struct", "impl", "code"],
        ".java": ["java", "spring", "class", "maven", "gradle",
                  "code"],
        ".rb": ["ruby", "rails", "gem", "code"],
        ".php": ["php", "laravel", "composer", "code"],
    }

    # Source code extensions — files that contain application logic
    source_exts = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
        ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
        ".kt", ".scala", ".lua", ".r", ".m", ".html", ".css",
        ".vue", ".svelte",
    }

    # Ops/infra extensions — only relevant if query is about deploy/config
    ops_exts = {
        ".service", ".conf", ".cfg", ".ini", ".env", ".toml",
        ".yml", ".yaml", ".json", ".xml", ".tf", ".hcl",
    }

    # Query about ops/infra? Then don't penalize those files
    ops_keywords = {
        "deploy", "service", "config", "install", "setup", "systemd",
        "docker", "nginx", "grafana", "telegraf", "ansible", "terraform",
        "ci", "cd", "pipeline", "infra", "infrastructure",
    }
    query_is_ops = bool(ops_keywords & set(words))

    # Source code files get a strong bonus over config/metadata
    source_entry_points = {
        "app.py": 40, "main.py": 40, "server.py": 35,
        "index.js": 40, "index.ts": 40, "index.html": 35,
        "main.go": 40, "main.rs": 40, "main.java": 40,
        "manage.py": 30, "wsgi.py": 25, "asgi.py": 25,
    }
    # Config/metadata get a small bonus — useful but not primary
    config_entry_points = {
        "readme.md": 10, "package.json": 8, "requirements.txt": 8,
        "dockerfile": 8, "docker-compose.yml": 8, "makefile": 8,
        "setup.py": 8, "setup.cfg": 5, "pyproject.toml": 8,
    }

    scores = []
    for f in file_list:
        path_lower = f["path"].lower()
        basename = os.path.basename(f["path"]).lower()
        name_no_ext = os.path.splitext(basename)[0]
        score = 0

        ext = f.get("ext", os.path.splitext(f["path"])[1]).lower()

        # Exact filename mentioned in query — always relevant
        if basename in query_lower:
            score += 100
        if name_no_ext in query_lower and len(name_no_ext) >= 3:
            score += 80

        # Query words appear in path
        for word in words:
            if word in basename:
                score += 30
            elif word in path_lower:
                score += 10

        # Extension-keyword association
        for kw in ext_keywords.get(ext, []):
            if kw in query_lower:
                score += 15

        # Entry point bonuses — source code over config
        if basename in source_entry_points:
            score += source_entry_points[basename]
        elif basename in config_entry_points:
            score += config_entry_points[basename]

        # Source code files get a natural boost
        if ext in source_exts:
            score += 10

        # Penalize ops/infra files unless query is about ops
        if not query_is_ops and ext in ops_exts:
            score -= 50
        # Penalize files in config/deploy/infra directories
        if not query_is_ops:
            for seg in ("config", "deploy", "infra", ".github"):
                if seg in path_lower.split("/"):
                    score -= 30
                    break

        # Apply learned preferences
        file_path = f["path"]
        if file_path in prefs["excluded"]:
            continue  # skip entirely
        if basename in prefs["excluded"]:
            continue  # also match by basename
        if file_path in prefs["boosted"]:
            score += 60
        elif basename in prefs["boosted"]:
            score += 60

        if score > 0:
            scores.append((f, score))

    scores.sort(key=lambda x: -x[1])
    return scores[:5]


def log_metric(measurement, tags, fields):
    """Log a metric to InfluxDB using line protocol."""
    try:
        def sanitize_tag(v):
            # Replace characters that break line protocol instead of escaping
            return (str(v).replace(" ", "_").replace(",", "_")
                    .replace("=", "_").replace('"', "").replace("'", ""))

        tag_str = ",".join(
            f"{sanitize_tag(k)}={sanitize_tag(v)}" for k, v in tags.items()
        ) if tags else ""
        field_parts = []
        for k, v in fields.items():
            if isinstance(v, int):
                field_parts.append(f"{k}={v}i")
            elif isinstance(v, float):
                field_parts.append(f"{k}={v}")
            else:
                escaped = str(v).replace('"', '\\"')
                field_parts.append(f'{k}="{escaped}"')
        field_str = ",".join(field_parts)
        line = measurement
        if tag_str:
            line += f",{tag_str}"
        line += f" {field_str}"
        r = requests.post(
            f"{INFLUXDB_URL}/write?db={INFLUXDB_DB}", data=line, timeout=5
        )
        if r.status_code != 204:
            app.logger.warning("InfluxDB write failed (%s): %s", r.status_code, r.text)
    except Exception as e:
        app.logger.warning("InfluxDB log_metric error: %s", e)


def get_system_resources():
    """Detect host resources to optimize SD parameters."""
    info = {"cpu_count": os.cpu_count() or 4, "has_gpu": False, "ram_gb": 4}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    info["ram_gb"] = int(line.split()[1]) / 1024 / 1024
                    break
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            info["has_gpu"] = True
            info["gpu_name"] = result.stdout.strip()
    except Exception:
        pass
    return info


def load_tuned_params():
    """Load previously tuned parameters from disk."""
    try:
        with open(TUNE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_tuned_params(params):
    """Persist tuned parameters to disk."""
    with open(TUNE_FILE, "w") as f:
        json.dump(params, f, indent=2)


def get_sd_params():
    """Return SD params: tuned if available, otherwise resource-based defaults."""
    tuned = load_tuned_params()
    if tuned:
        return {
            "steps": tuned["steps"],
            "width": tuned["width"],
            "height": tuned["height"],
            "sampler_name": tuned["sampler_name"],
            "cfg_scale": tuned.get("cfg_scale", 7),
            "batch_size": 1,
        }
    return get_default_sd_params()


def get_default_sd_params():
    """Resource-based defaults before tuning has run."""
    res = get_system_resources()
    if res["has_gpu"]:
        return {
            "steps": 30, "width": 512, "height": 512,
            "sampler_name": "DPM++ 2M", "cfg_scale": 7, "batch_size": 1,
        }
    ram = res["ram_gb"]
    cpus = res["cpu_count"]
    if ram < 8 or cpus <= 2:
        return {
            "steps": 12, "width": 384, "height": 384,
            "sampler_name": "Euler", "cfg_scale": 7, "batch_size": 1,
        }
    if ram < 20 or cpus <= 4:
        return {
            "steps": 15, "width": 512, "height": 512,
            "sampler_name": "Euler", "cfg_scale": 7, "batch_size": 1,
        }
    return {
        "steps": 20, "width": 512, "height": 512,
        "sampler_name": "Euler a", "cfg_scale": 7, "batch_size": 1,
    }


def benchmark_sd(prompt, params, timeout=600):
    """Run a single SD generation and return time taken, or None on failure."""
    try:
        start = time.time()
        r = requests.post(
            f"{SD_URL}/sdapi/v1/txt2img",
            json={**params, "prompt": prompt, "negative_prompt": "", "batch_size": 1},
            timeout=timeout,
        )
        elapsed = time.time() - start
        if r.status_code == 200 and r.json().get("images"):
            return elapsed
    except Exception:
        pass
    return None


def run_autotune():
    """Benchmark samplers, step counts, and resolutions using the active SD model."""
    global tuning_state

    # Identify which model is currently loaded
    active_model = "unknown"
    try:
        r = requests.get(f"{SD_URL}/sdapi/v1/options", timeout=10)
        active_model = r.json().get("sd_model_checkpoint", "unknown")
    except Exception:
        pass

    tuning_state = {
        "running": True,
        "progress": f"Tuning for model: {active_model}...",
        "results": None,
    }

    test_prompt = "a simple red cube on a white background"
    results = []

    # Samplers to test (fast ones for CPU)
    samplers = ["Euler", "Euler a", "LMS", "DPM++ 2M", "DDIM", "UniPC"]
    resolutions = [(384, 384), (512, 512)]
    step_counts = [10, 15, 20]

    # Phase 1: find fastest sampler at low res/steps
    tuning_state["progress"] = "Phase 1: Testing samplers..."
    sampler_times = {}
    for sampler in samplers:
        tuning_state["progress"] = f"Testing sampler: {sampler}..."
        params = {"steps": 10, "width": 384, "height": 384, "sampler_name": sampler, "cfg_scale": 7}
        t = benchmark_sd(test_prompt, params, timeout=300)
        if t is not None:
            sampler_times[sampler] = t
            results.append({
                "sampler": sampler, "steps": 10,
                "res": "384x384", "time": round(t, 1),
            })

    if not sampler_times:
        tuning_state = {
            "running": False,
            "progress": "Failed: no sampler produced output",
            "results": None,
        }
        return

    # Pick top 3 fastest samplers
    ranked_samplers = sorted(sampler_times.items(), key=lambda x: x[1])
    top_samplers = [s[0] for s in ranked_samplers[:3]]

    # Phase 2: test step counts with top samplers
    tuning_state["progress"] = "Phase 2: Optimizing step count..."
    for sampler in top_samplers:
        for steps in step_counts:
            if steps == 10:
                continue  # Already tested
            tuning_state["progress"] = f"Testing {sampler} @ {steps} steps..."
            params = {
                "steps": steps, "width": 384, "height": 384,
                "sampler_name": sampler, "cfg_scale": 7,
            }
            t = benchmark_sd(test_prompt, params, timeout=300)
            if t is not None:
                results.append({
                    "sampler": sampler, "steps": steps,
                    "res": "384x384", "time": round(t, 1),
                })

    # Phase 3: test resolutions with best sampler
    best_sampler = ranked_samplers[0][0]
    tuning_state["progress"] = "Phase 3: Testing resolutions..."
    for w, h in resolutions:
        if w == 384:
            continue  # Already tested
        tuning_state["progress"] = f"Testing {best_sampler} @ {w}x{h}..."
        params = {
            "steps": 15, "width": w, "height": h,
            "sampler_name": best_sampler, "cfg_scale": 7,
        }
        t = benchmark_sd(test_prompt, params, timeout=600)
        if t is not None:
            results.append({
                "sampler": best_sampler, "steps": 15,
                "res": f"{w}x{h}", "time": round(t, 1),
            })

    # Phase 4: tune OMP/MKL thread counts
    tuning_state["progress"] = "Phase 4: Tuning thread counts..."
    cpu_count = os.cpu_count() or 4
    thread_options = list({max(1, cpu_count // 2), cpu_count, max(1, cpu_count - 1)})
    thread_results = {}
    base_params = {
        "steps": 10, "width": 384, "height": 384,
        "sampler_name": best_sampler, "cfg_scale": 7,
    }

    for threads in sorted(thread_options):
        tuning_state["progress"] = f"Testing {threads} threads..."
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["TORCH_NUM_THREADS"] = str(threads)
        t = benchmark_sd(test_prompt, base_params, timeout=300)
        if t is not None:
            thread_results[threads] = t
            results.append({"test": f"threads={threads}", "time": round(t, 1)})

    best_threads = min(thread_results, key=thread_results.get) if thread_results else cpu_count

    # Decide optimal params — balance quality vs time
    # Allow up to 120s for a generation on CPU; pick highest quality within budget
    TIME_BUDGET = 120
    viable = [r for r in results if r["time"] <= TIME_BUDGET and "sampler" in r]
    if not viable:
        viable = sorted([r for r in results if "sampler" in r], key=lambda x: x["time"])[:1]

    if viable:
        # Prefer higher steps and resolution within budget
        viable.sort(key=lambda r: (-int(r["res"].split("x")[0]), -r["steps"]))
        pick = viable[0]
        w, h = pick["res"].split("x")
        optimal = {
            "sampler_name": pick["sampler"],
            "steps": pick["steps"],
            "width": int(w),
            "height": int(h),
            "cfg_scale": 7,
            "best_threads": best_threads,
            "benchmark_results": results,
            "tuned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tuned_model": active_model,
        }
    else:
        optimal = {
            "sampler_name": best_sampler,
            "steps": 10,
            "width": 384,
            "height": 384,
            "cfg_scale": 7,
            "best_threads": best_threads,
            "benchmark_results": results,
            "tuned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tuned_model": active_model,
        }

    save_tuned_params(optimal)

    # Apply optimal thread settings
    os.environ["OMP_NUM_THREADS"] = str(best_threads)
    os.environ["MKL_NUM_THREADS"] = str(best_threads)
    os.environ["TORCH_NUM_THREADS"] = str(best_threads)

    tuning_state = {"running": False, "progress": "Complete", "results": optimal}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def list_models():
    """List available Ollama models."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        models = []
    return jsonify(models=models)


@app.route("/api/resources")
def resources():
    """Return detected resources, SD params, tuning state, and active models."""
    tuned = load_tuned_params()
    # Get active SD model
    sd_model = ""
    try:
        opts = requests.get(f"{SD_URL}/sdapi/v1/options", timeout=3).json()
        sd_model = opts.get("sd_model_checkpoint", "")
    except Exception:
        pass
    return jsonify(
        resources=get_system_resources(),
        sd_params=get_sd_params(),
        sd_model=sd_model,
        is_tuned=tuned is not None,
        tuned_at=tuned.get("tuned_at") if tuned else None,
        tuning=tuning_state,
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """Stream a chat response from Ollama."""
    data = request.json
    model = data.get("model", "mistral:latest")
    messages = data.get("messages", [])
    options = data.get("options", {})

    # Track active chat
    active_tasks["ollama"]["count"] += 1
    active_tasks["ollama"]["active"] = True
    active_tasks["ollama"]["model"] = model
    active_tasks["ollama"]["started"] = time.time()
    last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    active_tasks["ollama"]["prompt"] = last_user[:100]

    def generate():
        start = time.time()
        token_count = 0
        try:
            payload = {"model": model, "messages": messages, "stream": True}
            if options:
                payload["options"] = {k: v for k, v in options.items() if v is not None}
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                stream=True, timeout=300
            )
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    done = chunk.get("done", False)
                    if token:
                        token_count += 1
                    yield f"data: {json.dumps({'token': token, 'done': done})}\n\n"
                    if done:
                        elapsed = round(time.time() - start, 2)
                        tps = round(token_count / elapsed, 2) if elapsed > 0 else 0
                        log_metric(
                            "chat_response",
                            {"model": model},
                            {"tokens": token_count,
                             "response_time": elapsed,
                             "tokens_per_sec": tps},
                        )
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        finally:
            active_tasks["ollama"]["count"] -= 1
            if active_tasks["ollama"]["count"] <= 0:
                active_tasks["ollama"]["active"] = False
                active_tasks["ollama"]["count"] = 0

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/image", methods=["POST"])
def generate_image():
    """Generate an image via Stable Diffusion API with queuing."""
    data = request.json
    prompt = data.get("prompt", "")
    negative_prompt = data.get("negative_prompt", "")

    params = get_sd_params()
    for key in ("steps", "width", "height", "cfg_scale", "sampler_name", "seed"):
        if key in data and data[key] is not None:
            params[key] = data[key]
    params["prompt"] = prompt
    params["negative_prompt"] = negative_prompt or "blurry, low quality, distorted"

    # If the lock is held, track this request as queued
    if sd_lock.locked():
        sd_queue_prompts.append(prompt[:100])
        app.logger.info(
            "Image request queued (position %d): %s",
            len(sd_queue_prompts), prompt[:60],
        )

    # Block until it's our turn
    with sd_lock:
        if prompt[:100] in sd_queue_prompts:
            sd_queue_prompts.remove(prompt[:100])

        active_model = "unknown"
        try:
            opts = requests.get(f"{SD_URL}/sdapi/v1/options", timeout=10).json()
            active_model = opts.get("sd_model_checkpoint", "unknown")
        except Exception:
            pass

        active_tasks["sd"]["active"] = True
        active_tasks["sd"]["prompt"] = prompt[:100]
        active_tasks["sd"]["started"] = time.time()
        active_tasks["sd"]["model"] = active_model

        try:
            start = time.time()
            r = requests.post(f"{SD_URL}/sdapi/v1/txt2img", json=params, timeout=600)
            elapsed = round(time.time() - start, 1)
            result = r.json()
            images = result.get("images", [])
            info = result.get("info", "")
            params["generation_time"] = f"{elapsed}s"

            log_metric(
                "sd_generation",
                {"model": active_model, "sampler": params.get("sampler_name", "unknown")},
                {
                    "generation_time": float(elapsed),
                    "steps": int(params.get("steps", 0)),
                    "width": int(params.get("width", 0)),
                    "height": int(params.get("height", 0)),
                    "cfg_scale": float(params.get("cfg_scale", 7)),
                },
            )

            return jsonify(images=images, info=info, params_used=params)
        except Exception as e:
            return jsonify(error=str(e)), 500
        finally:
            active_tasks["sd"]["active"] = False


@app.route("/api/image/progress")
def image_progress():
    """Proxy SD WebUI's progress endpoint, with queue status."""
    queued = len(sd_queue_prompts)
    try:
        r = requests.get(
            f"{SD_URL}/sdapi/v1/progress?skip_current_image=false",
            timeout=5,
        )
        data = r.json()
        progress = data.get("progress", 0)
        state = data.get("state", {})
        step = state.get("sampling_step", 0)
        total_steps = state.get("sampling_steps", 0)
        eta = data.get("eta_relative", 0)
        current_image = data.get("current_image", None)
        return jsonify(
            progress=round(progress * 100, 1),
            step=step,
            total_steps=total_steps,
            eta=round(eta, 1),
            current_image=current_image,
            queued=queued,
            active=active_tasks["sd"]["active"],
            active_prompt=active_tasks["sd"].get("prompt", ""),
        )
    except Exception:
        return jsonify(
            progress=0, step=0, total_steps=0, eta=0,
            current_image=None, queued=queued,
            active=active_tasks["sd"]["active"],
            active_prompt=active_tasks["sd"].get("prompt", ""),
        )


@app.route("/api/image/cancel", methods=["POST"])
def image_cancel():
    """Interrupt the current SD generation."""
    try:
        requests.post(f"{SD_URL}/sdapi/v1/interrupt", timeout=5)
        return jsonify(status="cancelled")
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/image/skip", methods=["POST"])
def image_skip():
    """Skip remaining steps and return current result."""
    try:
        requests.post(f"{SD_URL}/sdapi/v1/skip", timeout=5)
        return jsonify(status="skipped")
    except Exception as e:
        return jsonify(error=str(e)), 500


# ── Music generation (Ollama → LilyPond → MIDI → MP3) ─────────────────────


MUSIC_SYSTEM_PROMPT = r"""You are a music composition assistant.
When asked to compose music, respond with valid LilyPond notation
inside a ```lilypond code block.

CRITICAL RULES:
1. Start with \version "2.22.1"
2. Include \header with title and composer
3. Use a \score block containing music, \layout { } and \midi { \tempo 4 = BPM }
4. NEVER put \markup inside \score — put it outside or omit it
5. Every bar MUST have the correct total duration for the time signature
   (e.g. 4/4 = 4 quarter beats)
6. Count beats carefully: c4 = 1 beat, c2 = 2 beats, c1 = 4 beats, c8 = 0.5 beats
7. Use \relative c' for simple melodies, \new Staff for each voice in polyphony
8. For piano, use \new PianoStaff << \new Staff { treble } \new Staff { \clef bass bass } >>
9. Keep pieces 8-32 measures. Use repeat bars (e.g. \repeat volta 2 { }) when appropriate
10. Add dynamics (\p, \f, \mf, \<, \>) and articulations (-., ->, --)
11. Before the code block, write 1-2 sentences describing what you composed

Example:
A gentle waltz in C major with a simple melody.

```lilypond
\version "2.22.1"
\header {
  title = "Simple Waltz"
  composer = "AI Composer"
}
\score {
  \relative c' {
    \time 3/4
    \tempo 4 = 120
    c4\mf e g | c2. | b4 d g | c,2.\fermata |
  }
  \layout { }
  \midi { }
}
```"""


@app.route("/api/music", methods=["POST"])
def generate_music():
    """Generate sheet music and MP3 from a text prompt via Ollama + LilyPond."""
    data = request.json
    prompt = data.get("prompt", "")
    model = data.get("model", "mistral:latest")

    if not prompt:
        return jsonify(error="No prompt provided"), 400

    active_tasks["music"]["active"] = True
    active_tasks["music"]["prompt"] = prompt[:100]
    active_tasks["music"]["started"] = time.time()
    active_tasks["music"]["model"] = model
    active_tasks["music"]["phase"] = "composing"
    active_tasks["music"]["cancel"] = False

    try:
        # Step 1: Ask Ollama to generate LilyPond notation (streaming to avoid timeout)
        active_tasks["music"]["phase"] = "composing"
        app.logger.info("Music: asking %s to compose: %s", model, prompt[:80])

        ollama_resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": MUSIC_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
            },
            stream=True,
            timeout=600,
        )
        full_response = ""
        for line in ollama_resp.iter_lines():
            if active_tasks["music"]["cancel"]:
                ollama_resp.close()
                active_tasks["music"]["active"] = False
                return jsonify(error="Music generation cancelled"), 499
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                full_response += token
                if chunk.get("done", False):
                    break

        if not full_response:
            return jsonify(error="Ollama returned empty response"), 500

        # Extract LilyPond code from response
        lilypond_code = None
        # Try ```lilypond ... ``` first
        match = re.search(r"```lilypond\s*\n(.*?)```", full_response, re.DOTALL)
        if match:
            lilypond_code = match.group(1).strip()
        else:
            # Try generic ``` ... ```
            match = re.search(r"```\s*\n(.*?)```", full_response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if "\\relative" in code or "\\score" in code or "\\version" in code:
                    lilypond_code = code

        if not lilypond_code:
            return jsonify(
                error="Could not extract LilyPond notation from response",
                raw_response=full_response,
            ), 422

        # Clean up common LLM mistakes in LilyPond output
        # Remove \markup inside \score (invalid)
        lilypond_code = re.sub(
            r"\\markup\s*\{[^}]*\}\s*(?=\})", "", lilypond_code
        )
        # Remove prose lines that snuck inside the code block
        cleaned_lines = []
        for line in lilypond_code.split("\n"):
            stripped = line.strip()
            # Skip lines that look like English prose (not LilyPond)
            if (stripped and not stripped.startswith("\\") and
                not stripped.startswith("%") and
                not stripped.startswith("{") and
                not stripped.startswith("}") and
                not stripped.startswith("<<") and
                not stripped.startswith(">>") and
                not any(c in stripped for c in "{}\\|~") and
                len(stripped) > 3 and
                stripped[0].isalpha() and " " in stripped and
                not any(stripped.startswith(n) for n in
                        ["c", "d", "e", "f", "g", "a", "b", "r", "s"])):
                continue
            cleaned_lines.append(line)
        lilypond_code = "\n".join(cleaned_lines)

        # Ensure \midi block exists for MIDI output
        if "\\midi" not in lilypond_code:
            lilypond_code = lilypond_code.replace(
                "\\layout { }",
                "\\layout { }\n  \\midi { \\tempo 4 = 120 }",
            )
            if "\\midi" not in lilypond_code:
                # No \layout either — inject both before final }
                lilypond_code = lilypond_code.rstrip().rstrip("}")
                lilypond_code += "\n  \\layout { }\n  \\midi { \\tempo 4 = 120 }\n}"

        # Ensure \version header
        if "\\version" not in lilypond_code:
            lilypond_code = '\\version "2.22.1"\n' + lilypond_code

        # Extract description (text before the code block)
        desc_match = re.search(r"^(.*?)```", full_response, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""

        # Step 2: Render with LilyPond → PNG + MIDI
        active_tasks["music"]["phase"] = "engraving"
        app.logger.info("Music: rendering LilyPond notation")

        work_dir = tempfile.mkdtemp(dir=MUSIC_DIR)

        def compile_lilypond(code, attempt=1):
            """Compile LilyPond code, retry once by asking Ollama to fix errors."""
            ly_file = os.path.join(work_dir, f"score{'_retry' if attempt > 1 else ''}.ly")
            with open(ly_file, "w") as f:
                f.write(code)

            result = subprocess.run(
                [
                    "lilypond",
                    "-dbackend=eps",
                    "-dno-gs-load-fonts",
                    "-dinclude-eps-fonts",
                    "--png",
                    "-dresolution=200",
                    ly_file,
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=work_dir,
            )

            if result.returncode != 0 and attempt == 1:
                # Ask Ollama to fix the error
                active_tasks["music"]["phase"] = "fixing"
                app.logger.info("Music: LilyPond failed, asking Ollama to fix")
                err_snippet = result.stderr[-400:]
                fix_resp = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": MUSIC_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": f"```lilypond\n{code}\n```"},
                            {
                                "role": "user",
                                "content": (
                                    f"The LilyPond code has errors:\n{err_snippet}\n\n"
                                    "Fix the code. Reply with ONLY the corrected "
                                    "LilyPond code in a ```lilypond block. "
                                    "Count beats carefully for each bar."
                                ),
                            },
                        ],
                        "stream": True,
                    },
                    stream=True,
                    timeout=600,
                )
                fix_text = ""
                for fline in fix_resp.iter_lines():
                    if fline:
                        fchunk = json.loads(fline)
                        fix_text += fchunk.get("message", {}).get("content", "")
                        if fchunk.get("done", False):
                            break
                fix_match = re.search(r"```lilypond\s*\n(.*?)```", fix_text, re.DOTALL)
                if not fix_match:
                    fix_match = re.search(r"```\s*\n(.*?)```", fix_text, re.DOTALL)
                if fix_match:
                    fixed_code = fix_match.group(1).strip()
                    if "\\version" not in fixed_code:
                        fixed_code = '\\version "2.22.1"\n' + fixed_code
                    active_tasks["music"]["phase"] = "engraving"
                    return compile_lilypond(fixed_code, attempt=2)

            return result, code

        lily_result, lilypond_code = compile_lilypond(lilypond_code)

        if lily_result.returncode != 0:
            app.logger.warning("LilyPond failed: %s", lily_result.stderr)
            return jsonify(
                error="LilyPond compilation failed",
                details=lily_result.stderr[-500:],
                lilypond=lilypond_code,
                description=description,
            ), 422

        # Find output files
        png_files = sorted(globmod.glob(os.path.join(work_dir, "*.png")))
        midi_files = globmod.glob(os.path.join(work_dir, "*.midi")) + globmod.glob(
            os.path.join(work_dir, "*.mid")
        )

        if not png_files:
            return jsonify(error="LilyPond produced no PNG output", lilypond=lilypond_code), 500

        # Read sheet music PNG(s)
        sheet_images = []
        for png_file in png_files:
            with open(png_file, "rb") as f:
                sheet_images.append(base64.b64encode(f.read()).decode("ascii"))

        result = {
            "description": description,
            "lilypond": lilypond_code,
            "sheet_music": sheet_images,
        }

        # Step 3: MIDI → WAV → MP3
        if midi_files:
            active_tasks["music"]["phase"] = "synthesizing"
            app.logger.info("Music: synthesizing MIDI to audio")

            midi_file = midi_files[0]
            wav_file = os.path.join(work_dir, "output.wav")
            mp3_file = os.path.join(work_dir, "output.mp3")

            # Timidity: MIDI → WAV
            tim_result = subprocess.run(
                ["timidity", midi_file, "-Ow", "-o", wav_file],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if tim_result.returncode == 0 and os.path.exists(wav_file):
                # ffmpeg: WAV → MP3
                active_tasks["music"]["phase"] = "encoding"
                ff_result = subprocess.run(
                    ["ffmpeg", "-y", "-i", wav_file, "-b:a", "192k", mp3_file],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if ff_result.returncode == 0 and os.path.exists(mp3_file):
                    with open(mp3_file, "rb") as f:
                        result["mp3"] = base64.b64encode(f.read()).decode("ascii")
                    result["mp3_size"] = os.path.getsize(mp3_file)
                else:
                    app.logger.warning("ffmpeg failed: %s", ff_result.stderr[-300:])
                    result["audio_error"] = "MP3 encoding failed"
            else:
                app.logger.warning("Timidity failed: %s", tim_result.stderr[-300:])
                result["audio_error"] = "MIDI synthesis failed"
        else:
            result["audio_error"] = "No MIDI output from LilyPond"

        elapsed = round(time.time() - active_tasks["music"]["started"], 1)
        result["generation_time"] = f"{elapsed}s"

        log_metric(
            "music_generation",
            {"model": model},
            {"generation_time": float(elapsed)},
        )

        # Clean up all temp files — data is already base64-encoded in result
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

        return jsonify(result)

    except requests.exceptions.Timeout:
        return jsonify(error="Request timed out during music generation"), 504
    except Exception as e:
        app.logger.error("Music generation error: %s", e)
        return jsonify(error=str(e)), 500
    finally:
        active_tasks["music"]["active"] = False
        active_tasks["music"]["phase"] = ""


@app.route("/api/music/progress")
def music_progress():
    """Return current music generation phase."""
    task = active_tasks["music"]
    if task["active"]:
        elapsed = round(time.time() - task["started"], 1) if task["started"] else 0
        return jsonify(active=True, phase=task["phase"], elapsed=elapsed)
    return jsonify(active=False, phase="")


# ── Code workspace routes ───────────────────────────────────────────────────


@app.route("/api/workspaces")
def list_workspaces():
    """List all code workspaces."""
    workspaces = []
    try:
        for name in sorted(os.listdir(WORKSPACES_DIR)):
            ws_path = os.path.join(WORKSPACES_DIR, name)
            if not os.path.isdir(ws_path):
                continue
            file_count = 0
            total_size = 0
            for root, dirs, filenames in os.walk(ws_path):
                dirs[:] = [
                    d for d in dirs if d not in SKIP_DIRS
                ]
                for f in filenames:
                    ext = os.path.splitext(f)[1].lower()
                    if ext not in SKIP_EXTENSIONS:
                        file_count += 1
                        try:
                            total_size += os.path.getsize(
                                os.path.join(root, f)
                            )
                        except OSError:
                            pass
            workspaces.append({
                "name": name,
                "file_count": file_count,
                "size_mb": round(total_size / 1048576, 1),
            })
    except FileNotFoundError:
        pass
    return jsonify(workspaces=workspaces)


@app.route("/api/workspace/upload", methods=["POST"])
def workspace_upload():
    """Upload a directory of files as a workspace."""
    if "files" not in request.files:
        return jsonify(error="No files provided"), 400

    uploaded = request.files.getlist("files")
    if not uploaded:
        return jsonify(error="No files provided"), 400

    # Derive workspace name from common directory prefix
    name = request.form.get("name", "")
    if not name:
        first_path = (
            uploaded[0].filename
            or getattr(uploaded[0], "webkitRelativePath", "") or ""
        )
        parts = first_path.replace("\\", "/").split("/")
        name = parts[0] if len(parts) > 1 else "upload"
    name = sanitize_workspace_name(name)

    ws_path = os.path.join(WORKSPACES_DIR, name)
    os.makedirs(ws_path, exist_ok=True)

    saved = 0
    for f in uploaded:
        rel = (
            f.filename
            or getattr(f, "webkitRelativePath", "") or ""
        ).replace("\\", "/")
        if not rel:
            continue

        # Strip common root directory from path
        parts = rel.split("/")
        if len(parts) > 1:
            rel = "/".join(parts[1:])

        # Security: reject path traversal
        if ".." in rel:
            continue

        dest = os.path.join(ws_path, rel)
        real_dest = os.path.realpath(dest)
        if not real_dest.startswith(os.path.realpath(ws_path)):
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        f.save(dest)
        saved += 1

    files, tree = build_file_tree(ws_path)
    app.logger.info(
        "Workspace uploaded: %s (%d files)", name, saved
    )
    return jsonify(name=name, file_count=len(files), tree=tree)


@app.route("/api/workspace/clone", methods=["POST"])
def workspace_clone():
    """Clone a git repo as a workspace (background)."""
    if clone_state["active"]:
        return jsonify(error="Clone already in progress"), 409

    data = request.json
    url = data.get("url", "")
    if not url:
        return jsonify(error="No URL provided"), 400

    # Only allow https:// URLs
    if not url.startswith("https://"):
        return jsonify(error="Only HTTPS URLs are supported"), 400

    name = data.get("name", "")
    if not name:
        # Derive name from URL
        name = url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
    name = sanitize_workspace_name(name)
    ws_path = os.path.join(WORKSPACES_DIR, name)

    def do_clone():
        global clone_state
        clone_state = {
            "active": True, "progress": "Cloning...", "name": name,
        }
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, ws_path],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                clone_state = {
                    "active": False, "name": name,
                    "progress": f"Failed: {result.stderr[:200]}",
                }
                return
            clone_state = {
                "active": False, "progress": "Complete",
                "name": name,
            }
            app.logger.info("Workspace cloned: %s from %s", name, url)
        except subprocess.TimeoutExpired:
            clone_state = {
                "active": False, "progress": "Failed: timeout",
                "name": name,
            }
        except Exception as e:
            clone_state = {
                "active": False,
                "progress": f"Failed: {e}",
                "name": name,
            }

    t = threading.Thread(target=do_clone, daemon=True)
    t.start()
    return jsonify(status="started", name=name)


@app.route("/api/workspace/clone/progress")
def workspace_clone_progress():
    """Poll clone progress."""
    return jsonify(clone_state)


@app.route("/api/workspace/<name>", methods=["DELETE"])
def workspace_delete(name):
    """Delete a workspace."""
    name = sanitize_workspace_name(name)
    ws_path = os.path.join(WORKSPACES_DIR, name)
    if not os.path.isdir(ws_path):
        return jsonify(error="Workspace not found"), 404
    real = os.path.realpath(ws_path)
    if not real.startswith(os.path.realpath(WORKSPACES_DIR)):
        return jsonify(error="Invalid workspace"), 400
    shutil.rmtree(ws_path, ignore_errors=True)
    app.logger.info("Workspace deleted: %s", name)
    return jsonify(status="deleted")


@app.route("/api/workspace/<name>/tree")
def workspace_tree(name):
    """Return file tree for a workspace."""
    name = sanitize_workspace_name(name)
    ws_path = os.path.join(WORKSPACES_DIR, name)
    if not os.path.isdir(ws_path):
        return jsonify(error="Workspace not found"), 404
    files, tree = build_file_tree(ws_path)
    return jsonify(name=name, tree=tree, files=files)


@app.route("/api/workspace/<name>/file")
def workspace_file(name):
    """Read a single file from a workspace."""
    name = sanitize_workspace_name(name)
    path = request.args.get("path", "")
    if not path or ".." in path:
        return jsonify(error="Invalid path"), 400

    ws_path = os.path.join(WORKSPACES_DIR, name)
    full = os.path.join(ws_path, path)
    real = os.path.realpath(full)
    if not real.startswith(os.path.realpath(ws_path)):
        return jsonify(error="Path traversal denied"), 403
    if not os.path.isfile(full):
        return jsonify(error="File not found"), 404

    size = os.path.getsize(full)
    if size > MAX_FILE_SIZE:
        # Read first MAX_FILE_SIZE bytes
        with open(full, "r", errors="replace") as f:
            content = f.read(MAX_FILE_SIZE)
        content += f"\n\n[... truncated, file is {size} bytes ...]"
    else:
        try:
            with open(full, "r", errors="replace") as f:
                content = f.read()
        except Exception as e:
            return jsonify(error=str(e)), 500

    ext = os.path.splitext(path)[1].lower()
    lang_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".html": "html", ".css": "css", ".json": "json",
        ".yml": "yaml", ".yaml": "yaml", ".sh": "bash",
        ".go": "go", ".rs": "rust", ".java": "java",
        ".rb": "ruby", ".php": "php", ".sql": "sql",
        ".md": "markdown", ".xml": "xml", ".toml": "toml",
        ".cfg": "ini", ".ini": "ini", ".env": "bash",
        ".jsx": "javascript", ".tsx": "typescript",
        ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    }
    return jsonify(
        path=path, content=content, size=size,
        language=lang_map.get(ext, "text"),
    )


@app.route("/api/workspace/<name>/context", methods=["POST"])
def workspace_context(name):
    """Build context for code-aware chat."""
    name = sanitize_workspace_name(name)
    ws_path = os.path.join(WORKSPACES_DIR, name)
    if not os.path.isdir(ws_path):
        return jsonify(error="Workspace not found"), 404

    data = request.json
    query = data.get("query", "")
    mentioned = data.get("mentioned_files", [])
    num_ctx = data.get("num_ctx", 4096) or 4096

    # Token budget: 60% for code context
    token_budget = int(num_ctx * 0.6)

    files, tree_str = build_file_tree(ws_path)

    # System prompt with file tree
    system_prompt = (
        "You are analyzing code in the workspace '"
        + name + "'. Here is the file structure:\n\n"
        + tree_str + "\n\n"
        "When discussing code, reference specific file paths and "
        "line numbers. Be precise and technical. If the user "
        "references a file with @filename, its contents are "
        "included below."
    )

    tokens_used = estimate_tokens(system_prompt)

    # Gather files to include: mentioned first, then auto-relevant
    files_to_include = []
    mentioned_paths = set()

    # Parse @mentions from the query
    at_mentions = re.findall(r"@([\w./\\-]+)", query)
    all_mentioned = list(mentioned) + list(at_mentions)

    # Match mentioned files against workspace files
    for mention in all_mentioned:
        mention_lower = mention.lower().replace("\\", "/")
        for f in files:
            path_lower = f["path"].lower()
            basename = os.path.basename(path_lower)
            if (path_lower == mention_lower
                    or basename == mention_lower
                    or path_lower.endswith("/" + mention_lower)):
                if f["path"] not in mentioned_paths:
                    mentioned_paths.add(f["path"])
                    files_to_include.append(
                        (f, 200)  # high priority
                    )
                break

    # Auto-relevant files (with learned preferences)
    prefs = load_workspace_prefs(ws_path)
    relevant = find_relevant_files(query, files, prefs)
    for f, score in relevant:
        if f["path"] not in mentioned_paths:
            files_to_include.append((f, score))

    # Cross-reference tracking: scan top files for imports/references
    # to other workspace files and add those too
    included_paths = {f["path"] for f, _ in files_to_include}
    cross_ref_files = []
    for f, score in files_to_include[:3]:  # scan top 3 files only
        full = os.path.join(ws_path, f["path"])
        try:
            with open(full, "r", errors="replace") as fh:
                preview = fh.read(MAX_FILE_SIZE)
        except Exception:
            continue
        refs = find_cross_references(preview, files)
        for ref_path in refs:
            if ref_path not in included_paths:
                included_paths.add(ref_path)
                ref_file = next(
                    (x for x in files if x["path"] == ref_path), None
                )
                if ref_file:
                    cross_ref_files.append((ref_file, score * 0.7))
    files_to_include.extend(cross_ref_files)

    # Extract query keywords for smart section extraction
    stop_words = {
        "the", "and", "for", "how", "does", "what", "this", "that",
        "with", "from", "can", "you", "are", "is", "it", "in", "to",
        "of", "a", "an", "my", "me", "do", "not", "but", "or",
        "tell", "show", "explain", "describe",
    }
    words = [
        w for w in re.findall(r"\w+", query.lower())
        if len(w) >= 3 and w not in stop_words
    ]

    # large files to send relevant sections instead of the top N lines
    file_blocks = []
    remaining = token_budget - tokens_used
    for f, score in files_to_include:
        if remaining <= 100:
            break
        full = os.path.join(ws_path, f["path"])
        try:
            with open(full, "r", errors="replace") as fh:
                full_content = fh.read(MAX_FILE_SIZE)
        except Exception:
            continue

        all_lines = full_content.split("\n")
        max_lines = (remaining * 4) // 60  # ~60 chars per line

        # For large files, extract relevant sections
        if len(all_lines) > max_lines and words:
            content = extract_relevant_sections(
                all_lines, words, max_lines
            )
        elif len(all_lines) > max_lines:
            content = "\n".join(all_lines[:max_lines])
            content += (
                f"\n[...truncated at line {max_lines}"
                f" of {len(all_lines)}...]"
            )
        else:
            content = full_content

        file_tokens = estimate_tokens(content)
        if file_tokens > remaining:
            char_limit = remaining * 4
            content = content[:char_limit]
            clines = content.split("\n")
            content = "\n".join(clines[:-1])
            content += "\n[...truncated to fit context window...]"
            file_tokens = estimate_tokens(content)

        file_blocks.append({
            "path": f["path"],
            "content": content,
            "relevance_score": score,
        })
        remaining -= file_tokens
        tokens_used += file_tokens

    return jsonify(
        system_prompt=system_prompt,
        file_blocks=file_blocks,
        tokens_used=tokens_used,
        tokens_budget=token_budget,
    )


@app.route("/api/workspace/<name>/feedback", methods=["POST"])
def workspace_feedback(name):
    """Learn file preferences: exclude or boost files."""
    name = sanitize_workspace_name(name)
    ws_path = os.path.join(WORKSPACES_DIR, name)
    if not os.path.isdir(ws_path):
        return jsonify(error="Workspace not found"), 404

    data = request.json
    action = data.get("action", "")  # "exclude" or "boost"
    file_path = data.get("file", "")
    if not file_path or action not in ("exclude", "boost", "reset"):
        return jsonify(error="Invalid action or file"), 400

    prefs = load_workspace_prefs(ws_path)

    if action == "exclude":
        if file_path not in prefs["excluded"]:
            prefs["excluded"].append(file_path)
        # Remove from boosted if present
        prefs["boosted"] = [
            f for f in prefs["boosted"] if f != file_path
        ]
        app.logger.info(
            "Workspace %s: excluded %s", name, file_path
        )
    elif action == "boost":
        if file_path not in prefs["boosted"]:
            prefs["boosted"].append(file_path)
        # Remove from excluded if present
        prefs["excluded"] = [
            f for f in prefs["excluded"] if f != file_path
        ]
    elif action == "reset":
        prefs["excluded"] = [
            f for f in prefs["excluded"] if f != file_path
        ]
        prefs["boosted"] = [
            f for f in prefs["boosted"] if f != file_path
        ]

    save_workspace_prefs(ws_path, prefs)
    return jsonify(status="ok", prefs=prefs)


@app.route("/api/workspace/<name>/prefs")
def workspace_prefs(name):
    """Get learned preferences for a workspace."""
    name = sanitize_workspace_name(name)
    ws_path = os.path.join(WORKSPACES_DIR, name)
    if not os.path.isdir(ws_path):
        return jsonify(error="Workspace not found"), 404
    return jsonify(load_workspace_prefs(ws_path))


# ── Admin status page & task management ──────────────────────────────────────


@app.route("/status")
def status_page():
    return render_template("status.html")


@app.route("/api/status")
def api_status():
    """Return status of all services and active tasks."""
    now = time.time()
    result = {"services": {}, "tasks": {}}

    # Check Ollama service
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        result["services"]["ollama"] = {"status": "running", "models": models}
    except Exception:
        result["services"]["ollama"] = {"status": "unreachable", "models": []}

    # Check Ollama running models (loaded in memory)
    try:
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=3)
        running = r.json().get("models", [])
        result["services"]["ollama"]["loaded_models"] = [
            {"name": m.get("name", ""), "size": m.get("size", 0),
             "expires_at": m.get("expires_at", "")}
            for m in running
        ]
    except Exception:
        result["services"]["ollama"]["loaded_models"] = []

    # Check SD WebUI service
    try:
        r = requests.get(f"{SD_URL}/sdapi/v1/progress?skip_current_image=true", timeout=3)
        data = r.json()
        progress = data.get("progress", 0)
        state = data.get("state", {})
        sd_status = "generating" if progress > 0 and not state.get("interrupted", False) else "idle"
        result["services"]["sd"] = {
            "status": sd_status,
            "progress": round(progress * 100, 1),
            "step": state.get("sampling_step", 0),
            "total_steps": state.get("sampling_steps", 0),
        }
    except Exception:
        result["services"]["sd"] = {"status": "unreachable"}

    # Get active SD model
    try:
        opts = requests.get(f"{SD_URL}/sdapi/v1/options", timeout=3).json()
        result["services"]["sd"]["active_model"] = opts.get("sd_model_checkpoint", "unknown")
    except Exception:
        pass

    # Active tasks from our tracking
    for key in ("sd", "ollama", "music"):
        task = active_tasks[key]
        if task["active"]:
            elapsed = round(now - task["started"], 1) if task["started"] else 0
            result["tasks"][key] = {
                "active": True,
                "prompt": task["prompt"],
                "model": task["model"],
                "elapsed": elapsed,
            }
            if key == "ollama":
                result["tasks"][key]["count"] = task["count"]
            if key == "music":
                result["tasks"][key]["phase"] = task.get("phase", "")
        else:
            result["tasks"][key] = {"active": False}

    # Queue info
    result["queue"] = {"sd": len(sd_queue_prompts), "sd_prompts": list(sd_queue_prompts)}

    return jsonify(result)


@app.route("/api/status/cancel/sd", methods=["POST"])
def status_cancel_sd():
    """Cancel running SD generation and clear the queue."""
    try:
        requests.post(f"{SD_URL}/sdapi/v1/interrupt", timeout=5)
        active_tasks["sd"]["active"] = False
        sd_queue_prompts.clear()
        return jsonify(status="cancelled")
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/status/queue/clear", methods=["POST"])
def status_queue_clear():
    """Clear queued requests without cancelling the active generation."""
    sd_queue_prompts.clear()
    return jsonify(status="cleared")


@app.route("/api/status/cancel/ollama", methods=["POST"])
def status_cancel_ollama():
    """Abort Ollama — unload all models to kill running inference."""
    errors = []
    try:
        # Ollama doesn't have a direct cancel API, but we can unload the model
        # by loading it with keep_alive=0
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=3)
        running = r.json().get("models", [])
        for m in running:
            name = m.get("name", "")
            if name:
                requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": name, "keep_alive": 0},
                    timeout=5,
                )
    except Exception as e:
        errors.append(str(e))

    active_tasks["ollama"]["active"] = False
    active_tasks["ollama"]["count"] = 0
    if errors:
        return jsonify(status="cancelled with errors", errors=errors)
    return jsonify(status="cancelled")


@app.route("/api/status/cancel/music", methods=["POST"])
def status_cancel_music():
    """Cancel running music generation."""
    active_tasks["music"]["cancel"] = True
    active_tasks["music"]["active"] = False
    return jsonify(status="cancelled")


@app.route("/api/status/restart/sd", methods=["POST"])
def status_restart_sd():
    """Restart the SD WebUI systemd service."""
    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", "sd-webui"],
            capture_output=True, text=True, timeout=30,
        )
        active_tasks["sd"]["active"] = False
        return jsonify(status="restarting")
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/status/restart/ollama", methods=["POST"])
def status_restart_ollama():
    """Restart the Ollama systemd service."""
    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", "ollama"],
            capture_output=True, text=True, timeout=30,
        )
        active_tasks["ollama"]["active"] = False
        active_tasks["ollama"]["count"] = 0
        return jsonify(status="restarting")
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/status/restart/unified-chat", methods=["POST"])
def status_restart_self():
    """Restart the unified-chat service (self-restart)."""
    try:
        subprocess.Popen(
            ["sudo", "systemctl", "restart", "unified-chat"],
        )
        return jsonify(status="restarting")
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/tune", methods=["POST"])
def start_tune():
    """Start auto-tuning in a background thread."""
    if tuning_state["running"]:
        return jsonify(error="Tuning already in progress"), 409
    t = threading.Thread(target=run_autotune, daemon=True)
    t.start()
    return jsonify(status="started")


@app.route("/api/tune", methods=["GET"])
def tune_status():
    """Poll tuning progress."""
    return jsonify(tuning_state)


@app.route("/api/tune", methods=["DELETE"])
def reset_tune():
    """Clear tuned parameters and revert to defaults."""
    try:
        os.remove(TUNE_FILE)
    except FileNotFoundError:
        pass
    return jsonify(status="reset")


# ── Ollama model management ──────────────────────────────────────────────────

@app.route("/api/ollama/models")
def ollama_models():
    """List installed Ollama models with details."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = r.json().get("models", [])
        return jsonify(models=[{
            "name": m["name"],
            "size": m.get("size", 0),
            "modified": m.get("modified_at", ""),
            "digest": m.get("digest", "")[:12],
            "family": m.get("details", {}).get("family", ""),
            "params": m.get("details", {}).get("parameter_size", ""),
            "quant": m.get("details", {}).get("quantization_level", ""),
        } for m in models])
    except Exception as e:
        return jsonify(error=str(e), models=[]), 500


@app.route("/api/ollama/pull", methods=["POST"])
def ollama_pull():
    """Pull/download an Ollama model. Streams progress."""
    data = request.json
    model_name = data.get("name", "")
    if not model_name:
        return jsonify(error="No model name provided"), 400

    def stream():
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/pull",
                json={"name": model_name, "stream": True},
                stream=True, timeout=3600,
            )
            for line in r.iter_lines():
                if line:
                    yield f"data: {line.decode()}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(stream()), mimetype="text/event-stream")


@app.route("/api/ollama/search")
def ollama_search():
    """Search for Ollama models from the library."""
    query = request.args.get("q", "")
    try:
        # Ollama doesn't have a public search API, so we use their website
        r = requests.get(
            f"https://ollama.com/search?q={query}",
            headers={"Accept": "application/json"},
            timeout=15,
        )
        # If JSON response
        if "application/json" in r.headers.get("content-type", ""):
            data = r.json()
            return jsonify(models=data.get("models", []))
        # Parse from HTML as fallback
        html = r.text
        models = []
        # Find model cards: look for /library/modelname patterns
        items = re.findall(
            r'href="/library/([^"]+)"[^>]*>.*?'
            r'(?:<span[^>]*>([^<]*)</span>)?',
            html, re.DOTALL
        )
        seen = set()
        for name, desc in items:
            name = name.strip().split("?")[0].rstrip("/")
            if name and name not in seen and "/" not in name:
                seen.add(name)
                models.append({"name": name, "description": desc.strip() if desc else ""})
        return jsonify(models=models[:20])
    except Exception:
        # Fallback: return a curated list filtered by query
        curated = [
            {"name": "llama3.2", "description": "Llama 3.2 - fast",
             "sizes": "1b, 3b"},
            {"name": "llama3.1", "description": "Llama 3.1 - quality",
             "sizes": "8b, 70b"},
            {"name": "mistral", "description": "Mistral 7B - general",
             "sizes": "7b"},
            {"name": "mixtral", "description": "Mixture of experts",
             "sizes": "8x7b"},
            {"name": "phi3", "description": "Phi-3 - efficient",
             "sizes": "mini, medium"},
            {"name": "gemma2", "description": "Gemma 2 - lightweight",
             "sizes": "2b, 9b, 27b"},
            {"name": "qwen2.5", "description": "Qwen 2.5 - multilingual",
             "sizes": "0.5b-72b"},
            {"name": "codellama", "description": "Code Llama - coding",
             "sizes": "7b, 13b, 34b"},
            {"name": "deepseek-coder", "description": "DeepSeek coding",
             "sizes": "1.3b, 6.7b, 33b"},
            {"name": "starcoder2", "description": "StarCoder 2 - code",
             "sizes": "3b, 7b, 15b"},
            {"name": "llava", "description": "Vision-language model",
             "sizes": "7b, 13b"},
            {"name": "neural-chat", "description": "Neural chat",
             "sizes": "7b"},
            {"name": "dolphin-mixtral", "description": "Dolphin Mixtral",
             "sizes": "8x7b"},
            {"name": "openhermes", "description": "OpenHermes - tuned",
             "sizes": "7b"},
            {"name": "vicuna", "description": "Vicuna - fine-tuned",
             "sizes": "7b, 13b"},
            {"name": "tinyllama", "description": "TinyLlama - fast",
             "sizes": "1.1b"},
            {"name": "stable-code", "description": "Stable code model",
             "sizes": "3b"},
            {"name": "nous-hermes2", "description": "Nous Hermes 2",
             "sizes": "7b, 34b"},
            {"name": "wizardcoder", "description": "WizardCoder",
             "sizes": "7b, 13b, 34b"},
            {"name": "samantha-mistral", "description": "Samantha AI",
             "sizes": "7b"},
        ]
        q = query.lower()
        if q:
            filtered = [
                m for m in curated
                if q in m["name"].lower()
                or q in m.get("description", "").lower()
            ]
        else:
            filtered = curated
        return jsonify(models=filtered)


@app.route("/api/ollama/delete", methods=["POST"])
def ollama_delete():
    """Delete an Ollama model."""
    data = request.json
    model_name = data.get("name", "")
    if not model_name:
        return jsonify(error="No model name provided"), 400
    try:
        r = requests.delete(f"{OLLAMA_URL}/api/delete", json={"name": model_name}, timeout=30)
        if r.status_code == 200:
            return jsonify(status="deleted")
        return jsonify(error=r.text), r.status_code
    except Exception as e:
        return jsonify(error=str(e)), 500


# ── Stable Diffusion model management ────────────────────────────────────────

@app.route("/api/sd/search")
def sd_search():
    """Search HuggingFace for SD models that have a usable checkpoint file."""
    query = request.args.get("q", "stable diffusion")
    diffusers_dirs = ("unet/", "vae/", "text_encoder/", "safety_checker/",
                      "scheduler/", "tokenizer/", "feature_extractor/")
    try:
        r = requests.get(
            "https://huggingface.co/api/models",
            params={
                "search": query,
                "filter": "diffusers",
                "sort": "downloads",
                "direction": "-1",
                "limit": 50,
                "expand[]": "siblings",
            },
            timeout=15,
        )
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        return jsonify(error=str(e), models=[]), 500

    models = []
    for m in raw:
        tags = m.get("tags", [])
        tags_lower = [t.lower() for t in tags]
        model_id_lower = m.get("id", "").lower()

        # Skip incompatible architectures and non-checkpoint model types
        # Check both tags and model ID since tags can be empty
        all_text = tags_lower + [model_id_lower]
        if any(t in tags_lower for t in ("lora", "template:diffusion-lora")):
            continue
        if "lora" in model_id_lower:
            continue
        if any("flux" in t for t in all_text):
            continue
        if any("sd3" in t or "stable-diffusion-3" in t for t in all_text):
            continue

        # Only include repos that have at least one root-level checkpoint
        siblings = m.get("siblings", [])
        has_checkpoint = any(
            s.get("rfilename", "").endswith((".safetensors", ".ckpt"))
            and not any(s["rfilename"].startswith(d) for d in diffusers_dirs)
            for s in siblings
        )
        if not has_checkpoint:
            continue

        # Classify architecture: SD 1.5 vs SDXL
        is_sdxl = any("xl" in t for t in tags_lower) or "xl" in m.get("id", "").lower()
        arch = "sdxl" if is_sdxl else "sd15"

        model_id = m.get("id", "")
        downloads = m.get("downloads", 0)
        likes = m.get("likes", 0)
        models.append({
            "id": model_id,
            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
            "author": model_id.split("/")[0] if "/" in model_id else "",
            "downloads": downloads,
            "likes": likes,
            "arch": arch,
            "tags": [t for t in tags[:5] if t not in ("diffusers", "arxiv:2112.10752")],
            "url": f"https://huggingface.co/{model_id}",
        })
    return jsonify(models=models)


@app.route("/api/sd/browse", methods=["POST"])
def sd_browse_repo():
    """List downloadable checkpoint files in a HuggingFace repo with labels."""
    data = request.json
    repo_id = data.get("repo_id", "")
    if not repo_id:
        return jsonify(error="No repo_id provided"), 400

    diffusers_dirs = ("unet/", "vae/", "text_encoder/", "safety_checker/",
                      "scheduler/", "tokenizer/", "feature_extractor/")
    try:
        # Use tree API to get file sizes
        r = requests.get(
            f"https://huggingface.co/api/models/{repo_id}/tree/main",
            timeout=15,
        )
        r.raise_for_status()
        tree = r.json()

        files = []
        for f in tree:
            fname = f.get("path", "")
            if not fname.endswith((".safetensors", ".ckpt")):
                continue
            if any(fname.startswith(d) for d in diffusers_dirs):
                continue
            size = f.get("lfs", {}).get("size", 0) or f.get("size", 0)
            name_lower = fname.lower()

            # Classify the file variant
            tags = []
            if "inpainting" in name_lower or "inpaint" in name_lower:
                tags.append("inpainting")
            if "fp16" in name_lower:
                tags.append("fp16")
            if "no-ema" in name_lower or "noema" in name_lower:
                tags.append("pruned")
            if fname.endswith(".ckpt"):
                tags.append("ckpt")

            # Score for ranking: prefer safetensors, full precision, non-inpainting
            score = 0
            if fname.endswith(".safetensors"):
                score += 4   # prefer safetensors format
            if "inpainting" not in name_lower:
                score += 2   # prefer standard over inpainting
            if "fp16" not in name_lower:
                score += 1   # prefer full precision for CPU

            files.append({
                "filename": fname,
                "size": size,
                "tags": tags,
                "score": score,
                "download_url": f"https://huggingface.co/{repo_id}/resolve/main/{fname}",
            })

        # Sort best first
        files.sort(key=lambda x: -x["score"])
        if files:
            files[0]["recommended"] = True

        return jsonify(repo_id=repo_id, files=files)
    except Exception as e:
        return jsonify(error=str(e), files=[]), 500


SD_MODELS_DIR = os.environ.get("SD_MODELS_DIR", "/opt/sd-models")

# Track download progress
download_state = {"active": False, "progress": "", "filename": ""}


@app.route("/api/sd/models")
def sd_models():
    """List installed SD models and the active one."""
    try:
        models_r = requests.get(f"{SD_URL}/sdapi/v1/sd-models", timeout=10)
        options_r = requests.get(f"{SD_URL}/sdapi/v1/options", timeout=10)
        models = [{"title": m["title"], "name": m["model_name"], "hash": m.get("hash", "")}
                  for m in models_r.json()]
        active = options_r.json().get("sd_model_checkpoint", "")
        return jsonify(models=models, active=active)
    except Exception as e:
        return jsonify(error=str(e), models=[], active=""), 500


@app.route("/api/sd/switch", methods=["POST"])
def sd_switch():
    """Switch the active SD model."""
    data = request.json
    title = data.get("title", "")
    if not title:
        return jsonify(error="No model title provided"), 400
    try:
        r = requests.post(
            f"{SD_URL}/sdapi/v1/options",
            json={"sd_model_checkpoint": title},
            timeout=120,
        )
        if r.status_code == 200:
            return jsonify(status="switched")
        return jsonify(error=r.text), r.status_code
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/sd/delete", methods=["POST"])
def sd_delete():
    """Delete an SD model file."""
    data = request.json
    filename = data.get("filename", "")
    if not filename or "/" in filename or "\\" in filename:
        return jsonify(error="Invalid filename"), 400
    path = os.path.join(SD_MODELS_DIR, filename)
    if not os.path.isfile(path):
        return jsonify(error="File not found"), 404
    try:
        os.remove(path)
        requests.post(f"{SD_URL}/sdapi/v1/refresh-checkpoints", timeout=10)
        return jsonify(status="deleted")
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/sd/download", methods=["POST"])
def sd_download():
    """Download an SD model from a URL into the models directory."""
    if download_state["active"]:
        return jsonify(error="Download already in progress"), 409

    data = request.json
    url = data.get("url", "")
    filename = data.get("filename", "")
    if not url:
        return jsonify(error="No URL provided"), 400

    if not filename:
        filename = url.split("/")[-1].split("?")[0]
        if not filename.endswith((".safetensors", ".ckpt")):
            filename += ".safetensors"

    def do_download():
        global download_state
        download_state = {"active": True, "progress": "Starting...", "filename": filename}
        dest = os.path.join(SD_MODELS_DIR, filename)
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = round(downloaded / total * 100, 1)
                        dl_mb = downloaded // 1048576
                        tot_mb = total // 1048576
                        download_state["progress"] = (
                            f"{pct}% ({dl_mb}MB / {tot_mb}MB)"
                        )
                    else:
                        download_state["progress"] = f"{downloaded // 1024 // 1024}MB downloaded"
            requests.post(f"{SD_URL}/sdapi/v1/refresh-checkpoints", timeout=10)
            download_state = {"active": False, "progress": "Complete", "filename": filename}
        except Exception as e:
            if os.path.exists(dest):
                os.remove(dest)
            download_state = {"active": False, "progress": f"Failed: {e}", "filename": filename}

    t = threading.Thread(target=do_download, daemon=True)
    t.start()
    return jsonify(status="started", filename=filename)


@app.route("/api/sd/samplers")
def sd_samplers():
    """List available SD samplers."""
    try:
        r = requests.get(f"{SD_URL}/sdapi/v1/samplers", timeout=10)
        samplers = [s["name"] for s in r.json()]
        return jsonify(samplers=samplers)
    except Exception:
        return jsonify(samplers=["Euler", "Euler a", "LMS", "DPM++ 2M", "DDIM", "UniPC"])


@app.route("/api/ollama/show", methods=["POST"])
def ollama_show():
    """Get detailed info about an Ollama model including its default parameters."""
    data = request.json
    model_name = data.get("name", "")
    if not model_name:
        return jsonify(error="No model name provided"), 400
    try:
        r = requests.post(f"{OLLAMA_URL}/api/show", json={"name": model_name}, timeout=10)
        info = r.json()
        details = info.get("details", {})
        # Parse PARAMETER lines from the parameters field
        params = {}
        for line in (info.get("parameters", "") or "").split("\n"):
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                key, val = parts
                try:
                    val = float(val) if "." in val else int(val)
                except ValueError:
                    pass
                if key in params:
                    # Some params repeat (e.g. stop tokens) — skip for UI
                    continue
                params[key] = val
        return jsonify(
            family=details.get("family", ""),
            parameter_size=details.get("parameter_size", ""),
            quantization=details.get("quantization_level", ""),
            context_length=params.get("num_ctx", 2048),
            defaults=params,
        )
    except Exception as e:
        return jsonify(error=str(e), defaults={}), 500


@app.route("/api/sd/download", methods=["GET"])
def sd_download_status():
    """Poll download progress."""
    return jsonify(download_state)


if __name__ == "__main__":
    # Apply tuned thread settings on startup if available
    tuned = load_tuned_params()
    if tuned and "best_threads" in tuned:
        threads = str(tuned["best_threads"])
        os.environ.setdefault("OMP_NUM_THREADS", threads)
        os.environ.setdefault("MKL_NUM_THREADS", threads)
        os.environ.setdefault("TORCH_NUM_THREADS", threads)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cert = os.path.join(base_dir, "cert.pem")
    key = os.path.join(base_dir, "key.pem")
    if os.path.exists(cert) and os.path.exists(key):
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        app.run(host="0.0.0.0", port=443, ssl_context=ctx, threaded=True)
    else:
        app.run(host="0.0.0.0", port=5005, threaded=True)
