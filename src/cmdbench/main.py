import argparse
import subprocess
import time
import psutil
import statistics
import sys
import threading
import os
import math # For ceil
import signal # For timeout handling
import shlex # For splitting command strings safely
import re # For sanitizing filenames
from typing import List, Dict, Any, Optional, Tuple

# Optional: Attempt to import pynvml for NVIDIA GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_GPU_SUPPORT = True
except (ImportError, pynvml.NVMLError):
    NVIDIA_GPU_SUPPORT = False

# Optional: Attempt to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend suitable for scripts
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d
    PLOTTING_SUPPORT = True
except ImportError:
    PLOTTING_SUPPORT = False
    # Define dummy classes/functions if plotting libs are missing
    class np: pass
    class interp1d: pass
    class plt: pass


# --- ResourceMonitor Class ---
# (ResourceMonitor class remains the same as in the previous version)
# ... (Keep the full ResourceMonitor class here) ...
class ResourceMonitor:
    """Monitors CPU, RAM, and optionally GPU usage of a process."""

    def __init__(self, pid: int, interval: float = 0.1, monitor_gpu: bool = False):
        self.pid = pid
        self.interval = interval
        self.monitor_gpu = monitor_gpu and NVIDIA_GPU_SUPPORT
        self._process = None
        self._thread = None
        self._stop_event = threading.Event()
        self.results: Dict[str, List[float]] = {
            "timestamps_rel_s": [], # Relative time in seconds since monitoring start
            "cpu_percent": [],
            "memory_rss_mb": [],
            "gpu_utilization_percent": [],
            "gpu_memory_used_mb": []
        }
        self.monitor_start_time: Optional[float] = None
        self.error: Optional[str] = None
        self.gpu_handles: List[Any] = [] # Store NVML device handles

        try:
            self._process = psutil.Process(self.pid)
            if not self._process.is_running():
                 raise psutil.NoSuchProcess(self.pid) # Process already gone

            if self.monitor_gpu:
                try:
                    device_count = pynvml.nvmlDeviceGetCount()
                    self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
                    if not self.gpu_handles:
                        print("Warning: NVML initialized but no GPU handles found.", file=sys.stderr)
                        self.monitor_gpu = False
                except pynvml.NVMLError as e:
                    print(f"Warning: Failed to get NVIDIA GPU handles: {e}", file=sys.stderr)
                    self.monitor_gpu = False

        except psutil.NoSuchProcess:
            self.error = f"Process with PID {self.pid} ended before monitoring could start."
            self._process = None
        except psutil.AccessDenied:
             self.error = f"Access denied when trying to access process {self.pid} for monitoring."
             self._process = None
        except Exception as e:
            self.error = f"Error initializing monitor for PID {self.pid}: {e}"
            self._process = None

    def _monitor_loop(self):
        """The actual monitoring loop run in a separate thread."""
        if not self._process:
            return

        self.monitor_start_time = time.perf_counter()

        while not self._stop_event.is_set():
            current_time = time.perf_counter()
            timestamp_rel = current_time - self.monitor_start_time

            try:
                if not self._process or not self._process.is_running():
                    break

                # --- CPU Usage ---
                cpu = self._process.cpu_percent(interval=None) # Non-blocking since last call
                cpu_normalized = cpu / psutil.cpu_count()

                # --- Memory Usage ---
                mem_info = self._process.memory_info()
                mem_rss_mb = mem_info.rss / (1024 * 1024)

                # --- GPU Usage (NVIDIA only) ---
                gpu_util = 0.0
                gpu_mem_mb = 0.0
                if self.monitor_gpu and self.gpu_handles:
                    try:
                        total_gpu_util = 0.0
                        total_gpu_mem_used = 0.0
                        for handle in self.gpu_handles:
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            total_gpu_util += util.gpu
                            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            total_gpu_mem_used += mem.used

                        gpu_util = total_gpu_util / len(self.gpu_handles) if self.gpu_handles else 0
                        gpu_mem_mb = (total_gpu_mem_used / len(self.gpu_handles) / (1024 * 1024)) if self.gpu_handles else 0

                    except pynvml.NVMLError as e:
                        if "last_nvml_error" not in self.__dict__ or self.last_nvml_error != str(e):
                             print(f"Warning: NVML Error during monitoring: {e}", file=sys.stderr)
                             self.last_nvml_error = str(e)
                        gpu_util = 0.0
                        gpu_mem_mb = 0.0

                # --- Append results ---
                self.results["timestamps_rel_s"].append(timestamp_rel)
                self.results["cpu_percent"].append(cpu_normalized)
                self.results["memory_rss_mb"].append(mem_rss_mb)
                self.results["gpu_utilization_percent"].append(gpu_util)
                self.results["gpu_memory_used_mb"].append(gpu_mem_mb)

            except psutil.NoSuchProcess:
                break
            except psutil.AccessDenied:
                self.error = "Access denied while monitoring process resources."
                break
            except Exception as e:
                self.error = f"Unexpected error during monitoring: {e}"
                break

            # Calculate time to sleep until the next interval
            elapsed_in_loop = time.perf_counter() - current_time
            sleep_time = max(0, self.interval - elapsed_in_loop)
            self._stop_event.wait(sleep_time)

    def start(self):
        """Starts the monitoring thread."""
        if self.error:
             print(f"Error: Cannot start monitor: {self.error}", file=sys.stderr)
             return False
        if not self._process:
             print("Error: Cannot start monitor, process object not initialized.", file=sys.stderr)
             return False

        self.results = {key: [] for key in self.results}
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
        """Stops the monitoring thread and returns aggregated summary and raw results."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 2 + 1) # Add timeout for join

        summary = {}
        for key, values in self.results.items():
            if key == "timestamps_rel_s": continue

            valid_values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]

            if valid_values:
                summary[f"{key}_avg"] = statistics.mean(valid_values)
                summary[f"{key}_max"] = max(valid_values)
                summary[f"{key}_min"] = min(valid_values)
            else:
                 is_gpu_metric = key.startswith("gpu_")
                 gpu_should_have_data = self.monitor_gpu and self.gpu_handles
                 if is_gpu_metric and not gpu_should_have_data:
                     summary[f"{key}_avg"] = 0.0
                     summary[f"{key}_max"] = 0.0
                     summary[f"{key}_min"] = 0.0
                 else:
                     summary[f"{key}_avg"] = float('nan')
                     summary[f"{key}_max"] = float('nan')
                     summary[f"{key}_min"] = float('nan')

        if self.error:
            summary["monitoring_error"] = self.error

        return summary, self.results


# --- Plotting Functions ---
def sanitize_filename(name: str, max_len: int = 50) -> str:
    """Basic sanitization for filenames derived from commands."""
    # Remove potentially problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*\s]+', '_', name)
    # Limit length
    return sanitized[:max_len]

# Updated plotting functions to accept command_index and command_str
def generate_plot(command_index: int, run_index: int, total_runs: int, timeseries_data: Dict[str, List[float]], output_filename: str, command_str: str):
    """Generates a plot of resource usage over time for a single run."""
    if not PLOTTING_SUPPORT: return

    timestamps = timeseries_data.get("timestamps_rel_s", [])
    if not timestamps or len(timestamps) < 2:
        print(f"Warning: [Cmd {command_index+1}] Not enough data points for Run {run_index} to generate plot.", file=sys.stderr)
        return

    # ... [Rest of plotting logic is the same, but update title] ...
    cpu_usage = timeseries_data.get("cpu_percent", [])
    mem_usage = timeseries_data.get("memory_rss_mb", [])
    gpu_util = timeseries_data.get("gpu_utilization_percent", [])
    gpu_mem = timeseries_data.get("gpu_memory_used_mb", [])
    has_meaningful_gpu_data = (gpu_util or gpu_mem) and (any(v > 0 for v in gpu_util if isinstance(v, (int, float))) or any(v > 0 for v in gpu_mem if isinstance(v, (int, float))))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    title_str = f"Cmd {command_index+1} - Run {run_index}/{total_runs}: Resource Usage\n`{command_str[:80]}{'...' if len(command_str)>80 else ''}`"
    fig.suptitle(title_str, fontsize=11) # Smaller font size
    ax1.set_xlabel("Time (s)")

    color_cpu = 'tab:red'
    ax1.set_ylabel("CPU / GPU Utilization (%)", color=color_cpu)
    l1 = ax1.plot(timestamps, cpu_usage, color=color_cpu, label="CPU Usage (%)")
    if has_meaningful_gpu_data:
        color_gpu_util = 'tab:purple'
        l2 = ax1.plot(timestamps, gpu_util, color=color_gpu_util, linestyle='--', label="Avg GPU Utilization (%)")
    ax1.tick_params(axis='y', labelcolor=color_cpu)
    max_y1_val = 0
    if cpu_usage: max_y1_val = max(max_y1_val, max(cpu_usage))
    if has_meaningful_gpu_data and gpu_util: max_y1_val = max(max_y1_val, max(gpu_util))
    max_y1 = max(105, math.ceil(max_y1_val / 10.0) * 10 if max_y1_val > 0 else 105)
    ax1.set_ylim(bottom=0, top=max_y1)


    color_mem = 'tab:blue'
    ax2.set_ylabel("Memory Usage (MB)", color=color_mem)
    l3 = ax2.plot(timestamps, mem_usage, color=color_mem, label="RAM RSS (MB)")
    if has_meaningful_gpu_data:
         color_gpu_mem = 'tab:cyan'
         l4 = ax2.plot(timestamps, gpu_mem, color=color_gpu_mem, linestyle='--', label="Avg GPU Memory Used (MB)")
    ax2.tick_params(axis='y', labelcolor=color_mem)
    max_y2_val = 0
    if mem_usage: max_y2_val = max(max_y2_val, max(mem_usage))
    if has_meaningful_gpu_data and gpu_mem: max_y2_val = max(max_y2_val, max(gpu_mem))
    max_y2 = max(50, math.ceil(max_y2_val / 50.0) * 50 if max_y2_val > 0 else 50)
    ax2.set_ylim(bottom=0, top=max_y2)

    lines = l1
    if has_meaningful_gpu_data: lines += l2
    lines += l3
    if has_meaningful_gpu_data: lines += l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(len(labels), 4))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    try:
        plt.savefig(output_filename, dpi=100)
        print(f"Plot saved to: {output_filename}", file=sys.stderr)
    except Exception as e:
        print(f"Error saving plot {output_filename}: {e}", file=sys.stderr)
    finally:
        plt.close(fig)


def generate_average_plot(command_index: int, run_results: List[Dict], output_filename: str, command_str: str):
    """Generates a plot showing the average resource usage across all runs for a specific command."""
    if not PLOTTING_SUPPORT: return

    valid_runs_data = [r for r in run_results if r.get('timeseries') and len(r['timeseries'].get('timestamps_rel_s', [])) >= 2]
    if not valid_runs_data:
        print(f"Warning: [Cmd {command_index+1}] No valid run data for average plot.", file=sys.stderr)
        return

    num_runs = len(valid_runs_data)
    print(f"\n[Cmd {command_index+1}] Generating average plot from {num_runs} successful run(s)...", file=sys.stderr)

    # ... [Interpolation logic remains the same] ...
    max_duration = 0
    for r in valid_runs_data:
        ts = r['timeseries']['timestamps_rel_s']
        if ts: max_duration = max(max_duration, ts[-1])
    if max_duration == 0: return

    num_points = 200
    common_time = np.linspace(0, max_duration, num_points)
    interpolated_data = { k: [] for k in ["cpu_percent", "memory_rss_mb", "gpu_utilization_percent", "gpu_memory_used_mb"] }
    metrics_to_plot = list(interpolated_data.keys())
    has_any_gpu_data = False

    for run_data in valid_runs_data:
        timeseries = run_data['timeseries']
        timestamps = timeseries['timestamps_rel_s']
        run_has_gpu = any(v > 0 for v in timeseries.get('gpu_utilization_percent', []) if isinstance(v, (int, float))) or \
                      any(v > 0 for v in timeseries.get('gpu_memory_used_mb', []) if isinstance(v, (int, float)))
        if run_has_gpu: has_any_gpu_data = True

        for metric in metrics_to_plot:
            values = timeseries.get(metric, [])
            if len(timestamps) != len(values): continue
            try:
                 interp_func = interp1d(timestamps, values, kind='linear', bounds_error=False, fill_value=0)
                 interpolated_values = interp_func(common_time)
                 interpolated_data[metric].append(interpolated_values)
            except ValueError as e:
                 print(f"Warning: [Cmd {command_index+1}] Interpolation error for '{metric}': {e}. Skipping for average plot.", file=sys.stderr)

    averages = {}
    std_devs = {}
    valid_metrics = []
    for metric, data_list in interpolated_data.items():
        if len(data_list) == num_runs:
            stacked_data = np.vstack(data_list)
            averages[metric] = np.mean(stacked_data, axis=0)
            std_devs[metric] = np.std(stacked_data, axis=0)
            valid_metrics.append(metric)
        elif data_list: # Inconsistent data across runs
             print(f"Warning: [Cmd {command_index+1}] Metric '{metric}' only has data from {len(data_list)}/{num_runs} runs for avg plot.", file=sys.stderr)


    # ... [Plotting averages logic is the same, but update title] ...
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    title_str = f"Cmd {command_index+1} - Average Resource Usage ({num_runs} Runs)\n`{command_str[:80]}{'...' if len(command_str)>80 else ''}`"
    fig.suptitle(title_str, fontsize=11)
    ax1.set_xlabel("Time (s)")

    lines = []
    max_y1_val = 0
    color_cpu = 'tab:red'
    ax1.set_ylabel("Avg CPU / GPU Utilization (%)", color=color_cpu)
    if 'cpu_percent' in valid_metrics:
        avg_cpu = averages['cpu_percent']
        std_cpu = std_devs['cpu_percent']
        l1 = ax1.plot(common_time, avg_cpu, color=color_cpu, label="Avg CPU Usage (%)")
        ax1.fill_between(common_time, avg_cpu - std_cpu, avg_cpu + std_cpu, color=color_cpu, alpha=0.2, label='_nolegend_')
        lines.extend(l1)
        max_y1_val = max(max_y1_val, np.max(avg_cpu + std_cpu))

    color_gpu_util = 'tab:purple'
    if has_any_gpu_data and 'gpu_utilization_percent' in valid_metrics:
        avg_gpu_util = averages['gpu_utilization_percent']
        std_gpu_util = std_devs['gpu_utilization_percent']
        l2 = ax1.plot(common_time, avg_gpu_util, color=color_gpu_util, linestyle='--', label="Avg GPU Utilization (%)")
        ax1.fill_between(common_time, avg_gpu_util - std_gpu_util, avg_gpu_util + std_gpu_util, color=color_gpu_util, alpha=0.2, label='_nolegend_')
        lines.extend(l2)
        max_y1_val = max(max_y1_val, np.max(avg_gpu_util + std_gpu_util))

    ax1.tick_params(axis='y', labelcolor=color_cpu)
    ax1.set_ylim(bottom=0, top=max(105, math.ceil(max_y1_val / 10.0) * 10 if max_y1_val > 0 else 105))

    max_y2_val = 0
    color_mem = 'tab:blue'
    ax2.set_ylabel("Avg Memory Usage (MB)", color=color_mem)
    if 'memory_rss_mb' in valid_metrics:
        avg_mem = averages['memory_rss_mb']
        std_mem = std_devs['memory_rss_mb']
        l3 = ax2.plot(common_time, avg_mem, color=color_mem, label="Avg RAM RSS (MB)")
        ax2.fill_between(common_time, avg_mem - std_mem, avg_mem + std_mem, color=color_mem, alpha=0.2, label='_nolegend_')
        lines.extend(l3)
        max_y2_val = max(max_y2_val, np.max(avg_mem + std_mem))

    color_gpu_mem = 'tab:cyan'
    if has_any_gpu_data and 'gpu_memory_used_mb' in valid_metrics:
         avg_gpu_mem = averages['gpu_memory_used_mb']
         std_gpu_mem = std_devs['gpu_memory_used_mb']
         l4 = ax2.plot(common_time, avg_gpu_mem, color=color_gpu_mem, linestyle='--', label="Avg GPU Memory Used (MB)")
         ax2.fill_between(common_time, avg_gpu_mem - std_gpu_mem, avg_gpu_mem + std_gpu_mem, color=color_gpu_mem, alpha=0.2, label='_nolegend_')
         lines.extend(l4)
         max_y2_val = max(max_y2_val, np.max(avg_gpu_mem + std_gpu_mem))

    ax2.tick_params(axis='y', labelcolor=color_mem)
    ax2.set_ylim(bottom=0, top=max(50, math.ceil(max_y2_val / 50.0) * 50 if max_y2_val > 0 else 50))

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(len(labels), 4))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    try:
        plt.savefig(output_filename, dpi=100)
        print(f"Average plot saved to: {output_filename}", file=sys.stderr)
    except Exception as e:
        print(f"Error saving average plot {output_filename}: {e}", file=sys.stderr)
    finally:
        plt.close(fig)


# --- Utility Functions ---
# (format_duration remains the same)
def format_duration(seconds: float) -> str:
    """Formats duration in a human-readable way."""
    if seconds < 0 or math.isnan(seconds): return "N/A"
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.3f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.3f}s"


# (run_command remains mostly the same, but takes List[str] directly)
def run_command(command_list: List[str], # Takes list directly now
                command_str: str, # Keep string for printing
                run_prefix: str, # Pass prefix for consistent logging
                monitor_interval: float = 0.1,
                monitor_gpu: bool = False,
                timeout: Optional[float] = None,
                verbose_level: int = 0
                ) -> Tuple[Optional[float], Optional[Dict[str, Any]], Optional[Dict[str, List[float]]], Optional[int]]:
    """
    Runs the command once and monitors it.
    Returns (duration, resource_summary, raw_timeseries, return_code).
    """
    print(f"{run_prefix} Executing: {command_str}", file=sys.stderr)

    monitor = None
    resource_summary = None
    raw_timeseries = None
    return_code = None
    duration = None
    timed_out = False
    stdout = ""
    stderr = ""
    process = None # Initialize process to None

    try:
        start_time = time.perf_counter()
        preexec_fn = os.setsid if sys.platform != "win32" else None
        process = subprocess.Popen(
            command_list, # Use the list here
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=preexec_fn,
            bufsize=1 # Line buffered
        )

        time.sleep(0.02) # Short delay

        monitor = ResourceMonitor(process.pid, interval=monitor_interval, monitor_gpu=monitor_gpu)
        monitor_started = monitor.start()

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
             timed_out = True
             print(f"\n{run_prefix} Command timed out after {timeout} seconds. Terminating process group...", file=sys.stderr)
             pgid = -1
             try:
                 if sys.platform != "win32":
                     pgid = os.getpgid(process.pid)
                     os.killpg(pgid, signal.SIGTERM)
                 else:
                     subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], check=False, capture_output=True)
             except Exception as term_err: pass # Ignore termination errors initially
             try: process.wait(timeout=1.5)
             except subprocess.TimeoutExpired:
                 print(f"{run_prefix} Force killing...", file=sys.stderr)
                 try:
                     if sys.platform != "win32" and pgid > 0: os.killpg(pgid, signal.SIGKILL)
                     else: process.kill()
                 except Exception: pass # Ignore kill errors
             stdout, stderr = process.communicate() # Get remaining output
             return_code = -9 # Timeout code
             duration = timeout

        if duration is None:
             end_time = time.perf_counter()
             duration = end_time - start_time

        # Stop monitoring
        if monitor_started:
             resource_summary, raw_timeseries = monitor.stop()
        elif monitor:
             resource_summary = {"monitoring_error": monitor.error if monitor.error else "Monitor failed to start."}
             raw_timeseries = monitor.results
        else:
             resource_summary = {"monitoring_error": "Process finished too quickly or monitor failed initialization."}
             raw_timeseries = {key: [] for key in ResourceMonitor(0).results}

        # Verbose output
        if verbose_level > 1:
             print(f"{run_prefix} --- STDOUT ({len(stdout)} bytes) ---", file=sys.stderr)
             print(stdout, file=sys.stderr)
             print(f"{run_prefix} --- STDERR ({len(stderr)} bytes) ---", file=sys.stderr)
             print(stderr, file=sys.stderr)
             print(f"{run_prefix} --- End Output ---", file=sys.stderr)
        elif verbose_level > 0 and (stdout or stderr):
             print(f"{run_prefix} Output captured ({len(stdout)} bytes stdout, {len(stderr)} bytes stderr). Use -vv to view.", file=sys.stderr)

        if return_code != 0 and not timed_out:
            print(f"{run_prefix} Warning: Command exited with non-zero status: {return_code}", file=sys.stderr)

    except FileNotFoundError:
        print(f"{run_prefix} Error: Command not found: {command_list[0]}", file=sys.stderr)
        return None, None, None, None
    except PermissionError:
         print(f"{run_prefix} Error: Permission denied executing: {command_list[0]}", file=sys.stderr)
         return None, None, None, None
    except Exception as e:
        print(f"{run_prefix} Error running or monitoring command: {e}", file=sys.stderr)
        if process and process.poll() is None:
             try: process.kill()
             except Exception: pass
        if monitor_started:
             try: monitor.stop()
             except Exception: pass
        return None, None, None, None

    return duration, resource_summary, raw_timeseries, return_code


def calculate_summary_stats(run_results: List[Dict], command_str: str) -> Dict[str, Any]:
    """Calculates summary statistics for a list of runs for one command."""
    summary_stats = {"command": command_str, "runs": len(run_results), "successful_runs": 0}
    if not run_results:
        return summary_stats

    valid_durations = [r['duration_s'] for r in run_results if r.get('duration_s') is not None]
    summary_stats["successful_runs"] = len(valid_durations)

    if not valid_durations:
        return summary_stats

    summary_stats["time_avg_s"] = statistics.mean(valid_durations)
    summary_stats["time_min_s"] = min(valid_durations)
    summary_stats["time_max_s"] = max(valid_durations)
    summary_stats["time_stdev_s"] = statistics.stdev(valid_durations) if len(valid_durations) > 1 else 0.0

    def aggregate_summary_stat(key):
        values = [r['summary'].get(key) for r in run_results if r.get('summary') and r['summary'].get(key) is not None and math.isfinite(r['summary'].get(key))]
        return statistics.mean(values) if values else float('nan')

    summary_stats["cpu_avg_avg"] = aggregate_summary_stat('cpu_percent_avg')
    summary_stats["cpu_max_avg"] = aggregate_summary_stat('cpu_percent_max')
    summary_stats["mem_avg_avg"] = aggregate_summary_stat('memory_rss_mb_avg')
    summary_stats["mem_max_avg"] = aggregate_summary_stat('memory_rss_mb_max')

    # Aggregate GPU stats if available in summaries
    summary_stats["gpu_util_avg_avg"] = aggregate_summary_stat('gpu_utilization_percent_avg')
    summary_stats["gpu_util_max_avg"] = aggregate_summary_stat('gpu_utilization_percent_max')
    summary_stats["gpu_mem_avg_avg"] = aggregate_summary_stat('gpu_memory_used_mb_avg')
    summary_stats["gpu_mem_max_avg"] = aggregate_summary_stat('gpu_memory_used_mb_max')

    return summary_stats


# --- Main Logic ---
def run_benchmark_logic(args):
    """Main logic function containing the benchmark execution flow."""
    command_strings = args.command # Now a list of strings
    num_commands = len(command_strings)

    print(f"Benchmarking {num_commands} command(s)...")
    print(f"Number of runs per command: {args.runs}")
    if args.warmup > 0: print(f"Warm-up runs per command: {args.warmup}")
    print(f"Monitoring interval: {args.interval}s")
    if args.timeout: print(f"Timeout per run: {args.timeout}s")

    # Plotting setup
    plot_dir = args.plot_dir
    if plot_dir:
        if not PLOTTING_SUPPORT:
            print("Warning: Plotting requested, but matplotlib/numpy/scipy not found. Plotting disabled.", file=sys.stderr)
            plot_dir = None
        else:
            try:
                os.makedirs(plot_dir, exist_ok=True)
                print(f"Plots will be saved to: {os.path.abspath(plot_dir)}")
            except OSError as e:
                print(f"Error creating plot directory '{plot_dir}': {e}. Plotting disabled.", file=sys.stderr)
                plot_dir = None

    # GPU Status
    gpu_status = "Disabled"
    gpu_enabled_arg = args.gpu
    if gpu_enabled_arg:
        if NVIDIA_GPU_SUPPORT:
            try:
                count = pynvml.nvmlDeviceGetCount()
                gpu_status = f"Enabled (NVIDIA, Found {count} GPU(s))"
            except pynvml.NVMLError as e:
                 gpu_status = f"Error enabling NVIDIA GPU monitoring: {e}"
        else:
            gpu_status = "Enabled attempt, but pynvml not found or failed to load."
    print(f"GPU Monitoring: {gpu_status}")


    all_results = {} # Store results per command string: {cmd_str: [run_result1, run_result2,...]}

    # --- Loop through each command ---
    for cmd_idx, command_str in enumerate(command_strings):
        print("-" * 20, file=sys.stderr)
        print(f"Benchmarking Command {cmd_idx+1}/{num_commands}: `{command_str}`", file=sys.stderr)

        # Use shlex to split command string safely for different OS
        try:
            command_list = shlex.split(command_str)
            if not command_list:
                 print(f"Warning: Command {cmd_idx+1} is empty. Skipping.", file=sys.stderr)
                 all_results[command_str] = [] # Store empty results
                 continue
        except ValueError as e:
            print(f"Error parsing command {cmd_idx+1}: {command_str}\n -> {e}. Skipping.", file=sys.stderr)
            all_results[command_str] = []
            continue


        current_command_results = []

        # --- Warm-up Runs (for this command) ---
        if args.warmup > 0:
            print(f"[Cmd {cmd_idx+1}] Starting {args.warmup} warm-up run(s)...", file=sys.stderr)
            for i in range(args.warmup):
                run_prefix = f"[Cmd {cmd_idx+1} Warmup {i+1}/{args.warmup}]"
                run_command(command_list, command_str, run_prefix,
                            monitor_interval=args.interval, monitor_gpu=gpu_enabled_arg,
                            timeout=args.timeout, verbose_level=args.verbose)
            print(f"[Cmd {cmd_idx+1}] Warm-up complete.", file=sys.stderr)
            print("-" * 10, file=sys.stderr)

        # --- Benchmark Runs (for this command) ---
        print(f"[Cmd {cmd_idx+1}] Starting {args.runs} benchmark run(s)...", file=sys.stderr)
        for i in range(args.runs):
            run_prefix = f"[Cmd {cmd_idx+1} Run {i+1}/{args.runs}]"
            duration, summary, timeseries, exit_code = run_command(
                command_list, command_str, run_prefix,
                monitor_interval=args.interval, monitor_gpu=gpu_enabled_arg,
                timeout=args.timeout, verbose_level=args.verbose
                )

            if duration is None or summary is None or timeseries is None or exit_code is None:
                 print(f"{run_prefix} Failed. Skipping results and plot for this run.", file=sys.stderr)
                 # Optionally store failure info? For now, just skip.
                 continue

            run_data = {
                "run": i + 1,
                "duration_s": duration,
                "exit_code": exit_code,
                "summary": summary,
                "timeseries": timeseries
            }
            current_command_results.append(run_data)

            # Print intermediate result summary for this run
            res_str = f"Time: {format_duration(duration)}, CPU Max: {summary.get('cpu_percent_max', 'N/A'):.1f}%, Mem Max: {summary.get('memory_rss_mb_max', 'N/A'):.1f} MB"
            gpu_max_util = summary.get('gpu_utilization_percent_max', float('nan'))
            gpu_max_mem = summary.get('gpu_memory_used_mb_max', float('nan'))
            if gpu_enabled_arg and NVIDIA_GPU_SUPPORT and not math.isnan(gpu_max_util):
                 res_str += f", GPU Max: {gpu_max_util:.1f}%, GPU Mem Max: {gpu_max_mem:.1f} MB"
            if summary.get("monitoring_error"):
                 res_str += f" (Monitor Warn: {summary['monitoring_error']})"
            print(f"{run_prefix} {res_str}")


            # Generate plot for this individual run if requested
            if plot_dir:
                if timeseries and len(timeseries.get('timestamps_rel_s', [])) >= 2:
                     # Use command index and run index for unique filenames
                     sanitized_cmd_part = sanitize_filename(f"cmd_{cmd_idx+1}")
                     plot_filename = os.path.join(plot_dir, f"{sanitized_cmd_part}_run_{i+1:03d}_resources.png")
                     generate_plot(cmd_idx, i + 1, args.runs, timeseries, plot_filename, command_str)
                elif timeseries:
                     print(f"{run_prefix} Skipping plot due to insufficient timeseries data.", file=sys.stderr)

        # Store results for this command
        all_results[command_str] = current_command_results

        # Generate average plot for this command if requested
        if plot_dir and current_command_results:
             sanitized_cmd_part = sanitize_filename(f"cmd_{cmd_idx+1}")
             avg_plot_filename = os.path.join(plot_dir, f"{sanitized_cmd_part}_average_resources.png")
             generate_average_plot(cmd_idx, current_command_results, avg_plot_filename, command_str)

    # --- End of loop through commands ---

    print("=" * 30, file=sys.stderr)
    print("Benchmark Complete.", file=sys.stderr)

    # --- Aggregate and Print Final Comparison Summary ---
    summarized_results = []
    for command_str, results_list in all_results.items():
        if results_list: # Only summarize if there were successful runs
            stats = calculate_summary_stats(results_list, command_str)
            summarized_results.append(stats)
        else:
             # Include a basic entry for commands that failed completely
             summarized_results.append({"command": command_str, "runs": 0, "successful_runs": 0})

    if not summarized_results:
        print("\nNo commands were successfully benchmarked.")
        # Cleanup NVML
        if NVIDIA_GPU_SUPPORT and gpu_enabled_arg: pynvml.nvmlShutdown()
        sys.exit(1)

    # Sort results by average time (fastest first) for comparison
    summarized_results.sort(key=lambda x: x.get("time_avg_s", float('inf')))

    print("\n--- Comparison Summary ---")

    # Find the fastest average time for relative comparison
    fastest_avg_time = min(r.get("time_avg_s", float('inf')) for r in summarized_results if r.get("successful_runs", 0) > 0)
    if fastest_avg_time == float('inf'): fastest_avg_time = None # Handle case where no runs succeeded

    # Determine max command length for formatting
    max_cmd_len = max(len(r['command']) for r in summarized_results)
    max_cmd_len = min(max_cmd_len, 60) # Cap length for display

    # Header
    header = f"{'Command':<{max_cmd_len}} | {'Avg Time':>12} | {'Relative':>10} | {'Min Time':>12} | {'Max Time':>12} | {'Avg Max Mem':>12} | {'Avg Max CPU':>12}"
    gpu_header = f" | {'Avg Max GPU %':>14} | {'Avg Max GPU Mem':>15}" if gpu_enabled_arg and NVIDIA_GPU_SUPPORT else ""
    print(header + gpu_header)
    print("-" * len(header + gpu_header))

    # Data Rows
    for stats in summarized_results:
        cmd_display = stats['command'][:max_cmd_len] + ('...' if len(stats['command']) > max_cmd_len else '')
        if stats.get("successful_runs", 0) == 0:
            print(f"{cmd_display:<{max_cmd_len}} | {'-'*12} | {'-'*10} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12}" + (f" | {'-'*14} | {'-'*15}" if gpu_enabled_arg and NVIDIA_GPU_SUPPORT else ""))
            continue

        avg_time_s = stats.get('time_avg_s', float('nan'))
        min_time_s = stats.get('time_min_s', float('nan'))
        max_time_s = stats.get('time_max_s', float('nan'))
        avg_max_mem = stats.get('mem_max_avg', float('nan'))
        avg_max_cpu = stats.get('cpu_max_avg', float('nan'))

        relative_speed = f"{(avg_time_s / fastest_avg_time):.2f}x" if fastest_avg_time and not math.isnan(avg_time_s) else "-"

        time_str = format_duration(avg_time_s)
        min_time_str = format_duration(min_time_s)
        max_time_str = format_duration(max_time_s)
        mem_str = f"{avg_max_mem:.1f} MB" if not math.isnan(avg_max_mem) else "-"
        cpu_str = f"{avg_max_cpu:.1f}%" if not math.isnan(avg_max_cpu) else "-"

        row = f"{cmd_display:<{max_cmd_len}} | {time_str:>12} | {relative_speed:>10} | {min_time_str:>12} | {max_time_str:>12} | {mem_str:>12} | {cpu_str:>12}"

        if gpu_enabled_arg and NVIDIA_GPU_SUPPORT:
            avg_max_gpu_util = stats.get('gpu_util_max_avg', float('nan'))
            avg_max_gpu_mem = stats.get('gpu_mem_max_avg', float('nan'))
            gpu_util_str = f"{avg_max_gpu_util:.1f}%" if not math.isnan(avg_max_gpu_util) else "-"
            gpu_mem_str = f"{avg_max_gpu_mem:.1f} MB" if not math.isnan(avg_max_gpu_mem) else "-"
            row += f" | {gpu_util_str:>14} | {gpu_mem_str:>15}"

        print(row)


    if plot_dir:
        print(f"\nPlots for each run and command average saved in: {os.path.abspath(plot_dir)}")

    # Cleanup NVML
    if NVIDIA_GPU_SUPPORT and gpu_enabled_arg:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass # Ignore shutdown errors


# --- Entry Point ---
def entry_point():
    """Handles argument parsing and calls the main benchmark logic."""
    parser = argparse.ArgumentParser(
        description="Benchmark and compare command-line commands for execution time and resource usage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # Updated epilog example
        epilog="Example: cmdbench --plot-dir ./plots -n 5 --gpu -- 'sleep 1' 'echo hello && sleep 0.5'"
    )
    # --- General Options ---
    parser.add_argument(
        "-n", "--runs", type=int, default=3,
        help="Number of times to run EACH command for benchmarking."
    )
    parser.add_argument(
        "-w", "--warmup", type=int, default=1,
        help="Number of warm-up runs to perform before benchmarking EACH command."
    )
    parser.add_argument(
        "-i", "--interval", type=float, default=0.1,
        help="Resource monitoring interval in seconds."
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Set a timeout in seconds for EACH command run."
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity. -v shows run output hints, -vv shows full stdout/stderr."
    )

    # --- Resource Monitoring ---
    parser.add_argument(
        "--gpu", action="store_true",
        help="Enable NVIDIA GPU monitoring (requires 'pip install cmdbench[gpu]' and NVIDIA driver)."
    )

    # --- Plotting ---
    parser.add_argument(
        "--plot-dir", type=str, default=None,
        help="Directory to save resource usage plots (individual runs + average per command)."
    )

    # --- Command(s) to Run ---
    parser.add_argument(
        "command", # Changed from nargs=REMAINDER
        nargs='+', # Expect one or more command strings
        metavar='COMMAND', # Help text hint
        help="One or more command strings to benchmark and compare. Enclose commands with spaces in quotes."
    )

    args = parser.parse_args()

    # Check plotting dependencies if plotting is requested
    if args.plot_dir and not PLOTTING_SUPPORT:
         print("Error: Plotting requested (--plot-dir), but matplotlib/numpy/scipy are not installed correctly.", file=sys.stderr)
         print("Please ensure these libraries are installed in your environment.", file=sys.stderr)
         sys.exit(1)

    run_benchmark_logic(args)


# --- No if __name__ == "__main__": block needed when using entry_point ---
