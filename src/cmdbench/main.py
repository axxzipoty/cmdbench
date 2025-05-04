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
    # Define dummy classes/functions if plotting libs are missing to avoid NameErrors later
    class np: pass
    class interp1d: pass
    class plt: pass


# --- Monitoring Thread ---
# (ResourceMonitor class remains the same as in the previous version)
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
        # Call cpu_percent once outside loop if interval=0 to establish baseline
        # self._process.cpu_percent(interval=None)

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
                        gpu_util = 0.0 # Or NaN? Or previous value? 0 seems reasonable.
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

        # Aggregate results for summary
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

def generate_plot(run_index: int, total_runs: int, timeseries_data: Dict[str, List[float]], output_filename: str, command_str: str):
    """Generates a plot of resource usage over time for a single run."""
    if not PLOTTING_SUPPORT:
        # This check should ideally happen before calling, but double-check
        return

    timestamps = timeseries_data.get("timestamps_rel_s", [])
    if not timestamps or len(timestamps) < 2:
        print(f"Warning: Not enough data points for Run {run_index} to generate plot.", file=sys.stderr)
        return

    cpu_usage = timeseries_data.get("cpu_percent", [])
    mem_usage = timeseries_data.get("memory_rss_mb", [])
    gpu_util = timeseries_data.get("gpu_utilization_percent", [])
    gpu_mem = timeseries_data.get("gpu_memory_used_mb", [])

    # Check if GPU data exists and has non-zero values (optional check)
    has_meaningful_gpu_data = (gpu_util or gpu_mem) and (any(v > 0 for v in gpu_util if isinstance(v, (int, float))) or any(v > 0 for v in gpu_mem if isinstance(v, (int, float))))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    title_str = f"Resource Usage: Run {run_index}/{total_runs}\nCommand: {command_str[:80]}{'...' if len(command_str)>80 else ''}"
    fig.suptitle(title_str, fontsize=12)
    ax1.set_xlabel("Time (s)")

    color_cpu = 'tab:red'
    ax1.set_ylabel("CPU / GPU Utilization (%)", color=color_cpu)
    l1 = ax1.plot(timestamps, cpu_usage, color=color_cpu, label="CPU Usage (%)")
    if has_meaningful_gpu_data:
        color_gpu_util = 'tab:purple'
        l2 = ax1.plot(timestamps, gpu_util, color=color_gpu_util, linestyle='--', label="Avg GPU Utilization (%)")
    ax1.tick_params(axis='y', labelcolor=color_cpu)
    # Dynamic Y axis, ensuring 0 is visible and top has some margin
    max_y1 = max(105, math.ceil(max(cpu_usage + (gpu_util if has_meaningful_gpu_data else [])) / 10.0) * 10 if cpu_usage or has_meaningful_gpu_data else 105)
    ax1.set_ylim(bottom=0, top=max_y1)

    color_mem = 'tab:blue'
    ax2.set_ylabel("Memory Usage (MB)", color=color_mem)
    l3 = ax2.plot(timestamps, mem_usage, color=color_mem, label="RAM RSS (MB)")
    if has_meaningful_gpu_data:
         color_gpu_mem = 'tab:cyan'
         l4 = ax2.plot(timestamps, gpu_mem, color=color_gpu_mem, linestyle='--', label="Avg GPU Memory Used (MB)")
    ax2.tick_params(axis='y', labelcolor=color_mem)
    # Dynamic Y axis, ensuring 0 is visible and top has some margin
    max_y2 = max(50, math.ceil(max(mem_usage + (gpu_mem if has_meaningful_gpu_data else [])) / 50.0) * 50 if mem_usage or has_meaningful_gpu_data else 50)
    ax2.set_ylim(bottom=0, top=max_y2)


    # Combine legends
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

def generate_average_plot(run_results: List[Dict], output_filename: str, command_str: str):
    """Generates a plot showing the average resource usage across all runs."""
    if not PLOTTING_SUPPORT:
        return # Should have been checked earlier

    valid_runs_data = [r for r in run_results if r.get('timeseries') and len(r['timeseries'].get('timestamps_rel_s', [])) >= 2]

    if not valid_runs_data:
        print("Warning: No valid run data with sufficient timeseries to generate average plot.", file=sys.stderr)
        return

    num_runs = len(valid_runs_data)
    print(f"\nGenerating average plot from {num_runs} successful run(s)...", file=sys.stderr)

    # Determine the common time axis based on the longest run duration
    max_duration = 0
    for r in valid_runs_data:
        ts = r['timeseries']['timestamps_rel_s']
        if ts:
            max_duration = max(max_duration, ts[-1])

    if max_duration == 0:
        print("Warning: Max duration is 0, cannot generate average plot.", file=sys.stderr)
        return

    # Define common time points for interpolation (e.g., 200 points)
    num_points = 200
    common_time = np.linspace(0, max_duration, num_points)

    # Store interpolated data for each metric from all runs
    interpolated_data = {
        "cpu_percent": [],
        "memory_rss_mb": [],
        "gpu_utilization_percent": [],
        "gpu_memory_used_mb": []
    }
    metrics_to_plot = list(interpolated_data.keys())
    has_any_gpu_data = False # Track if any run had meaningful GPU data

    for run_data in valid_runs_data:
        timeseries = run_data['timeseries']
        timestamps = timeseries['timestamps_rel_s']
        run_has_gpu = any(v > 0 for v in timeseries.get('gpu_utilization_percent', []) if isinstance(v, (int, float))) or \
                      any(v > 0 for v in timeseries.get('gpu_memory_used_mb', []) if isinstance(v, (int, float)))
        if run_has_gpu:
            has_any_gpu_data = True

        for metric in metrics_to_plot:
            values = timeseries.get(metric, [])
            if len(timestamps) != len(values):
                 print(f"Warning: Mismatch length ts({len(timestamps)}) vs {metric}({len(values)}) in a run. Skipping metric for average plot.", file=sys.stderr)
                 # Pad interpolated with NaNs or handle differently? For now, just skip appending bad data.
                 continue

            # Create interpolation function for this run's metric
            # Use linear interpolation, fill with 0 beyond run's end time
            try:
                 interp_func = interp1d(timestamps, values, kind='linear', bounds_error=False, fill_value=0)
                 # Evaluate interpolation function on the common time axis
                 interpolated_values = interp_func(common_time)
                 interpolated_data[metric].append(interpolated_values)
            except ValueError as e:
                 print(f"Warning: Interpolation error for metric '{metric}' in a run: {e}. Skipping.", file=sys.stderr)
                 # Append NaNs to keep array sizes consistent if needed, or filter later
                 # interpolated_data[metric].append(np.full(common_time.shape, np.nan))


    # Calculate average and standard deviation for each metric
    averages = {}
    std_devs = {}
    valid_metrics = []

    for metric, data_list in interpolated_data.items():
        if len(data_list) == num_runs: # Ensure all runs contributed data for this metric
            stacked_data = np.vstack(data_list)
            averages[metric] = np.mean(stacked_data, axis=0)
            std_devs[metric] = np.std(stacked_data, axis=0)
            valid_metrics.append(metric)
        elif data_list: # Some runs contributed, but not all (due to errors)
             print(f"Warning: Metric '{metric}' only has data from {len(data_list)}/{num_runs} runs. Average might be skewed.", file=sys.stderr)
             # Could still average, but be cautious:
             # stacked_data = np.vstack(data_list)
             # averages[metric] = np.nanmean(stacked_data, axis=0) # Use nanmean if NaNs were inserted
             # std_devs[metric] = np.nanstd(stacked_data, axis=0)
             # valid_metrics.append(metric)
        else:
             print(f"Info: No data collected for metric '{metric}' across runs for average plot.", file=sys.stderr)


    # Plotting the averages
    fig, ax1 = plt.subplots(figsize=(12, 7)) # Slightly taller for std dev bands
    ax2 = ax1.twinx()

    title_str = f"Average Resource Usage ({num_runs} Runs)\nCommand: {command_str[:80]}{'...' if len(command_str)>80 else ''}"
    fig.suptitle(title_str, fontsize=12)
    ax1.set_xlabel("Time (s)")

    lines = [] # Collect lines for legend

    # Plot CPU and GPU Utilization (Avg +/- Std Dev) on ax1
    color_cpu = 'tab:red'
    ax1.set_ylabel("Avg CPU / GPU Utilization (%)", color=color_cpu)
    if 'cpu_percent' in valid_metrics:
        avg_cpu = averages['cpu_percent']
        std_cpu = std_devs['cpu_percent']
        l1 = ax1.plot(common_time, avg_cpu, color=color_cpu, label="Avg CPU Usage (%)")
        ax1.fill_between(common_time, avg_cpu - std_cpu, avg_cpu + std_cpu, color=color_cpu, alpha=0.2, label='_nolegend_')
        lines.extend(l1)
    max_y1_val = 0
    if 'cpu_percent' in valid_metrics: max_y1_val = max(max_y1_val, np.max(averages['cpu_percent'] + std_devs['cpu_percent']))


    color_gpu_util = 'tab:purple'
    if has_any_gpu_data and 'gpu_utilization_percent' in valid_metrics:
        avg_gpu_util = averages['gpu_utilization_percent']
        std_gpu_util = std_devs['gpu_utilization_percent']
        l2 = ax1.plot(common_time, avg_gpu_util, color=color_gpu_util, linestyle='--', label="Avg GPU Utilization (%)")
        ax1.fill_between(common_time, avg_gpu_util - std_gpu_util, avg_gpu_util + std_gpu_util, color=color_gpu_util, alpha=0.2, label='_nolegend_')
        lines.extend(l2)
        if 'gpu_utilization_percent' in valid_metrics: max_y1_val = max(max_y1_val, np.max(averages['gpu_utilization_percent'] + std_devs['gpu_utilization_percent']))


    ax1.tick_params(axis='y', labelcolor=color_cpu)
    ax1.set_ylim(bottom=0, top=max(105, math.ceil(max_y1_val / 10.0) * 10 if max_y1_val > 0 else 105))


    # Plot Memory Usage (Avg +/- Std Dev) on ax2
    color_mem = 'tab:blue'
    ax2.set_ylabel("Avg Memory Usage (MB)", color=color_mem)
    max_y2_val = 0
    if 'memory_rss_mb' in valid_metrics:
        avg_mem = averages['memory_rss_mb']
        std_mem = std_devs['memory_rss_mb']
        l3 = ax2.plot(common_time, avg_mem, color=color_mem, label="Avg RAM RSS (MB)")
        ax2.fill_between(common_time, avg_mem - std_mem, avg_mem + std_mem, color=color_mem, alpha=0.2, label='_nolegend_')
        lines.extend(l3)
        if 'memory_rss_mb' in valid_metrics: max_y2_val = max(max_y2_val, np.max(averages['memory_rss_mb'] + std_devs['memory_rss_mb']))


    color_gpu_mem = 'tab:cyan'
    if has_any_gpu_data and 'gpu_memory_used_mb' in valid_metrics:
         avg_gpu_mem = averages['gpu_memory_used_mb']
         std_gpu_mem = std_devs['gpu_memory_used_mb']
         l4 = ax2.plot(common_time, avg_gpu_mem, color=color_gpu_mem, linestyle='--', label="Avg GPU Memory Used (MB)")
         ax2.fill_between(common_time, avg_gpu_mem - std_gpu_mem, avg_gpu_mem + std_gpu_mem, color=color_gpu_mem, alpha=0.2, label='_nolegend_')
         lines.extend(l4)
         if 'gpu_memory_used_mb' in valid_metrics: max_y2_val = max(max_y2_val, np.max(averages['gpu_memory_used_mb'] + std_devs['gpu_memory_used_mb']))

    ax2.tick_params(axis='y', labelcolor=color_mem)
    ax2.set_ylim(bottom=0, top=max(50, math.ceil(max_y2_val / 50.0) * 50 if max_y2_val > 0 else 50))


    # Combine legends
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(len(labels), 4)) # Legend below plot

    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    try:
        plt.savefig(output_filename, dpi=100)
        print(f"Average plot saved to: {output_filename}", file=sys.stderr)
    except Exception as e:
        print(f"Error saving average plot {output_filename}: {e}", file=sys.stderr)
    finally:
        plt.close(fig)


# --- Utility Functions ---
# (format_duration function remains the same)
def format_duration(seconds: float) -> str:
    """Formats duration in a human-readable way."""
    if seconds < 0: return "N/A" # Handle timeout case where duration might be negative temporarily
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.3f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.3f}s"

# (run_command function remains the same as previous version with plotting)
def run_command(command: List[str], run_index: int, total_runs: int, is_warmup: bool = False, monitor_interval: float = 0.1, monitor_gpu: bool = False, timeout: Optional[float] = None, verbose_level: int = 0) -> Tuple[Optional[float], Optional[Dict[str, Any]], Optional[Dict[str, List[float]]], Optional[int]]:
    """
    Runs the command once and monitors it.
    Returns (duration, resource_summary, raw_timeseries, return_code).
    Returns None for components if errors occur.
    """
    prefix = "[Warmup]" if is_warmup else f"[Run {run_index}/{total_runs}]"
    print(f"{prefix} Executing: {' '.join(command)}", file=sys.stderr)

    monitor = None
    resource_summary = None
    raw_timeseries = None
    return_code = None
    duration = None
    timed_out = False

    try:
        start_time = time.perf_counter()
        preexec_fn = os.setsid if sys.platform != "win32" else None
        process = subprocess.Popen(
            command,
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
             print(f"\n{prefix} Command timed out after {timeout} seconds. Terminating process group...", file=sys.stderr)
             pgid = -1
             try:
                 if sys.platform != "win32":
                     pgid = os.getpgid(process.pid)
                     os.killpg(pgid, signal.SIGTERM)
                 else:
                     # Use taskkill on Windows for better child process termination
                     subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], check=False, capture_output=True)
                     # process.terminate() # Fallback
             except ProcessLookupError: pass # Process already gone
             except Exception as term_err:
                 print(f"{prefix} Error during SIGTERM/taskkill: {term_err}", file=sys.stderr)

             # Wait briefly then force kill if needed
             try:
                 process.wait(timeout=1.5)
             except subprocess.TimeoutExpired:
                 print(f"{prefix} Force killing...", file=sys.stderr)
                 try:
                     if sys.platform != "win32" and pgid > 0:
                         os.killpg(pgid, signal.SIGKILL)
                     else:
                         process.kill()
                 except Exception as kill_err:
                      print(f"{prefix} Error during SIGKILL: {kill_err}", file=sys.stderr)

             stdout, stderr = process.communicate() # Get any remaining output
             return_code = -9 # Convention for SIGKILL/Timeout
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
             print(f"{prefix} --- STDOUT ({len(stdout)} bytes) ---", file=sys.stderr)
             print(stdout, file=sys.stderr)
             print(f"{prefix} --- STDERR ({len(stderr)} bytes) ---", file=sys.stderr)
             print(stderr, file=sys.stderr)
             print(f"{prefix} --- End Output ---", file=sys.stderr)
        elif verbose_level > 0 and (stdout or stderr):
             print(f"{prefix} Output captured ({len(stdout)} bytes stdout, {len(stderr)} bytes stderr). Use -vv to view.", file=sys.stderr)

        if return_code != 0 and not timed_out:
            print(f"{prefix} Warning: Command exited with non-zero status: {return_code}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Command not found: {command[0]}", file=sys.stderr)
        return None, None, None, None
    except PermissionError:
         print(f"Error: Permission denied executing: {command[0]}", file=sys.stderr)
         return None, None, None, None
    except Exception as e:
        print(f"{prefix} Error running or monitoring command: {e}", file=sys.stderr)
        if 'process' in locals() and process.poll() is None:
             try: process.kill()
             except Exception: pass
        if monitor_started:
             try: monitor.stop()
             except Exception: pass
        return None, None, None, None

    return duration, resource_summary, raw_timeseries, return_code


# --- Main Execution ---

def run_benchmark_logic(args):
    """Main function to parse args and run benchmarks."""
    command_to_run = args.command
    if not command_to_run:
        print("Error: No command provided to benchmark.", file=sys.stderr)
        sys.exit(1)

    command_str_short = ' '.join(command_to_run)

    print(f"Benchmarking command: {command_str_short}")
    # ... [Print other config like runs, warmup, interval, timeout] ...
    print(f"Number of runs: {args.runs}")
    if args.warmup > 0: print(f"Warm-up runs: {args.warmup}")
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
    # ... [GPU status print remains the same] ...
    gpu_status = "Disabled"
    if args.gpu:
        if NVIDIA_GPU_SUPPORT:
            try:
                count = pynvml.nvmlDeviceGetCount()
                gpu_status = f"Enabled (NVIDIA, Found {count} GPU(s))"
            except pynvml.NVMLError as e:
                 gpu_status = f"Error enabling NVIDIA GPU monitoring: {e}"
        else:
            gpu_status = "Enabled attempt, but pynvml not found or failed to load."
    print(f"GPU Monitoring: {gpu_status}")
    print("-" * 20, file=sys.stderr)


    # --- Warm-up Runs ---
    if args.warmup > 0:
        print("Starting warm-up runs...", file=sys.stderr)
        for i in range(args.warmup):
            run_command(command_to_run, i + 1, args.warmup, is_warmup=True,
                        monitor_interval=args.interval, monitor_gpu=args.gpu,
                        timeout=args.timeout, verbose_level=args.verbose)
        print("Warm-up complete.", file=sys.stderr)
        print("-" * 20, file=sys.stderr)

    # --- Benchmark Runs ---
    run_results = []
    print("Starting benchmark runs...", file=sys.stderr)
    for i in range(args.runs):
        duration, summary, timeseries, exit_code = run_command(
            command_to_run, i + 1, args.runs, is_warmup=False,
            monitor_interval=args.interval, monitor_gpu=args.gpu,
            timeout=args.timeout, verbose_level=args.verbose
            )

        if duration is None or summary is None or timeseries is None or exit_code is None:
             print(f"[Run {i+1}/{args.runs}] Failed. Skipping results and plot for this run.", file=sys.stderr)
             continue

        run_data = {
            "run": i + 1,
            "duration_s": duration,
            "exit_code": exit_code,
            "summary": summary,
            "timeseries": timeseries
        }
        run_results.append(run_data)

        # Print intermediate result summary
        res_str = f"Time: {format_duration(duration)}, CPU Max: {summary.get('cpu_percent_max', 'N/A'):.1f}%, Mem Max: {summary.get('memory_rss_mb_max', 'N/A'):.1f} MB"
        gpu_max_util = summary.get('gpu_utilization_percent_max', float('nan'))
        gpu_max_mem = summary.get('gpu_memory_used_mb_max', float('nan'))
        if args.gpu and NVIDIA_GPU_SUPPORT and not math.isnan(gpu_max_util):
             res_str += f", GPU Max: {gpu_max_util:.1f}%, GPU Mem Max: {gpu_max_mem:.1f} MB"
        if summary.get("monitoring_error"):
             res_str += f" (Monitor Warn: {summary['monitoring_error']})"
        print(f"[Run {i+1}/{args.runs}] {res_str}")


        # Generate plot for this run if requested
        if plot_dir:
            # Check timeseries has enough data before plotting
             if timeseries and len(timeseries.get('timestamps_rel_s', [])) >= 2:
                 plot_filename = os.path.join(plot_dir, f"run_{i+1:03d}_resources.png")
                 generate_plot(i + 1, args.runs, timeseries, plot_filename, command_str_short)
             elif timeseries:
                  print(f"[Run {i+1}/{args.runs}] Skipping plot due to insufficient timeseries data ({len(timeseries.get('timestamps_rel_s',[]))} points).", file=sys.stderr)


    print("-" * 20, file=sys.stderr)
    print("Benchmark Complete.")

    # --- Aggregate and Print Final Summary ---
    if not run_results:
        print("\nNo successful runs completed. Cannot provide summary.")
        # Cleanup NVML even on failure
        if NVIDIA_GPU_SUPPORT: pynvml.nvmlShutdown()
        sys.exit(1)

    # ... [Summary printing code remains the same] ...
    valid_durations = [r['duration_s'] for r in run_results]
    avg_duration = statistics.mean(valid_durations)
    min_duration = min(valid_durations)
    max_duration = max(valid_durations)
    stdev_duration = statistics.stdev(valid_durations) if len(valid_durations) > 1 else 0.0

    print("\n--- Summary ---")
    print(f"Command: {command_str_short}")
    print(f"Total successful runs: {len(run_results)}")

    print("\nExecution Time:")
    print(f"  Average: {format_duration(avg_duration)}")
    print(f"  Min:     {format_duration(min_duration)}")
    print(f"  Max:     {format_duration(max_duration)}")
    print(f"  StdDev:  {format_duration(stdev_duration)}")

    def aggregate_summary_stat(key):
        values = [r['summary'].get(key) for r in run_results if r['summary'].get(key) is not None and math.isfinite(r['summary'].get(key))]
        if not values: return float('nan')
        return statistics.mean(values)

    avg_cpu_max = aggregate_summary_stat('cpu_percent_max')
    avg_cpu_avg = aggregate_summary_stat('cpu_percent_avg')
    avg_mem_max = aggregate_summary_stat('memory_rss_mb_max')
    avg_mem_avg = aggregate_summary_stat('memory_rss_mb_avg')

    print("\nResource Usage (Averages across runs):")
    print(f"  Avg CPU Usage (Avg): {avg_cpu_avg:.1f}%")
    print(f"  Avg CPU Usage (Max): {avg_cpu_max:.1f}%")
    print(f"  Avg Memory RSS (Avg): {avg_mem_avg:.1f} MB")
    print(f"  Avg Memory RSS (Max): {avg_mem_max:.1f} MB")

    if args.gpu and NVIDIA_GPU_SUPPORT:
         avg_gpu_util_max = aggregate_summary_stat('gpu_utilization_percent_max')
         avg_gpu_util_avg = aggregate_summary_stat('gpu_utilization_percent_avg')
         avg_gpu_mem_max = aggregate_summary_stat('gpu_memory_used_mb_max')
         avg_gpu_mem_avg = aggregate_summary_stat('gpu_memory_used_mb_avg')

         if not math.isnan(avg_gpu_util_max): # Check one metric is enough
             print(f"  Avg GPU Utilization (Avg): {avg_gpu_util_avg:.1f}%")
             print(f"  Avg GPU Utilization (Max): {avg_gpu_util_max:.1f}%")
             print(f"  Avg GPU Memory Used (Avg): {avg_gpu_mem_avg:.1f} MB") # System-wide
             print(f"  Avg GPU Memory Used (Max): {avg_gpu_mem_max:.1f} MB")
         else:
              print("  GPU monitoring enabled, but no valid GPU data collected across run summaries.")

    monitoring_errors = [r['summary']['monitoring_error'] for r in run_results if 'monitoring_error' in r['summary']]
    if monitoring_errors:
         print(f"\nNote: Encountered {len(monitoring_errors)} monitoring issue(s) during the runs.")


    # --- Generate Average Plot ---
    if plot_dir:
        avg_plot_filename = os.path.join(plot_dir, "average_resources.png")
        generate_average_plot(run_results, avg_plot_filename, command_str_short)
        print(f"\nIndividual run plots and average plot saved in: {os.path.abspath(plot_dir)}")
    else:
         # Ensure plots aren't mentioned if disabled
         pass


    # Cleanup NVML
    if NVIDIA_GPU_SUPPORT:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass # Ignore shutdown errors


def entry_point():
    """Handles argument parsing and calls the main benchmark logic."""
    parser = argparse.ArgumentParser(
        description="Benchmark a command-line command for execution time and resource usage, with optional plotting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: cmdbench --plot-dir ./plots -n 3 --gpu -- my_program arg1"
    )
    # Add all arguments previously defined in the if __name__ == "__main__": block
    parser.add_argument(
        "-n", "--runs", type=int, default=3,
        help="Number of times to run the command for benchmarking."
    )
    parser.add_argument(
        "-w", "--warmup", type=int, default=1,
        help="Number of warm-up runs to perform before benchmarking."
    )
    parser.add_argument(
        "-i", "--interval", type=float, default=0.1,
        help="Resource monitoring interval in seconds."
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Enable NVIDIA GPU monitoring (requires 'pip install cmdbench[gpu]' and NVIDIA driver)."
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Set a timeout in seconds for each command run."
    )
    parser.add_argument(
        "--plot-dir", type=str, default=None,
        help="Directory to save resource usage plots (individual runs + average)."
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity. -v shows run output hints, -vv shows full stdout/stderr."
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER,
        help="The command and its arguments to benchmark."
    )

    args = parser.parse_args()

    # Check plotting dependencies if plotting is requested
    if args.plot_dir and not PLOTTING_SUPPORT:
         print("Error: Plotting requested (--plot-dir), but matplotlib/numpy/scipy are not installed correctly.", file=sys.stderr)
         print("Please ensure these libraries are installed in your environment.", file=sys.stderr)
         sys.exit(1) # Exit if plotting requested but libs missing


    run_benchmark_logic(args) # Call the main logic function