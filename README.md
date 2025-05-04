# Command Line Benchmarker (cmdbench)

A command-line tool to benchmark other command-line commands, measuring execution time, CPU usage, RAM usage, and optionally NVIDIA GPU usage. It can perform multiple runs, handle warmups, and generate plots of resource usage over time.

## Installation

It is recommended to install in a virtual environment.

```bash
# Create and activate a virtual environment (example)
python -m venv venv
source venv/bin/activate # Linux/macOS
# venv\Scripts\activate # Windows

# Clone or download the project directory (cmdbench-pkg)
# cd cmdbench-pkg

# Install core package (includes plotting support)
pip install .

# To include NVIDIA GPU monitoring support:
pip install .[gpu]

## Usage

# Show help message
cmdbench --help

# Basic benchmark (3 runs, 1 warmup)
cmdbench -- sleep 1

# Benchmark with 5 runs, plot output, GPU monitoring
cmdbench -n 5 --plot-dir ./benchmark_plots --gpu -- your_gpu_program --args

# Benchmark a windows command (example)
cmdbench -- ping 8.8.8.8 -n 4