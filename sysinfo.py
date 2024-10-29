#!/usr/bin/env python3
import psutil
import platform
import argparse

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def get_system_info():
    print("="*40, "System Information", "="*40)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")

def get_cpu_info():
    print("="*40, "CPU Info", "="*40)
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))

def get_memory_info():
    print("="*40, "Memory Information", "="*40)
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")

def main():
    parser = argparse.ArgumentParser(description="CLI tool to get system information")
    parser.add_argument("-a", "--all", action="store_true", help="Show all information")
    parser.add_argument("-s", "--system", action="store_true", help="Show system information")
    parser.add_argument("-c", "--cpu", action="store_true", help="Show CPU information")
    parser.add_argument("-m", "--memory", action="store_true", help="Show memory information")

    args = parser.parse_args()

    if args.all or not any(vars(args).values()):
        get_system_info()
        get_cpu_info()
        get_memory_info()
    else:
        if args.system:
            get_system_info()
        if args.cpu:
            get_cpu_info()
        if args.memory:
            get_memory_info()

if __name__ == "__main__":
    main()
