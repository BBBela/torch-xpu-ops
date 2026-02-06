#!/usr/bin/env python3
"""
Performance benchmark script for triangular operations (triu/tril) on Intel XPU.
Tests various matrix sizes, data types, and operation parameters.
"""

import torch
import time
import gc
import sys
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np

def setup_xpu():
    """Initialize XPU device and verify availability."""
    if not torch.xpu.is_available():
        print("ERROR: XPU is not available!")
        sys.exit(1)
    
    device_count = torch.xpu.device_count()
    print(f"Found {device_count} XPU device(s)")
    
    device = torch.device('xpu:0')
    torch.xpu.set_device(0)
    
    # Warm up
    dummy = torch.randn(100, 100, device=device)
    torch.xpu.synchronize()
    
    return device

def benchmark_operation(tensor: torch.Tensor, operation: str, k: int = 0, 
                       iterations: int = 100) -> Dict[str, float]:
    """Benchmark a single triangular operation."""
    device = tensor.device
    
    # Warm up
    for _ in range(10):
        if operation == 'triu':
            result = torch.triu(tensor, diagonal=k)
        elif operation == 'tril':
            result = torch.tril(tensor, diagonal=k)
        elif operation == 'triu_inplace':
            temp = tensor.clone()
            temp.triu_(diagonal=k)
            result = temp
        elif operation == 'tril_inplace':
            temp = tensor.clone()
            temp.tril_(diagonal=k)
            result = temp
    
    torch.xpu.synchronize()
    
    # Actual benchmark
    torch.xpu.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        if operation == 'triu':
            result = torch.triu(tensor, diagonal=k)
        elif operation == 'tril':
            result = torch.tril(tensor, diagonal=k)
        elif operation == 'triu_inplace':
            temp = tensor.clone()
            temp.triu_(diagonal=k)
        elif operation == 'tril_inplace':
            temp = tensor.clone()
            temp.tril_(diagonal=k)
    
    torch.xpu.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    
    # Calculate throughput
    elements = tensor.numel()
    element_size = tensor.element_size()
    data_size_gb = (elements * element_size) / (1024**3)
    throughput_gb_s = data_size_gb / avg_time
    
    return {
        'avg_time_ms': avg_time * 1000,
        'throughput_gb_s': throughput_gb_s,
        'elements_per_sec': elements / avg_time
    }

def create_test_matrix(rows: int, cols: int, dtype: torch.dtype, 
                      device: torch.device) -> torch.Tensor:
    """Create a test matrix with specified dimensions and dtype."""
    if dtype in [torch.bool]:
        return torch.randint(0, 2, (rows, cols), dtype=dtype, device=device)
    elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        return torch.randint(-100, 100, (rows, cols), dtype=dtype, device=device)
    elif dtype in [torch.uint8]:
        return torch.randint(0, 255, (rows, cols), dtype=dtype, device=device)
    else:  # float types
        return torch.randn(rows, cols, dtype=dtype, device=device)

def validate_correctness(tensor: torch.Tensor, operation: str, k: int = 0) -> bool:
    """Validate operation correctness against CPU reference."""
    cpu_tensor = tensor.cpu()
    
    if operation in ['triu', 'triu_inplace']:
        xpu_result = torch.triu(tensor, diagonal=k)
        cpu_result = torch.triu(cpu_tensor, diagonal=k)
    else:  # tril operations
        xpu_result = torch.tril(tensor, diagonal=k)
        cpu_result = torch.tril(cpu_tensor, diagonal=k)
    
    return torch.allclose(xpu_result.cpu(), cpu_result, rtol=1e-5, atol=1e-8)

def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    device = setup_xpu()
    
    # Test configurations
    test_sizes = [
        (128, 128, "Small"),
        (1024, 1024, "Medium"),
        (4096, 4096, "Large"),
        (10000, 10000, "Very Large"),
        (50000, 50000, "Huge"),
        (100000, 100000, "Extreme"),  # The problematic size
    ]
    
    test_dtypes = [
        (torch.int8, "int8"),
        (torch.int32, "int32"),
        (torch.int64, "int64"),
        (torch.float16, "float16"), 
        (torch.float32, "float32"),
        (torch.float64, "float64"),
        (torch.bool, "bool"),
    ]
    
    operations = ['triu', 'tril', 'triu_inplace', 'tril_inplace']
    k_values = [0, 1, -1, 5, -5]  # Different diagonal offsets
    
    results = []
    
    print("=" * 80)
    print("TRIANGULAR OPERATIONS PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    for rows, cols, size_name in test_sizes:
        print(f"\n🔄 Testing {size_name} matrices ({rows}x{cols})...")
        
        for dtype, dtype_name in test_dtypes:
            print(f"  📊 Data type: {dtype_name}")
            
            try:
                # Create test matrix
                matrix = create_test_matrix(rows, cols, dtype, device)
                memory_gb = (matrix.numel() * matrix.element_size()) / (1024**3)
                print(f"     Memory usage: {memory_gb:.2f} GB")
                
                # Test correctness first (only for smaller matrices to save time)
                if rows <= 4096:
                    if not validate_correctness(matrix, 'triu', k=0):
                        print(f"     ⚠️  Correctness validation failed for triu!")
                        continue
                    if not validate_correctness(matrix, 'tril', k=0):
                        print(f"     ⚠️  Correctness validation failed for tril!")
                        continue
                    print(f"     ✅ Correctness validation passed")
                
                for operation in operations:
                    for k in k_values:
                        try:
                            # Adjust iterations based on matrix size
                            if rows >= 50000:
                                iterations = 10
                            elif rows >= 10000:
                                iterations = 20
                            elif rows >= 4096:
                                iterations = 50
                            else:
                                iterations = 100
                            
                            metrics = benchmark_operation(matrix, operation, k, iterations)
                            
                            result = {
                                'size_name': size_name,
                                'rows': rows,
                                'cols': cols,
                                'dtype': dtype_name,
                                'operation': operation,
                                'k': k,
                                'memory_gb': memory_gb,
                                'iterations': iterations,
                                **metrics
                            }
                            results.append(result)
                            
                            print(f"     {operation}(k={k:2d}): {metrics['avg_time_ms']:8.2f}ms, "
                                  f"{metrics['throughput_gb_s']:6.1f} GB/s")
                            
                        except Exception as e:
                            print(f"     ❌ {operation}(k={k}) failed: {str(e)}")
                            
                        # Force garbage collection
                        gc.collect()
                        torch.xpu.empty_cache()
                
                del matrix
                gc.collect()
                torch.xpu.empty_cache()
                
            except Exception as e:
                print(f"  ❌ Failed to test {dtype_name}: {str(e)}")
                continue
    
    return results

def generate_report(results: List[Dict[str, Any]], output_file: str = None):
    """Generate performance report."""
    if not results:
        print("No results to report!")
        return
    
    report_lines = []
    
    # Summary header
    report_lines.append("=" * 100)
    report_lines.append("TRIANGULAR OPERATIONS PERFORMANCE REPORT")
    report_lines.append("=" * 100)
    
    # Group results by size and operation
    size_groups = {}
    for result in results:
        size_key = (result['size_name'], result['rows'], result['cols'])
        if size_key not in size_groups:
            size_groups[size_key] = {}
        
        op_key = (result['operation'], result['k'])
        if op_key not in size_groups[size_key]:
            size_groups[size_key][op_key] = []
        
        size_groups[size_key][op_key].append(result)
    
    # Generate detailed report
    for size_key, operations in size_groups.items():
        size_name, rows, cols = size_key
        report_lines.append(f"\n📏 {size_name} Matrices ({rows}x{cols})")
        report_lines.append("-" * 60)
        
        for op_key, op_results in operations.items():
            operation, k = op_key
            report_lines.append(f"\n  🎯 Operation: {operation}(k={k})")
            
            # Sort by dtype for consistent ordering
            op_results.sort(key=lambda x: x['dtype'])
            
            report_lines.append(f"    {'Dtype':<10} {'Time(ms)':<10} {'Throughput(GB/s)':<15} {'Memory(GB)':<12}")
            report_lines.append(f"    {'-'*10} {'-'*10} {'-'*15} {'-'*12}")
            
            for result in op_results:
                report_lines.append(
                    f"    {result['dtype']:<10} "
                    f"{result['avg_time_ms']:<10.2f} "
                    f"{result['throughput_gb_s']:<15.1f} "
                    f"{result['memory_gb']:<12.2f}"
                )
    
    # Performance summary
    report_lines.append(f"\n📊 PERFORMANCE SUMMARY")
    report_lines.append("-" * 60)
    
    # Find fastest operations per size category
    fastest_by_size = {}
    for result in results:
        size_key = result['size_name']
        if size_key not in fastest_by_size or result['throughput_gb_s'] > fastest_by_size[size_key]['throughput_gb_s']:
            fastest_by_size[size_key] = result
    
    report_lines.append(f"Fastest operations by matrix size:")
    for size_name, result in fastest_by_size.items():
        report_lines.append(
            f"  {size_name:<12}: {result['operation']} {result['dtype']} "
            f"@ {result['throughput_gb_s']:.1f} GB/s"
        )
    
    # Memory scaling analysis
    report_lines.append(f"\n💾 MEMORY SCALING ANALYSIS")
    report_lines.append("-" * 60)
    
    large_matrices = [r for r in results if r['rows'] >= 10000]
    if large_matrices:
        avg_throughput = np.mean([r['throughput_gb_s'] for r in large_matrices])
        report_lines.append(f"Average throughput for large matrices (>=10k): {avg_throughput:.1f} GB/s")
        
        extreme_matrices = [r for r in results if r['rows'] >= 100000]
        if extreme_matrices:
            extreme_avg = np.mean([r['throughput_gb_s'] for r in extreme_matrices])
            scaling_ratio = extreme_avg / avg_throughput
            report_lines.append(f"Extreme matrix throughput (100k+): {extreme_avg:.1f} GB/s")
            report_lines.append(f"Scaling efficiency: {scaling_ratio:.2f}x")
    
    # Generate final report
    full_report = '\n'.join(report_lines)
    print(full_report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_report)
        print(f"\n📄 Report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark triangular operations on Intel XPU')
    parser.add_argument('--output', '-o', help='Output file for detailed report')
    parser.add_argument('--quick', action='store_true', help='Run quick test with limited configurations')
    args = parser.parse_args()
    
    try:
        if args.quick:
            print("🚀 Running quick benchmark...")
            # Override test configurations for quick test
            global test_sizes, test_dtypes
            test_sizes = [(1024, 1024, "Medium"), (10000, 10000, "Large")]
            test_dtypes = [(torch.float32, "float32"), (torch.int32, "int32")]
        
        results = run_benchmark_suite()
        generate_report(results, args.output)
        
        print(f"\n✅ Benchmark completed successfully!")
        print(f"   Total test cases: {len(results)}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()