#!/usr/bin/env python3
"""
Script to run a specific test multiple times and determine tolerance overrides needed.
This script runs the failing test multiple times, parses the error output, and tracks 
the maximum absolute and relative tolerances needed.
"""

import re
import subprocess
import sys
import os
from typing import List, Tuple, Dict, Any

def run_test_once(test_command: str) -> Tuple[bool, str, Dict[str, float]]:
    """
    Run the test once and parse tolerance information from output.
    
    Returns:
        (success, output, tolerance_info)
        tolerance_info contains: {'abs_diff': float, 'rel_diff': float, 'abs_allowed': float, 'rel_allowed': float}
    """
    try:
        result = subprocess.run(
            test_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout
        )
        
        output = result.stdout + result.stderr
        success = result.returncode == 0
        
        tolerance_info = {}
        
        if not success:
            # Parse tolerance information from error output
            abs_diff_match = re.search(r'Greatest absolute difference: ([\d\.e\-\+]+)', output)
            rel_diff_match = re.search(r'Greatest relative difference: ([\d\.e\-\+]+)', output)
            abs_allowed_match = re.search(r'\(up to ([\d\.e\-\+]+) allowed\)', output)
            rel_allowed_match = re.search(r'\(up to ([\d\.e\-\+]+) allowed\)', output)
            
            if abs_diff_match:
                tolerance_info['abs_diff'] = float(abs_diff_match.group(1))
            if rel_diff_match:
                tolerance_info['rel_diff'] = float(rel_diff_match.group(1))
            if abs_allowed_match:
                tolerance_info['abs_allowed'] = float(abs_allowed_match.group(1))
            if rel_allowed_match:
                tolerance_info['rel_allowed'] = float(rel_allowed_match.group(1))
                
        return success, output, tolerance_info
        
    except subprocess.TimeoutExpired:
        return False, "Test timed out", {}
    except Exception as e:
        return False, f"Error running test: {str(e)}", {}

def run_multiple_tests(test_command: str, num_runs: int) -> Dict[str, Any]:
    """
    Run the test multiple times and collect statistics.
    """
    results = {
        'total_runs': num_runs,
        'successful_runs': 0,
        'failed_runs': 0,
        'timeout_runs': 0,
        'max_abs_diff': 0.0,
        'max_rel_diff': 0.0,
        'current_abs_allowed': None,
        'current_rel_allowed': None,
        'tolerance_failures': []
    }
    
    print(f"Running test {num_runs} times...")
    print(f"Command: {test_command}")
    print("-" * 80)
    
    for run_num in range(1, num_runs + 1):
        print(f"{run_num}", end=" ", flush=True)
        
        success, output, tolerance_info = run_test_once(test_command)
        
        if success:
            results['successful_runs'] += 1
            print("P", end=" ", flush=True)
        else:
            if "timed out" in output:
                results['timeout_runs'] += 1
                print("TIMEOUT")
            else:
                results['failed_runs'] += 1
                print("\nFAIL\n")
                
                # Track tolerance information
                if tolerance_info:
                    if 'abs_diff' in tolerance_info:
                        results['max_abs_diff'] = max(results['max_abs_diff'], tolerance_info['abs_diff'])
                        
                    if 'rel_diff' in tolerance_info:
                        results['max_rel_diff'] = max(results['max_rel_diff'], tolerance_info['rel_diff'])
                        
                    if 'abs_allowed' in tolerance_info and results['current_abs_allowed'] is None:
                        results['current_abs_allowed'] = tolerance_info['abs_allowed']
                        
                    if 'rel_allowed' in tolerance_info and results['current_rel_allowed'] is None:
                        results['current_rel_allowed'] = tolerance_info['rel_allowed']
                    
                    results['tolerance_failures'].append({
                        'run': run_num,
                        'tolerance_info': tolerance_info
                    })
    
    return results

def generate_tolerance_override(results: Dict[str, Any]) -> str:
    """
    Generate tolerance override code based on results.
    """
    if results['failed_runs'] == 0:
        return "No tolerance override needed - all tests passed!"
    
    # Add safety margin (multiply by 1.5 to handle variation)
    safety_margin = 1.5
    
    recommended_atol = results['max_abs_diff'] * safety_margin
    recommended_rtol = results['max_rel_diff'] * safety_margin
    
    override_code = f"""
# Based on {results['total_runs']} test runs:
# Current tolerances: atol={results['current_abs_allowed']}, rtol={results['current_rel_allowed']}
# Max observed: abs_diff={results['max_abs_diff']:.2e}, rel_diff={results['max_rel_diff']:.2e}
# Recommended with safety margin (1.5x):

from torch.testing._internal.common_device_type import toleranceOverride, tol

@toleranceOverride({{torch.float32: tol(atol={recommended_atol:.2e}, rtol={recommended_rtol:.2e})}})
"""
    
    return override_code

def print_results(results: Dict[str, Any]):
    """
    Print detailed results summary.
    """
    print("\n" + "=" * 80)
    print("TOLERANCE ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Total runs: {results['total_runs']}")
    print(f"Successful: {results['successful_runs']}")
    print(f"Failed: {results['failed_runs']}")
    print(f"Timeouts: {results['timeout_runs']}")
    
    if results['failed_runs'] > 0:
        print(f"\nCurrent tolerance limits:")
        print(f"  Absolute (atol): {results['current_abs_allowed']}")
        print(f"  Relative (rtol): {results['current_rel_allowed']}")
        
        print(f"\nMaximum observed differences:")
        print(f"  Absolute: {results['max_abs_diff']:.2e}")
        print(f"  Relative: {results['max_rel_diff']:.2e}")
        
        print(f"\nRecommended tolerance override:")
        print(generate_tolerance_override(results))
        
        if len(results['tolerance_failures']) > 0:
            print(f"\nDetailed failure analysis (last 5 failures):")
            for failure in results['tolerance_failures'][-5:]:
                run_num = failure['run']
                info = failure['tolerance_info']
                print(f"  Run {run_num}:")
                for key, value in info.items():
                    print(f"    {key}: {value:.2e}")
    else:
        print("\nAll tests passed! No tolerance override needed.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python tolerance_finder.py <num_runs> [test_command]")
        print("\nExample:")
        print("  python tolerance_finder.py 20")
        print("  python tolerance_finder.py 50 'pytest -v functorch/test_ops_xpu.py -k test_vjp_nn_functional_conv3d_xpu_float32'")
        sys.exit(1)
    
    num_runs = int(sys.argv[1])
    
    # Default to the failing test
    # test_command = "pytest -v functorch/test_ops_xpu.py -k test_vjp_nn_functional_conv3d_xpu_float32"
    # test_command = "pytest -v functorch/test_ops_xpu.py -k test_grad_nn_functional_conv3d_xpu_float32"
    test_command = [
        "pytest -v functorch/test_ops_xpu.py -k test_vjp_nn_functional_conv3d_xpu_float32",
        "pytest -v functorch/test_ops_xpu.py -k test_grad_nn_functional_conv3d_xpu_float32",
        "pytest -v functorch/test_ops_xpu.py -k test_jvpvjp_nn_functional_conv3d_xpu_float32",
        # "pytest -v functorch/test_ops_xpu.py -k test_vjpvjp_index_reduce_prod_xpu_float32",
    ]

    # Change to the test directory
    os.chdir("/home/bbela/pytorch/third_party/torch-xpu-ops/test/xpu")
    
    for i in range(len(test_command)):
        # Run the tests
        results = run_multiple_tests(test_command[i], num_runs)
        
        # Print results
        print_results(results)



if __name__ == "__main__":
    main()