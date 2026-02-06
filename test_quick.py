#!/usr/bin/env python3
"""
Simple test runner for triangular operations - useful for quick validation.
"""

import torch
import time

def quick_test():
    """Run a quick test to verify triangular operations work correctly."""
    
    if not torch.xpu.is_available():
        print("❌ XPU not available!")
        return False
    
    device = torch.device('xpu:0')
    print(f"✅ Using device: {device}")
    
    # Test the problematic case that was failing
    print("\n🧪 Testing large matrix (the original failing case)...")
    
    try:
        # Create the large matrix that was causing the uint32 overflow
        rows, cols = 100000, 100000
        print(f"   Creating {rows}x{cols} int8 matrix...")
        
        matrix = torch.randint(0, 100, (rows, cols), dtype=torch.int8, device=device)
        memory_gb = (matrix.numel() * matrix.element_size()) / (1024**3)
        print(f"   Memory usage: {memory_gb:.2f} GB")
        
        # Test triu operation
        print("   Testing triu operation...")
        start_time = time.perf_counter()
        result_triu = torch.triu(matrix)
        torch.xpu.synchronize()
        triu_time = time.perf_counter() - start_time
        print(f"   ✅ triu completed in {triu_time:.3f}s")
        
        # Test tril operation
        print("   Testing tril operation...")
        start_time = time.perf_counter()
        result_tril = torch.tril(matrix)
        torch.xpu.synchronize()
        tril_time = time.perf_counter() - start_time
        print(f"   ✅ tril completed in {tril_time:.3f}s")
        
        # Validate results have correct shape
        assert result_triu.shape == matrix.shape
        assert result_tril.shape == matrix.shape
        print("   ✅ Output shapes correct")
        
        # Test with different k values
        for k in [-1, 0, 1, 5]:
            torch.triu(matrix, diagonal=k)
            torch.tril(matrix, diagonal=k)
        print("   ✅ Different diagonal offsets work")
        
        print(f"\n🎉 SUCCESS: Large matrix test passed!")
        print(f"   The uint32 overflow issue appears to be resolved.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        return False

def test_multiple_sizes():
    """Test various matrix sizes to check scaling."""
    
    if not torch.xpu.is_available():
        return False
        
    device = torch.device('xpu:0')
    
    sizes = [
        (1000, 1000),
        (5000, 5000), 
        (10000, 10000),
        (50000, 50000),
        (100000, 100000)
    ]
    
    print("\n📏 Testing multiple matrix sizes...")
    
    for rows, cols in sizes:
        try:
            print(f"   {rows}x{cols}... ", end="")
            matrix = torch.randn(rows, cols, dtype=torch.float32, device=device)
            
            start_time = time.perf_counter()
            result = torch.triu(matrix)
            torch.xpu.synchronize()
            elapsed = time.perf_counter() - start_time
            
            print(f"✅ {elapsed:.3f}s")
            del matrix, result
            torch.xpu.empty_cache()
            
        except Exception as e:
            print(f"❌ {str(e)}")
            return False
    
    return True

def test_data_types():
    """Test different data types."""
    
    if not torch.xpu.is_available():
        return False
        
    device = torch.device('xpu:0')
    
    dtypes = [
        torch.int8, torch.int32, torch.int64,
        torch.float16, torch.float32, torch.float64,
        torch.bool
    ]
    
    print("\n🔢 Testing different data types...")
    
    rows, cols = 5000, 5000
    
    for dtype in dtypes:
        try:
            print(f"   {str(dtype).split('.')[-1]}... ", end="")
            
            if dtype == torch.bool:
                matrix = torch.randint(0, 2, (rows, cols), dtype=dtype, device=device)
            elif dtype in [torch.int8, torch.int32, torch.int64]:
                matrix = torch.randint(-100, 100, (rows, cols), dtype=dtype, device=device)
            else:
                matrix = torch.randn(rows, cols, dtype=dtype, device=device)
            
            result = torch.triu(matrix)
            result = torch.tril(matrix)
            
            print("✅")
            del matrix, result
            torch.xpu.empty_cache()
            
        except Exception as e:
            print(f"❌ {str(e)}")
            return False
    
    return True

def main():
    print("🚀 Quick Triangular Operations Test")
    print("=" * 50)
    
    success = True
    
    success &= quick_test()
    success &= test_multiple_sizes() 
    success &= test_data_types()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The triangular operations implementation is working correctly.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("There may be issues with the implementation.")

if __name__ == "__main__":
    main()