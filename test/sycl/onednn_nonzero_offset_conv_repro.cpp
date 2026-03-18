#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using dnnl::algorithm;
using dnnl::convolution_forward;
using dnnl::engine;
using dnnl::memory;
using dnnl::prop_kind;
using dnnl::stream;

struct Options {
  int device_index = 0;
  float atol = 1.0e-4f;
};

static int64_t output_size_1d(int64_t in, int64_t k, int64_t s, int64_t p, int64_t d) {
  const int64_t kernel = d * (k - 1) + 1;
  return (in + 2 * p - kernel) / s + 1;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("size mismatch");
  }
  float mx = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    mx = std::max(mx, std::abs(a[i] - b[i]));
  }
  return mx;
}

static void cpu_conv2d_ref(const std::vector<float>& src, const std::vector<float>& wei, std::vector<float>& dst,
    int64_t n, int64_t c, int64_t h, int64_t w, int64_t o, int64_t kh, int64_t kw, int64_t sh, int64_t sw,
    int64_t ph, int64_t pw, int64_t dh, int64_t dw)
{
  const int64_t oh = output_size_1d(h, kh, sh, ph, dh);
  const int64_t ow = output_size_1d(w, kw, sw, pw, dw);

  auto src_off = [&](int64_t ni, int64_t ci, int64_t yi, int64_t xi) {
    return ((ni * c + ci) * h + yi) * w + xi;
  };
  auto wei_off = [&](int64_t oi, int64_t ci, int64_t ky, int64_t kx) {
    return ((oi * c + ci) * kh + ky) * kw + kx;
  };
  auto dst_off = [&](int64_t ni, int64_t oi, int64_t yi, int64_t xi) {
    return ((ni * o + oi) * oh + yi) * ow + xi;
  };

  for (int64_t ni = 0; ni < n; ++ni) {
    for (int64_t oi = 0; oi < o; ++oi) {
      for (int64_t yo = 0; yo < oh; ++yo) {
        for (int64_t xo = 0; xo < ow; ++xo) {
          float acc = 0.0f;
          for (int64_t ci = 0; ci < c; ++ci) {
            for (int64_t ky = 0; ky < kh; ++ky) {
              const int64_t yi = yo * sh - ph + ky * dh;
              if (yi < 0 || yi >= h) {
                continue;
              }
              for (int64_t kx = 0; kx < kw; ++kx) {
                const int64_t xi = xo * sw - pw + kx * dw;
                if (xi < 0 || xi >= w) {
                  continue;
                }
                acc += src[static_cast<size_t>(src_off(ni, ci, yi, xi))] *
                    wei[static_cast<size_t>(wei_off(oi, ci, ky, kx))];
              }
            }
          }
          dst[static_cast<size_t>(dst_off(ni, oi, yo, xo))] = acc;
        }
      }
    }
  }
}

template <typename T>
static T* alloc_shared(const engine& eng, size_t n) {
  auto dev = dnnl::sycl_interop::get_device(eng);
  auto ctx = dnnl::sycl_interop::get_context(eng);
  T* p = sycl::malloc_shared<T>(n, dev, ctx);
  if (!p) {
    throw std::runtime_error("USM allocation failed");
  }
  return p;
}

static std::vector<float> run_onednn_conv(const engine& eng, stream& strm, float* src_ptr, float* wei_ptr,
    int64_t n, int64_t c, int64_t h, int64_t w, int64_t o, int64_t kh, int64_t kw, int64_t sh, int64_t sw,
    int64_t ph, int64_t pw, int64_t dh, int64_t dw)
{
  const int64_t oh = output_size_1d(h, kh, sh, ph, dh);
  const int64_t ow = output_size_1d(w, kw, sw, pw, dw);
  const int64_t dst_n = n * o * oh * ow;

  float* dst_ptr = alloc_shared<float>(eng, static_cast<size_t>(dst_n));
  std::fill(dst_ptr, dst_ptr + dst_n, 0.0f);

  memory::dims src_dims = {n, c, h, w};
  memory::dims wei_dims = {o, c, kh, kw};
  memory::dims dst_dims = {n, o, oh, ow};
  memory::dims strides = {sh, sw};
  memory::dims dilates = {dh - 1, dw - 1};
  memory::dims pad_l = {ph, pw};
  memory::dims pad_r = {ph, pw};

  auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
  auto wei_md = memory::desc(wei_dims, memory::data_type::f32, memory::format_tag::oihw);
  auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

  auto src_m = dnnl::sycl_interop::make_memory(src_md, eng, dnnl::sycl_interop::memory_kind::usm, src_ptr);
  auto wei_m = dnnl::sycl_interop::make_memory(wei_md, eng, dnnl::sycl_interop::memory_kind::usm, wei_ptr);
  auto dst_m = dnnl::sycl_interop::make_memory(dst_md, eng, dnnl::sycl_interop::memory_kind::usm, dst_ptr);

  auto pd = convolution_forward::primitive_desc(eng, prop_kind::forward_training, algorithm::convolution_direct,
      src_md, wei_md, dst_md, strides, dilates, pad_l, pad_r);
  auto conv = convolution_forward(pd);

  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_WEIGHTS, wei_m});
  args.insert({DNNL_ARG_DST, dst_m});

  conv.execute(strm, args);
  strm.wait();

  std::vector<float> out(static_cast<size_t>(dst_n));
  std::copy(dst_ptr, dst_ptr + dst_n, out.begin());

  auto ctx = dnnl::sycl_interop::get_context(eng);
  sycl::free(dst_ptr, ctx);
  return out;
}

int main(int argc, char** argv) {
  try {
    if (engine::get_count(engine::kind::gpu) == 0) {
      std::cerr << "No oneDNN GPU engine found.\n";
      return 2;
    }

    auto v = dnnl_version();
    std::cout << "oneDNN version: " << v->major << "." << v->minor << "." << v->patch << "\n";

    const Options opt{};
    engine eng(engine::kind::gpu, opt.device_index);
    stream strm(eng);
    auto dev = dnnl::sycl_interop::get_device(eng);
    std::cout << "SYCL device: " << dev.get_info<sycl::info::device::name>() << "\n";

    // Minimal failing shape derived from the PyTorch issue.
    const int64_t n = 1, c = 4, h = 5, w = 5;
    const int64_t o = 1, kh = 2, kw = 3;
    const int64_t sh = 2, sw = 2, ph = 1, pw = 1, dh = 2, dw = 3;

    const int64_t src_compact_n = n * c * h * w; // 100
    const int64_t wei_compact_n = o * c * kh * kw; // 24
    const int64_t src_full_n = 2 * src_compact_n; // 200
    const int64_t wei_full_n = 2 * wei_compact_n; // 48
    const int64_t src_offset = src_compact_n; // intentionally unaligned for fp32
    const int64_t wei_offset = wei_compact_n;

    std::vector<float> src_full_h(static_cast<size_t>(src_full_n));
    std::vector<float> wei_full_h(static_cast<size_t>(wei_full_n));
    for (int64_t i = 0; i < src_full_n; ++i) {
      src_full_h[static_cast<size_t>(i)] = static_cast<float>(i) / 17.0f;
    }
    for (int64_t i = 0; i < wei_full_n; ++i) {
      wei_full_h[static_cast<size_t>(i)] = static_cast<float>(i) / 13.0f;
    }

    std::vector<float> src_compact_h(
        src_full_h.begin() + src_offset,
        src_full_h.begin() + src_offset + src_compact_n);
    std::vector<float> wei_compact_h(
        wei_full_h.begin() + wei_offset,
        wei_full_h.begin() + wei_offset + wei_compact_n);

    const int64_t oh = output_size_1d(h, kh, sh, ph, dh);
    const int64_t ow = output_size_1d(w, kw, sw, pw, dw);
    std::vector<float> ref(static_cast<size_t>(n * o * oh * ow), 0.0f);
    cpu_conv2d_ref(src_compact_h, wei_compact_h, ref, n, c, h, w, o, kh, kw, sh, sw, ph, pw, dh, dw);

    float* src_full_d = alloc_shared<float>(eng, static_cast<size_t>(src_full_n));
    float* wei_full_d = alloc_shared<float>(eng, static_cast<size_t>(wei_full_n));
    float* src_compact_d = alloc_shared<float>(eng, static_cast<size_t>(src_compact_n));
    float* wei_compact_d = alloc_shared<float>(eng, static_cast<size_t>(wei_compact_n));

    std::copy(src_full_h.begin(), src_full_h.end(), src_full_d);
    std::copy(wei_full_h.begin(), wei_full_h.end(), wei_full_d);
    std::copy(src_compact_h.begin(), src_compact_h.end(), src_compact_d);
    std::copy(wei_compact_h.begin(), wei_compact_h.end(), wei_compact_d);

    float* src_offset_ptr = src_full_d + src_offset;
    float* wei_offset_ptr = wei_full_d + wei_offset;

    std::cout << "src_offset(elements): " << src_offset
              << " ptr_mod64=" << (reinterpret_cast<uintptr_t>(src_offset_ptr) % 64) << "\n";
    std::cout << "wei_offset(elements): " << wei_offset
              << " ptr_mod64=" << (reinterpret_cast<uintptr_t>(wei_offset_ptr) % 64) << "\n";

    auto out_compact = run_onednn_conv(eng, strm, src_compact_d, wei_compact_d, n, c, h, w, o, kh, kw, sh, sw, ph, pw, dh, dw);
    auto out_offset = run_onednn_conv(eng, strm, src_offset_ptr, wei_offset_ptr, n, c, h, w, o, kh, kw, sh, sw, ph, pw, dh, dw);

    const float compact_vs_ref = max_abs_diff(out_compact, ref);
    const float offset_vs_ref = max_abs_diff(out_offset, ref);
    const float offset_vs_compact = max_abs_diff(out_offset, out_compact);

    const bool compact_control_pass = compact_vs_ref <= opt.atol;
    const bool offset_diverges = offset_vs_compact > opt.atol;

    std::cout << "compact_vs_ref max_abs=" << compact_vs_ref << "\n";
    std::cout << "offset_vs_ref max_abs=" << offset_vs_ref << "\n";
    std::cout << "offset_vs_compact max_abs=" << offset_vs_compact << "\n";

    std::cout << "ref_head: [" << ref[0] << ", " << ref[1] << ", " << ref[2] << "]\n";
    std::cout << "compact_head: [" << out_compact[0] << ", " << out_compact[1] << ", " << out_compact[2] << "]\n";
    std::cout << "offset_head: [" << out_offset[0] << ", " << out_offset[1] << ", " << out_offset[2] << "]\n";

    auto ctx = dnnl::sycl_interop::get_context(eng);
    sycl::free(src_full_d, ctx);
    sycl::free(wei_full_d, ctx);
    sycl::free(src_compact_d, ctx);
    sycl::free(wei_compact_d, ctx);

    return (compact_control_pass && offset_diverges) ? 0 : 1;
  } catch (const dnnl::error& e) {
    std::cerr << "oneDNN error: " << e.what() << " | status=" << e.status << "\n";
    return 10;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 11;
  }
}
