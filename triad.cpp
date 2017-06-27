#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <boost/compute.hpp>

namespace compute = boost::compute;

const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
  __kernel void triad(__global double* x,
                      __global double* y,
                      __global double* z,
                      const double s,
                      const size_t size) {
    const size_t gid = get_global_id(0);
    if (gid < size) {
      z[gid] = s * x[gid] + y[gid];
    }
  }
);

int main() {
  using namespace std::chrono;
  constexpr size_t size = 5000000;

  std::mt19937 mt;
  std::uniform_real_distribution<> urd(0.0, 1.0);

  std::vector<double> x_host(size), y_host(size), z_host(size);
  std::generate(x_host.begin(), x_host.end(), [&mt, &urd](){return urd(mt);});
  std::generate(y_host.begin(), y_host.end(), [&mt, &urd](){return urd(mt);});
  std::fill(z_host.begin(), z_host.end(), 0.0);

  compute::device gpu = compute::system::default_device();
  compute::context ctx(gpu);
  compute::command_queue queue(ctx, gpu);

  auto triad_kernel = compute::kernel::create_with_source(source, "triad", ctx);

  compute::vector<double> x_dev(size, ctx), y_dev(size, ctx), z_dev(size, ctx);

  compute::copy(x_host.begin(), x_host.end(), x_dev.begin(), queue);
  compute::copy(y_host.begin(), y_host.end(), y_dev.begin(), queue);
  compute::copy(z_host.begin(), z_host.end(), z_dev.begin(), queue);

  triad_kernel.set_args(x_dev, y_dev, z_dev, 2.0, size);
  queue.enqueue_1d_range_kernel(triad_kernel, 0, size, 0);

  compute::copy(z_dev.begin(), z_dev.end(), z_host.begin(), queue);

  // show results
  std::cout << "z = 2.0 * x + y\n";
  for (int i = 0; i < 10; i++) {
    std::cout << z_host[i] << " = " << "2.0 * " << x_host[i] << " + " << y_host[i] << std::endl;
  }
}
