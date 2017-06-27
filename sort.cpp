#include <vector>
#include <chrono>
#include <algorithm>
#include <boost/compute.hpp>

// NOTE: modified from boost.compute sample

namespace compute = boost::compute;

int main()
{
  using namespace std::chrono;

  constexpr size_t size = 5000000;

  // get the default compute device
  compute::device gpu = compute::system::default_device();

  // create a compute context and command queue
  compute::context ctx(gpu);
  compute::command_queue queue(ctx, gpu);

  // generate random numbers on the host
  std::vector<float> host_vector(size);
  std::generate(host_vector.begin(), host_vector.end(), rand);

  // create vector on the device
  compute::vector<float> device_vector(size, ctx);

  // copy data to the device
  compute::copy(
                host_vector.begin(), host_vector.end(), device_vector.begin(), queue
                );

  auto beg = system_clock::now();
  std::sort(host_vector.begin(), host_vector.end());
  auto end = system_clock::now();
  std::cout << duration_cast<milliseconds>(end - beg).count() << std::endl;

  beg = system_clock::now();
  // sort data on the device
  compute::sort(
                device_vector.begin(), device_vector.end(), queue
                );
  end = system_clock::now();
  std::cout << duration_cast<milliseconds>(end - beg).count() << std::endl;

  // copy data back to the host
  compute::copy(
                device_vector.begin(), device_vector.end(), host_vector.begin(), queue
                );
}
