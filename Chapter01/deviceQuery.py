import pycuda.driver as cuda
import pycuda.autoinit

def query_device_info():
    # Get the number of CUDA devices
    device_count = cuda.Device.count()

    print(f"CUDA device query (PyCUDA version)\n")
    print(f"Detected {device_count} CUDA Capable device(s)\n")

    for i in range(device_count):
        gpu = cuda.Device(i)
        print(f"Device {i}:", gpu.name())

if __name__ == "__main__":
    query_device_info()

"""
def query_device_info():
    # Get the number of CUDA devices
    device_count = cuda.Device.count()

    print(f"Number of CUDA devices: {device_count}")

    for i in range(device_count):
        gpu = cuda.Device(i)
        print(f"\nDevice {i + 1}:", gpu.name())

        compute_capability = ".".join(map(str, gpu.compute_capability()))
        print(f"Compute Capability: {compute_capability}")

        total_memory = gpu.total_memory() // (1024**2)  # Convert bytes to megabytes
        print(f"Total Memory: {total_memory} MB")

        multi_processor_count = gpu.multiprocessor_count()
        print(f"Multiprocessors: {multi_processor_count}")

        # Other device properties can be queried similarly
        # See the pycuda documentation for more details

if __name__ == "__main__":
    query_device_info()
"""