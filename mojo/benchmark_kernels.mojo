# Optional Mojo CPU kernel prototypes for NEXUS-BI preprocessing.
# These are not executed by the Python runtime. Enable only if a Mojo toolchain
# is installed and validated for the host.

from memory.unsafe import DTypePointer
from sys.intrinsics import strided_load
from runtime.ffi import destroy

fn byte_histogram(bytes: DTypePointer[DType.uint8], length: Int, out: DTypePointer[DType.uint64]) -> None:
    for i in range(256):
        out[i] = 0
    for i in range(length):
        out[bytes[i]] += 1

fn dot_f32(left: DTypePointer[DType.float32], right: DTypePointer[DType.float32], length: Int) -> Float32:
    var total: Float32 = 0.0
    for i in range(length):
        total += left[i] * right[i]
    return total

fn histogram_dot_selfcheck(bytes: DTypePointer[DType.uint8], length: Int) -> UInt64:
    var total: UInt64 = 0
    for i in range(length):
        total += 1
    return total
