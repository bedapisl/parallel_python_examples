from config import searched_words, searched_files
import tqdm
import pycuda.autoinit
import pycuda.driver as cuda 
import pycuda.gpuarray as gpuarray
import numpy as np

from pycuda.compiler import SourceModule


def cuda_block_search():
    """
    Input data has to end with \n
    """
    with open("block_search.cu", "r") as kernel_definition:
        cuda_module = SourceModule(kernel_definition.read())

    return cuda_module.get_function("block_search")


def search_cuda_fast(search_filename):
    BLOCK_SIZE = 100000000
    GRID_SIZE = 5000

    block_search = cuda_block_search()

    patterns, pattern_starts = encode_strings([bytes(string, 'utf-8') for string in searched_words()])
    patterns_gpu = gpuarray.to_gpu(patterns)
    pattern_starts_gpu = gpuarray.to_gpu(pattern_starts)

    matches = np.zeros(GRID_SIZE, dtype=np.int32)  # Must be higher than max id in kernel
    matches_gpu = gpuarray.to_gpu(matches)

    matches_count = 0
    keyword_count = len(searched_words())
    
    with open(search_filename, "rb") as search_file:
        while True:
            block = search_file.read(BLOCK_SIZE)
            if not block:
                break

            last_newline = block.rfind(b'\n')
            final_block = block
            final_block_gpu = gpuarray.to_gpu(np.array(final_block))

            if len(block) == BLOCK_SIZE:
                search_file.seek(last_newline - len(block), 1)

            block_search(final_block_gpu, np.int32(last_newline), patterns_gpu, pattern_starts_gpu, np.int32(len(searched_words())), np.int32(BLOCK_SIZE / GRID_SIZE), matches_gpu, block=(keyword_count, 1, 1), grid=(GRID_SIZE, 1))

    result = pycuda.gpuarray.sum(matches_gpu).get()

    print("File done")

    return result


def encode_strings(strings):
    converted = strings
    result_strings = b"".join(converted)
    lengths = np.zeros(len(converted) + 1, dtype=np.int32)
    for i, string in enumerate(converted):
        lengths[i + 1] = len(string)

    starts = np.cumsum(lengths, dtype=np.int32)
    return np.array(result_strings), starts


def main():
    matches = 0
    for search_filename in searched_files():
        matches += search_cuda_fast(search_filename)

    print(matches)

if __name__ == "__main__":
    main()
