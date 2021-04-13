from config import searched_words, searched_files
import tqdm
import numpy as np
from numba import cuda, float32, boolean


@cuda.jit
def block_search(input_data, input_length, patterns, pattern_starts, pattern_count, chunk_size, out_matches):
    SEPARATOR = 10  # This is newline

    chunk_size = chunk_size[0]
    input_length = input_length[0]
    pattern_count = pattern_count[0]

    block_id = cuda.blockIdx.x
    pattern_id = cuda.threadIdx.x

    index = block_id * chunk_size;
    end_index = (block_id + 1) * chunk_size;

    if index >= input_length:
        return

    #Find start of first message
    if index != 0:  # First doesn't start  with separator
        while input_data[index] != SEPARATOR:
            index += 1
    
    if index >= end_index:
        return

    is_match = cuda.shared.array((1), boolean)
    is_match[0] = False

    while index < input_length:
        cuda.syncthreads();
        if index >= end_index and input_data[index] == SEPARATOR:
            return

        pattern_is_match = True
        pattern_length = pattern_starts[pattern_id + 1] - pattern_starts[pattern_id]
        for char_index in range(0, pattern_length):
            if patterns[pattern_starts[pattern_id] + char_index] != input_data[char_index + index]:
                pattern_is_match = False
                break

        cuda.syncthreads();

        if pattern_is_match:
            is_match[0] = True
        
        cuda.syncthreads();

        if is_match[0]:
            if cuda.threadIdx.x == 0:  #just once
                out_matches[block_id] += 1

            cuda.syncthreads()
            is_match[0] = False
            cuda.syncthreads()

            # Go to next message
            while input_data[index] != SEPARATOR:
                index += 1
        else:
            index += 1


def search_cuda_fast(search_filename):
    BLOCK_SIZE = 100000000
    GRID_SIZE = 5000

    patterns, pattern_starts = encode_strings([bytes(string, 'utf-8') for string in searched_words()])
    pattern_gpu = cuda.to_device(patterns)
    pattern_starts_gpu = cuda.to_device(pattern_starts)

    matches = np.zeros(GRID_SIZE, dtype=np.int32)
    matches_gpu = cuda.to_device(matches)

    keyword_count = len(searched_words())
    
    with open(search_filename, "rb") as search_file:
        while True:
            block = search_file.read(BLOCK_SIZE)
            if not block:
                break

            last_newline = block.rfind(b'\n')
            final_block = block
            final_block_gpu = cuda.to_device(np.frombuffer(final_block, dtype=np.ubyte))

            if len(block) == BLOCK_SIZE:
                search_file.seek(last_newline - len(block), 1)

            block_search[GRID_SIZE, keyword_count](final_block_gpu, cuda.to_device(np.array([last_newline], dtype=np.int32)), pattern_gpu, pattern_starts_gpu, cuda.to_device(np.array([keyword_count], dtype=np.int32)), cuda.to_device(np.array([BLOCK_SIZE / GRID_SIZE], dtype=np.int32)), matches_gpu)

    result_array = matches_gpu.copy_to_host()
    result = result_array.sum()

    print("File done")

    return result


def encode_strings(strings):
    converted = strings
    result_strings = b"".join(converted)
    lengths = np.zeros(len(converted) + 1, dtype=np.int32)
    for i, string in enumerate(converted):
        lengths[i + 1] = len(string)

    starts = np.cumsum(lengths, dtype=np.int32)
    return np.frombuffer(result_strings, dtype=np.ubyte), starts


def main():
    matches = 0
    for search_filename in searched_files():
        matches += search_cuda_fast(search_filename)

    print(matches)

if __name__ == "__main__":
    main()
