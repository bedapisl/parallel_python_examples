 __global__ void block_search(char* input_data, int input_length, char* patterns, int* pattern_starts, int pattern_count, int chunk_size, int* out_matches) {

        char SEPARATOR = '\n';

        int id = blockIdx.x;
        int pattern_id = threadIdx.x;

        int index = id * chunk_size;
        int end_index = (id + 1) * chunk_size;

        if (index >= input_length) {
            return;
        }

        //Find start of first message
        if (index != 0) {  // First doesn't start  with separator
            while (input_data[index] != SEPARATOR) {
                index++;
            }
        }
        
        if (index >= end_index) {
            return;
        }

        __shared__ int is_match;
        is_match = false;

        while (index < input_length) {
            __syncthreads();
            if (index >= end_index && input_data[index] == SEPARATOR) {
                return;
            }

            int pattern_is_match = true;
            int pattern_length = pattern_starts[pattern_id + 1] - pattern_starts[pattern_id];
            for (int char_index=0; char_index < pattern_length; ++char_index) {
                if (patterns[pattern_starts[pattern_id] + char_index] != input_data[char_index + index]) {
                    pattern_is_match = false;
                    break;
                }
            }

            __syncthreads();
            if (pattern_is_match) {
                is_match = true;
            }
            __syncthreads();

            if (is_match) {

                if (threadIdx.x == 0) { // just once
                    out_matches[id] += 1;
                }

                __syncthreads();
                is_match = false;
                __syncthreads();

                // Go to next message
                while (input_data[index] != SEPARATOR) {
                    index++;
                }
            } else {
                index++;
            }
        }
    }

