from config import searched_words, searched_files
import tqdm
import multiprocessing
import queue


def is_match(text, keywords):
    for keyword in keywords:
        if keyword in text:
            return True

    return False


def search_file(search_filename):
    keywords = searched_words()
    matches = 0
    with open(search_filename, "r") as search_file:
        for line in tqdm.tqdm(search_file):
            matches += int(is_match(line, keywords))

    return matches


def text_reading_process(filename, text_queue, files_read):
    batch_size = 512
    batch = []
    with open(filename, 'r') as search_file:
        for line in tqdm.tqdm(search_file):
            batch.append(line)

            if len(batch) == batch_size:
                text_queue.put(batch)
                batch = []

    if batch:
        text_queue.put(batch)

    with files_read.get_lock():
        files_read.value += 1

    print("Reading process finished")


def searching_process(text_queue, matches_shared, files_read, file_count):
    keywords = searched_words()
    matches = 0
    while files_read.value != file_count:
        while True:
            try:
                batch = text_queue.get(True, 5.0)
            except queue.Empty as e:
                break

            for text in batch:
                if is_match(text, keywords):
                    matches += 1

    with matches_shared.get_lock():
        matches_shared.value += matches

    print("Searching process finished")

def main():
    text_queue = multiprocessing.Queue(10)
    matches = multiprocessing.Value('i', 0)  # in shared memory, contains lock
    files_read = multiprocessing.Value('i', 0)

    for search_filename in searched_files():
        p = multiprocessing.Process(target=text_reading_process, args=(search_filename, text_queue, files_read))
        p.start()

    searching_processes = []
    for i in range(10):
        p = multiprocessing.Process(target=searching_process, args=(text_queue, matches, files_read, len(searched_files())))
        p.start()
        searching_processes.append(p)
        

    for p in searching_processes:
        p.join()


    print(matches.value)


if __name__ == "__main__":
    main()
