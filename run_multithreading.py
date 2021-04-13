from config import searched_words, searched_files
import tqdm
import threading
import queue


lock = threading.RLock()
files_read = 0
global_matches = 0


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


def text_reading_worker(filename, text_queue):
    global files_read
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

    with lock:
        files_read += 1


def searching_worker(text_queue, file_count):
    global files_read
    global global_matches
    keywords = searched_words()
    matches = 0
    while files_read != file_count:  # if some files are still not read till the end, continue
        while True:
            try:
                batch = text_queue.get(True, 5.0)
            except queue.Empty as e:
                break

            for text in batch:
                if is_match(text, keywords):
                    matches += 1

    with lock:
        global_matches += matches


def main():
    global global_matches
    text_queue = queue.Queue(10)

    for search_filename in searched_files():
        t = threading.Thread(target=text_reading_worker, args=(search_filename, text_queue))
        t.start()

    searching_workers = []
    for i in range(6):
        t = threading.Thread(target=searching_worker, args=(text_queue, len(searched_files())))
        t.start()
        searching_workers.append(t)
        

    for t in searching_workers:
        t.join()


    print(global_matches)


if __name__ == "__main__":
    main()
