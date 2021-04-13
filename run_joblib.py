from config import searched_words, searched_files
import tqdm
from joblib import Parallel, delayed


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


def main():
    matches = Parallel(n_jobs=3)(delayed(search_file)(search_filename) for search_filename in searched_files())
    print(sum(matches))


if __name__ == "__main__":
    main()
