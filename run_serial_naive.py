from config import searched_words, searched_files
import tqdm


def is_match(text, keywords):
    for keyword in keywords:
        if keyword in text:
            return True

    return False


def search_file(search_filename):
    keywords = searched_words()
    matches = 0
    with open(search_filename, "r") as search_file:
        for i, line in enumerate(tqdm.tqdm(search_file)):
            matches += int(is_match(line, keywords))

    return matches


def main():
    matches = 0
    for search_filename in searched_files():
        matches += search_file(search_filename)

    print(matches)


if __name__ == "__main__":
    main()
