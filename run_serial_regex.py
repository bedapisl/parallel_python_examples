from config import searched_words, searched_files
import tqdm
import re


def is_match_regex(text, regex):
    result = regex.search(text)
    return bool(result)


def search_file(search_filename):
    keywords = searched_words()
    regex = re.compile('|'.join(keywords), re.UNICODE)
 
    matches = 0
    with open(search_filename, "r") as search_file:
        for line in tqdm.tqdm(search_file):
            matches += int(is_match_regex(line, regex))

    return matches


def main():
    matches = 0
    for search_filename in searched_files():
        matches += search_file(search_filename)

    print(matches)


if __name__ == "__main__":
    main()
