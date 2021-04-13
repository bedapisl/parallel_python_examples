from config import searched_words, searched_files
import tqdm
from flashtext import KeywordProcessor


def is_match_flashtext(text, processor):
    """
    Gives wrong results because it searches only for keywords separated by whitespaces
    """
    result = processor.extract_keywords(text)
    return bool(result)


def search_file(search_filename):
    keywords = searched_words()
    flashtext_processor = KeywordProcessor()
    for keyword in keywords:
        flashtext_processor.add_keyword(keyword)
 
    matches = 0
    with open(search_filename, "r") as search_file:
        for line in tqdm.tqdm(search_file):
            matches += int(is_match_flashtext(line, flashtext_processor))

    return matches


def main():
    matches = 0
    for search_filename in searched_files():
        matches += search_file(search_filename)

    print(matches)


if __name__ == "__main__":
    main()
