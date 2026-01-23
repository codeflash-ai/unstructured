from functools import lru_cache

from bs4 import BeautifulSoup


def indent_html(html_string: str, html_parser="html.parser") -> str:
    """
    Formats / indents HTML.

    This function takes an HTML string and formats it using the specified HTML parser.
    It parses the HTML content and returns a prettified version of it.

    Args:
        html_string (str): The HTML content to be formatted.
        html_parser (str, optional): The parser to use for parsing the HTML. Defaults to 'html5lib':
            - 'html.parser': The built-in HTML parser. Use when you need just parsing
            - 'html5lib': The slowest. Use when you expect valid HTML parsed
                          the same way a browser does. It adds some extra
                          tags and attributes like <html>, <head>, <body>
            More in docs https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-a-parser

    Returns:
        str: The formatted and indented HTML content.
    """
    # Use a small LRU cache to avoid reparsing identical inputs repeatedly.
    pretty_html = _cached_prettify(html_string, html_parser)
    return pretty_html


@lru_cache(maxsize=128)
def _cached_prettify(html_string: str, html_parser: str) -> str:
    soup = BeautifulSoup(html_string, html_parser)
    pretty_html = soup.prettify()
    return pretty_html
