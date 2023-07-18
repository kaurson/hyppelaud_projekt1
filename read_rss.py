from rss_parser import Parser
from requests import get


def get_news():
    rss_url = "https://www.err.ee/rss"
    response = get(rss_url)
    rss_titles_10 = []
    rss = Parser.parse(response.text)

    # Print out rss meta data
    print("Language", rss.channel.language)
    print("RSS", rss.version)
    rss_titles = []
    # Iteratively print feed items
    for item in rss.channel.items:
        rss_titles.append([item.title.content])
    rss_titles_10 = rss_titles[:10]
    return rss_titles_10


print(get_news())
