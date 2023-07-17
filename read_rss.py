from rss_parser import Parser
from requests import get


def get_news():
    rss_url = "http://feeds.bbci.co.uk/news/world/rss.xml"
    response = get(rss_url)

    rss = Parser.parse(response.text)

    # Print out rss meta data
    print("Language", rss.channel.language)
    print("RSS", rss.version)
    rss_titles = []
    # Iteratively print feed items
    for item in rss.channel.items:
        rss_titles.append([item.title.content])
        print(rss_titles)
    return rss_titles
