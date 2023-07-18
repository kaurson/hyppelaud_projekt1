import feedparser
from newspaper import Article
from newspaper import Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("TalTechNLP/mBART-ERRnews")
model = AutoModelForSeq2SeqLM.from_pretrained("TalTechNLP/mBART-ERRnews")

NewsFeed = feedparser.parse("https://www.err.ee/rss")


def extrac_text(url):
    USER_AGENT = 'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 7_3_9) AppleWebKit/533.43 (KHTML, like Gecko) Chrome/50.0.3156.245 Safari/601'
    config = Config()
    config.browser_user_agent = USER_AGENT
    config.request_timeout = 20

    article = Article(url, config=config)
    article.download()
    article.parse()
    text = article.text
    return text


for i in range(10):
    entry = NewsFeed.entries[i]
    print(entry.published)
    print("******")
    print(entry.summary)
    print("------News Link--------")
    print(entry.link)
    print(extrac_text(entry.link))
    inputs = tokenizer(extrac_text(entry.link), return_tensors='pt', max_length=1024)

    summary_ids = model.generate(inputs['input_ids'])
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    print(summary)
