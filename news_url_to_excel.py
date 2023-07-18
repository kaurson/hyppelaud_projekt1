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

def remove_text(text_to_replace, text):
    replaced_text = text.replace(text_to_replace,'')
    return replaced_text

news_results = []

for i in range(10):

    entry = NewsFeed.entries[i]
    print("------ Starting new RSS article --------")
    print(f'Published: {entry.published}')
    print(f'RSS summary: {entry.summary}')
    print(f'RSS link: {entry.link}')
    print("------ Captured article content --------")
    article_content = extrac_text(entry.link)
    print(f"Article text {article_content[0:200]}")
    cleaned_article = article_content
    cleaned_article = remove_text("ERR kasutab oma veebilehtedel http küpsiseid. Kasutamist jätkates nõustute kõikide ERR-i veebilehtede küpsiste seadetega", cleaned_article)
    cleaned_article = remove_text("Parema ja terviklikuma kasutajakogemuse tagamiseks soovitame alla laadida uusim versioon mõnest meie toetatud brauserist:", cleaned_article)
    cleaned_article = remove_text("Hea lugeja, näeme et kasutate vanemat brauseri versiooni või vähelevinud brauserit.", cleaned_article)
    print(f"Cleaned article text: {cleaned_article[0:200]}")
    print(f'Cleaned article length {len(cleaned_article)}')


    inputs = tokenizer(cleaned_article, return_tensors='pt', max_length=1024)

    summary_ids = model.generate(inputs['input_ids'])
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    print('--------------summary-------------')
    print(f'article summary {summary}')

    data = {
        'published': entry.published,
        'rss_summary': entry.summary,
        'article_link': entry.link,
        'scraped_article': article_content,
        'scraped_cleaned_article': cleaned_article,
        'article_generated_summary': summary
    }
    news_results.append(data)

import pandas as pd

df = pd.DataFrame(news_results)
df.to_excel('article_results.xlsx')

