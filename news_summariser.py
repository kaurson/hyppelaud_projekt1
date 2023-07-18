from newspaper import Article
from newspaper import Config
import gradio as gr
from gradio.mix import Parallel, Series



def extrac_text(url):
  USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
  config = Config()
  config.browser_user_agent = USER_AGENT
  config.request_timeout = 10

  article = Article(url, config=config)
  article.download()
  article.parse()
  text = article.text
  return text

extractor = gr.Interface(extrac_text, 'text', 'text')
summarizer = gr.Interface.load("huggingface/facebook/bart-large-cnn")

sample_url = [['https://www.bbc.com/news/world-us-canada-66226103'],
]

desc =  '''
         The news summarizer app uses bart-large-cnn model by Facebook to summarize the text of a news article.
        '''

iface = Series(extractor, summarizer,
  inputs = gr.inputs.Textbox(
      lines = 2,
      label = 'Enter URL below'
  ),
  outputs = 'text',
  title = 'News Summarizer',
  theme = 'grass',
  layout = 'horizontal',
  description = desc,
  examples=sample_url)

iface.launch()