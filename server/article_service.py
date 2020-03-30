from newspaper import Article
import nltk

nltk.download('punkt')

class ArticleService():
    def __init__(self):
        pass
    
    def get_article(self, url):
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        return article
        
    def get_article_text(self, url):
        article = Article(url)
        article.download()
        article.parse()
        return article.text