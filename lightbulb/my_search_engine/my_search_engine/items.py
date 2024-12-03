import scrapy

class MySearchEngineItem(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
    content = scrapy.Field()
    score = scrapy.Field()  # Will be set later during ranking (MCTS or NLP)
    meta = scrapy.Field()
    predicted_score = scrapy.Field()
    summary = scrapy.Field()