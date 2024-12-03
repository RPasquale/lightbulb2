# search_spider.py
import scrapy
from bs4 import BeautifulSoup
from my_search_engine.my_search_engine.items import MySearchEngineItem
import random
from urllib.parse import urlparse, urljoin
import traceback
import re
from twisted.internet.error import TCPTimedOutError, ConnectionRefusedError, TimeoutError
from scrapy.exceptions import CloseSpider
import logging

logger = logging.getLogger(__name__)

class SearchSpider(scrapy.Spider):
    name = "search_spider"
    allowed_domains = []  # To be set dynamically from search_sites

    def __init__(self, query=None, search_sites=None, max_depth=2, max_links_per_page=3, *args, **kwargs):
        super(SearchSpider, self).__init__(*args, **kwargs)
        self.query = query
        if not self.query:
            raise CloseSpider("No search query provided")
        self.max_depth = max_depth
        self.max_links_per_page = max_links_per_page
        if search_sites is None:
            self.search_sites = [
            f"https://en.wikibooks.org/w/index.php?search={self.query}",
            f"https://en.wikiversity.org/w/index.php?search={self.query}",
            f"https://commons.wikimedia.org/w/index.php?search={self.query}",
            f"https://stackexchange.com/search?q={self.query}",
            f"https://arxiv.org/search/?query={self.query}&searchtype=all",
            f"https://www.ncbi.nlm.nih.gov/pmc/?term={self.query}",
            f"https://www.gutenberg.org/ebooks/search/?query={self.query}",
            f"https://openlibrary.org/search?q={self.query}",
            f"https://doaj.org/search/articles?ref=homepage&q={self.query}",
            f"https://www.ted.com/search?q={self.query}",
            f"https://en.citizendium.org/wiki?search={self.query}",
            f"https://www.jstor.org/action/doBasicSearch?Query={self.query}",
            f"https://archive.org/search.php?query={self.query}",
            f"https://search.scielo.org/?q={self.query}",
            f"https://paperswithcode.com/search?q={self.query}",
            f"https://www.reddit.com/search/?q={self.query}",
            f"https://huggingface.co/models?search={self.query}",
            f"https://huggingface.co/datasets?search={self.query}",
            f"https://machinelearningmastery.com/?s={self.query}",
            f"https://www.kaggle.com/search?q={self.query}",
            f"https://towardsdatascience.com/search?q={self.query}",
            f"https://github.com/search?q={self.query}",
            f"https://stackoverflow.com/search?q={self.query}",
            f"https://www.youtube.com/results?search_query={self.query}",
            f"https://www.slideshare.net/search/slideshow?searchfrom=header&q={self.query}"
        ]

        else:
            self.search_sites = search_sites


    def start_requests(self):
        if not self.query:
            logger.error("No search query provided in start_requests")
            return

        self.allowed_domains = list(set([urlparse(url).netloc for url in self.search_sites]))
        logger.info(f"Starting requests for query: {self.query}")

        for url in self.search_sites:
            yield scrapy.Request(
                url,
                callback=self.parse,
                meta={
                    'dont_retry': True,
                    'handle_httpstatus_list': [302, 403, 404, 420, 429, 500, 503],
                    'depth': 1  # Start at depth 1
                },
                errback=self.errback_httpbin
            )

    def parse(self, response):
        depth = response.meta.get('depth', 1)
        if depth > self.max_depth:
            logger.debug(f"Reached max depth at {response.url}")
            return

        logger.info(f"Parsing response from {response.url} at depth {depth}")

        try:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Check for irrelevant or blocked content
            if any(term in soup.text.lower() for term in ['captcha', 'verification', 'no items found', 'no results', 'access denied']):
                logger.warning(f"Irrelevant page detected: {response.url}")
                return

            title = soup.find('title').get_text().strip() if soup.find('title') else 'No title'
            meta_description = soup.find('meta', {'name': 'description'})
            meta_description = meta_description['content'].strip() if meta_description else 'No description'

            content = self.extract_main_content(soup)
            summary = self.generate_summary(content, 200)
            total_links = len(soup.find_all('a', href=True))
            content_length = len(content.split())

            if content_length < 100:
                logger.info(f"Content too short ({content_length} words) for {response.url}")
                return

            item = MySearchEngineItem()
            item['title'] = title
            item['link'] = response.url
            item['content'] = content
            item['summary'] = summary
            item['meta'] = {
                'description': meta_description,
                'total_links': total_links,
                'content_length': content_length,
                'domain': urlparse(response.url).netloc,
            }
            yield item

            # Limit the number of links per page
            links = soup.find_all('a', href=True)
            random.shuffle(links)
            links = links[:self.max_links_per_page]  # Limit the number of links

            for link in links:
                href = link.get('href')
                full_url = urljoin(response.url, href)
                if self.is_valid_link(full_url):
                    logger.debug(f"Following link: {full_url}")
                    yield scrapy.Request(
                        url=full_url,
                        callback=self.parse,
                        meta={'depth': depth + 1},
                        errback=self.errback_httpbin
                    )
        except Exception as e:
            logger.error(f"Error parsing {response.url}: {str(e)}")
            logger.error(traceback.format_exc())

    def extract_main_content(self, soup):
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')

        if main_content:
            return ' '.join(main_content.stripped_strings)

        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text().strip() for p in paragraphs])

    def generate_summary(self, content, max_length=200):
        sentences = re.split(r'(?<=[.!?])\s+', content)
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) <= max_length:
                summary += sentence + " "
            else:
                break
        return summary.strip()

    def is_valid_link(self, url):
        parsed_url = urlparse(url)
        return any(domain in parsed_url.netloc for domain in self.allowed_domains)

    def errback_httpbin(self, failure):
        logger.error(f"Error on {failure.request.url}: {str(failure.value)}")
        logger.error(traceback.format_exc())

        if failure.check(ConnectionRefusedError):
            logger.warning(f"Connection refused: {failure.request.url}")
        elif failure.check(TimeoutError, TCPTimedOutError):
            logger.warning(f"Timeout: {failure.request.url}")
        else:
            logger.error(f"Failed to process: {failure.request.url}")