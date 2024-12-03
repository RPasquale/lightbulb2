# settings.py

# Scrapy configurations
BOT_NAME = 'my_search_engine'
SPIDER_MODULES = ['my_search_engine.spiders']
NEWSPIDER_MODULE = 'my_search_engine.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy
CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 1

# Configure a delay for requests to the same website
DOWNLOAD_DELAY = 2  # Fixed delay of 2 seconds

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Enable AutoThrottle
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1  # Initial download delay
AUTOTHROTTLE_MAX_DELAY = 10  # Maximum download delay in case of high latencies
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0  # Average number of requests Scrapy should be sending in parallel

# User Agent (default, only used if RotateUserAgentMiddleware fails)
USER_AGENT = "Mozilla/5.0 (compatible; MySearchEngine/1.0)"

# Downloader middlewares
DOWNLOADER_MIDDLEWARES = {
    'my_search_engine.middlewares.RotateUserAgentMiddleware': 543,
    # Uncomment the following line if using the ProxyMiddleware
    # 'my_search_engine.middlewares.ProxyMiddleware': 544,
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,  # Disable default user agent middleware
}

# Item pipelines
ITEM_PIPELINES = {
    'my_search_engine.pipelines.SaveToJSONPipeline': 300,
    'my_search_engine.pipelines.ContentCleanupPipeline': 400,
    'my_search_engine.pipelines.DisplayResultsPipeline': 200,
}

# Enable logging
LOG_ENABLED = True
LOG_LEVEL = 'INFO'  # Set to 'DEBUG' to see all logs, including middleware logs

# Additional settings can be added below as needed
