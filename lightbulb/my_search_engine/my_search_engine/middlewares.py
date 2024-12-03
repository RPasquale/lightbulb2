# middlewares.py

import random
import logging

logger = logging.getLogger(__name__)

class RotateUserAgentMiddleware:
    """Middleware for rotating user agents to avoid detection."""

    USER_AGENTS = [
        # Chrome User Agents
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
        " Chrome/93.0.4577.63 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
        " Chrome/93.0.4577.63 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
        " Chrome/93.0.4577.63 Safari/537.36",
        # Firefox User Agents
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11.5; rv:92.0) Gecko/20100101 Firefox/92.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0",
        # Safari User Agents
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1",
        # Edge User Agents
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36 Edg/93.0.961.38",
        # Opera User Agents
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36 OPR/78.0.4093.184",
        # Mobile User Agents
        "Mozilla/5.0 (Linux; Android 11; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.62 Mobile Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/93.0.4577.62 Mobile/15E148 Safari/604.1",
        # Add more user agents from different browsers and devices
    ]

    def process_request(self, request, spider):
        """Assign a random user agent to each request."""
        user_agent = random.choice(self.USER_AGENTS)
        request.headers['User-Agent'] = user_agent
        logger.debug(f"Using User-Agent: {user_agent}")

# Optional: Proxy Middleware
class ProxyMiddleware:
    """Middleware for rotating proxies."""

    PROXIES = [
        # Add proxy URLs if using proxies
        # 'http://proxy1.example.com:8000',
        # 'http://proxy2.example.com:8031',
    ]

    def process_request(self, request, spider):
        if self.PROXIES:
            proxy = random.choice(self.PROXIES)
            request.meta['proxy'] = proxy
            logger.debug(f"Using Proxy: {proxy}")
