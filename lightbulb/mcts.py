# mcts.py
import math
import random
from nltk.corpus import wordnet
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from twisted.internet import reactor, defer
from scrapy import signals
import logging
from my_search_engine.my_search_engine.spiders.search_spider import SearchSpider
from sentence_transformers import SentenceTransformer, util
from ranking import train_ranking_model
import time

logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.ucb_score = float('inf')

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, child_state, action=None):
        child_node = MCTSNode(child_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:  # Only calculate UCB if not root
            self.ucb_score = self.calculate_ucb()

    def calculate_ucb(self, exploration_weight=1.41):
        if self.visits == 0 or not self.parent:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class MCTS:
    def __init__(self, initial_state, num_simulations=20, exploration_weight=1.41):
        self.root = MCTSNode(initial_state)
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.query_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.results = []
        self.crawler_runner = CrawlerRunner(get_project_settings())
        self.initial_state = initial_state
        self.num_iterations = 5

    def select(self, node):
        while not node.is_leaf():
            if not node.children:
                return node
            node = max(node.children, key=lambda c: c.calculate_ucb(self.exploration_weight))
        return node

    def expand(self, node):
        if node.visits == 0:
            return node
        possible_refinements = self.get_possible_refinements(node.state)
        for refinement in possible_refinements:
            node.add_child(refinement)
        return random.choice(node.children) if node.children else node

    def calculate_combined_reward(self, ranking_score, state):
        state_length_reward = len(state) / 100
        if state:
            query_complexity = len(set(state.split())) / len(state.split())
        else:
            query_complexity = 0
        semantic_similarity = self.calculate_semantic_similarity(state, self.root.state)
        
        combined_reward = (
            0.5 * ranking_score +
            0.2 * state_length_reward +
            0.2 * query_complexity +
            0.1 * semantic_similarity
        )
        return combined_reward

    def calculate_semantic_similarity(self, query1, query2):
        embedding1 = self.query_model.encode(query1)
        embedding2 = self.query_model.encode(query2)
        return util.pytorch_cos_sim(embedding1, embedding2).item()

    def backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def best_action(self):
        if not self.root.children:
            return self.root

        def score(node):
            if node.visits == 0:
                return float('-inf')
            return node.value / node.visits

        return max(self.root.children, key=score)

    def refine_query(self, query):
        words = query.split()
        refined_query = []
        
        for word in words:
            if word.lower() not in {"how", "to", "get", "an", "the", "and", "or", "of", "build"}:
                synonyms = wordnet.synsets(word)
                if synonyms:
                    synonym_words = [lemma.name() for lemma in synonyms[0].lemmas() 
                                    if len(lemma.name().split()) == 1 and word != lemma.name()]
                    if synonym_words:
                        refined_query.append(random.choice(synonym_words))
                    else:
                        refined_query.append(word)
                else:
                    refined_query.append(word)
            else:
                refined_query.append(word)
        
        possible_intent_keywords = ['guide', 'tutorial', 'LLM', 'language model', 'NLP', 'GPT']
        refined_query.append(random.choice(possible_intent_keywords))
        
        return ' '.join(refined_query)

    def get_related_queries(self, query):
        query_embedding = self.query_model.encode(query)
        refined_query_variations = [query]
        words_to_avoid = {'how', 'to', 'get'}
        words = query.split()
        
        for word in words:
            if word.lower() not in words_to_avoid:
                synonyms = wordnet.synsets(word)
                if synonyms:
                    synonym_words = [lemma.name() for lemma in synonyms[0].lemmas() if lemma.name() != word]
                    if synonym_words:
                        refined_query = query.replace(word, random.choice(synonym_words))
                        refined_query_variations.append(refined_query)

        refined_query_variations = list(set(refined_query_variations))
        refined_query_embeddings = [self.query_model.encode(variation) for variation in refined_query_variations]
        similarity_scores = util.pytorch_cos_sim(query_embedding, refined_query_embeddings).tolist()[0]

        similarity_threshold = 0.8
        filtered_queries = [variation for idx, variation in enumerate(refined_query_variations)
                            if similarity_scores[idx] > similarity_threshold]
        
        return filtered_queries[:2] if filtered_queries else [query]

    def get_possible_refinements(self, query):
        refined_queries = self.get_related_queries(query)
        return refined_queries + [self.refine_query(query)]

    @defer.inlineCallbacks
    def web_search(self, query, search_sites=None):
        if not query.strip():
            logger.error("Cannot perform web search with an empty query.")
            defer.returnValue([])
        
        logger.info(f"Starting web search for query: {query}")
        configure_logging(install_root_handler=False)
        logging.basicConfig(level=logging.INFO)

        results = []

        def crawler_results(item, response, spider):
            logger.info(f"Received result: {item['title']}")
            results.append(item)

        try:
            crawler = self.crawler_runner.create_crawler(SearchSpider)
            crawler.signals.connect(crawler_results, signal=signals.item_scraped)
            
            # Start crawling, passing query and search_sites to the spider
            yield self.crawler_runner.crawl(crawler, query=query, search_sites=search_sites)
        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            defer.returnValue([])

        logger.info(f"Web search completed. Found {len(results)} results.")
        defer.returnValue(results)

    @defer.inlineCallbacks
    def run(self):
        logger.info(f"Starting MCTS run with {self.num_iterations} iterations")
        for i in range(self.num_iterations):
            logger.debug(f"Iteration {i+1}/{self.num_iterations}")
            leaf = self.select(self.root)
            child = self.expand(leaf)
            reward = yield self.simulate(child)
            self.backpropagate(child, reward)

        best_child = self.best_action()
        logger.info(f"MCTS run completed. Best action: {best_child.state}")
        defer.returnValue(best_child.state if best_child != self.root else self.root.state)
        
    @defer.inlineCallbacks
    def simulate(self, node):
        query_results = yield self.web_search(node.state)
        ranked_results = train_ranking_model(node.state, query_results)

        if ranked_results:
            top_score = ranked_results[0]['predicted_score']
        else:
            top_score = 0

        reward = self.calculate_combined_reward(top_score, node.state)
        defer.returnValue(reward)






