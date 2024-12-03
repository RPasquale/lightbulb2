# test_agent.py

import logging
from twisted.internet import reactor, defer, threads
from agent import AutonomousWebAgent
from ToTSearch import ToTSearch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the logger
logger = logging.getLogger(__name__)

# Suppress detailed logs for some libraries (like Scrapy or Transformers)
logging.getLogger('scrapy').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('twisted').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class TestAgent:
    def __init__(self):
        # Initialize the AutonomousWebAgent
        state_size = 7  # word_count, link_count, header_count, semantic_similarity, image_count, script_count, css_count
        action_size = 3  # 0: Click Link, 1: Summarize, 2: RAG Generate
        num_options = 3  # 0: Search, 1: Summarize, 2: RAG Generate

        self.agent = AutonomousWebAgent(
            state_size=state_size,
            action_size=action_size,
            num_options=num_options,
            hidden_size=64,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            knowledge_base_path='knowledge_base.json'
        )

        # Initialize ToTSearch with the agent
        self.tot_search = ToTSearch(self.agent)

        # Few-shot examples for Tree of Thoughts
        self.few_shot_examples = [
            {
                "query": "What are the effects of climate change on biodiversity?",
                "thoughts": [
                    "Loss of habitats due to rising sea levels and changing temperatures",
                    "Disruption of ecosystems and food chains",
                    "Increased extinction rates for vulnerable species"
                ],
                "answer": "Climate change significantly impacts biodiversity through habitat loss, ecosystem disruption, and increased extinction rates. Rising temperatures and sea levels alter habitats, forcing species to adapt or migrate. This disrupts established ecosystems and food chains. Species unable to adapt quickly face a higher risk of extinction, particularly those with specialized habitats or limited ranges."
            },
            {
                "query": "How can we promote sustainable energy adoption?",
                "thoughts": [
                    "Government policies and incentives",
                    "Public awareness and education campaigns",
                    "Technological advancements and cost reduction"
                ],
                "answer": "Promoting sustainable energy adoption requires a multi-faceted approach. Government policies and incentives can encourage both businesses and individuals to switch to renewable sources. Public awareness and education campaigns help people understand the importance and benefits of sustainable energy. Continued technological advancements and cost reductions make sustainable energy more accessible and economically viable for widespread adoption."
            }
        ]

    @defer.inlineCallbacks
    def process_query(self, query, is_few_shot=False):
        logger.info(f"Processing query: {query}")
        try:
            if is_few_shot:
                few_shot_prompt = self.create_few_shot_prompt(query)
                enhanced_query = f"{few_shot_prompt}\n\nQuery: {query}"
                logger.debug(f"Enhanced query for few-shot learning: {enhanced_query[:100]}...")
                final_answer = yield self.tot_search.search(enhanced_query)
            else:
                final_answer = yield self.tot_search.search(query)
            
            logger.info(f"Final answer for '{query}':")
            logger.info(final_answer)
            
            yield self.agent.add_document_to_kb(title=f"ToT Search Result: {query}", content=final_answer)
            
            yield self.agent.replay_worker(batch_size=32)
            yield self.agent.replay_manager(batch_size=32)
            
            return final_answer
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}", exc_info=True)
            return f"An error occurred: {str(e)}"
        
    def create_few_shot_prompt(self, query):
        prompt = "Here are some examples of how to approach queries using a Tree of Thoughts:\n\n"
        for example in self.few_shot_examples:
            prompt += f"Query: {example['query']}\n"
            prompt += "Thoughts:\n"
            for thought in example['thoughts']:
                prompt += f"- {thought}\n"
            prompt += f"Answer: {example['answer']}\n\n"
        prompt += f"Now, let's approach the following query in a similar manner:\n\nQuery: {query}\n"
        return prompt

    def save_models(self):
        self.agent.save_worker_model("worker_model_final.pth")
        self.agent.save_manager_model("manager_model_final.pth")
        logger.info("Agent models saved.")


def get_user_input():
    return input("Enter your query (or 'quit' to exit): ")


@defer.inlineCallbacks
def run_test_session():
    test_agent = TestAgent()
    
    logger.info("Starting few-shot learning phase...")
    for example in test_agent.few_shot_examples:
        logger.info(f"Processing few-shot example: {example['query']}")
        try:
            yield test_agent.process_query(example['query'], is_few_shot=True)
        except Exception as e:
            logger.error(f"Error in few-shot learning: {str(e)}", exc_info=True)
    
    logger.info("Few-shot learning phase completed. Starting interactive session.")
    
    while True:
        query = yield threads.deferToThread(get_user_input)
        
        if query.lower() == 'quit':
            break
        
        try:
            answer = yield test_agent.process_query(query)
            print("\nAgent's response:")
            print(answer)
            print("\n" + "-"*50 + "\n")
        except Exception as e:
            logger.error(f"Error in interactive session: {str(e)}", exc_info=True)
    
    test_agent.save_models()
    reactor.stop()


if __name__ == "__main__":
    reactor.callWhenRunning(run_test_session)
    reactor.run()
