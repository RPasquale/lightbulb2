# ToTSearch.py
import random
from typing import List, Dict, Any, Generator
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from twisted.internet import defer
from agent import AutonomousWebAgent
from mcts import MCTS, MCTSNode
import logging
from twisted.internet.defer import Deferred


logger = logging.getLogger(__name__)

class ToTNode:
    def __init__(self, thought, parent=None):
        self.thought = thought
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.search_results = []
        self.mcts_node = None

    def add_child(self, child_thought):
        child = ToTNode(child_thought, self)
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.value += reward

class ToTSearch:
    def __init__(self, agent: AutonomousWebAgent, model='all-MiniLM-L6-v2', max_depth=3, num_thoughts=3, num_simulations=100):
        self.agent = agent
        self.model = SentenceTransformer(model)
        self.max_depth = max_depth
        self.num_thoughts = num_thoughts
        self.num_simulations = num_simulations
        self.mcts = MCTS(initial_state="", num_simulations=num_simulations)

    def generate_thoughts(self, query: str) -> List[str]:
        prompt = f"""Given the query "{query}", generate {self.num_thoughts} distinct thoughts or approaches to address it.
        Each thought should be a complete sentence and offer a unique perspective or solution path."""
        
        thoughts = self.agent.generate_text(prompt).split('\n')
        return [thought.strip() for thought in thoughts if thought.strip()]

    def expand_thought(self, thought: str) -> List[str]:
        prompt = f"""Expand on the following thought: "{thought}"
        Generate {self.num_thoughts} more specific sub-thoughts or considerations.
        Each sub-thought should be a complete sentence and offer additional detail or a new angle."""
        
        expansions = self.agent.generate_text(prompt).split('\n')
        return [exp.strip() for exp in expansions if exp.strip()]

    def evaluate_thought(self, thought: str, query: str) -> float:
        thought_embedding = self.model.encode(thought)
        query_embedding = self.model.encode(query)
        return util.pytorch_cos_sim(thought_embedding, query_embedding).item()

    @defer.inlineCallbacks
    def search_and_augment(self, thought: str) -> Generator[Deferred, Any, List[Dict[str, Any]]]:
        search_results = yield self.agent.retrieve_from_web(thought)
        for result in search_results:
            result['originating_thought'] = thought
        defer.returnValue(search_results)

    def select(self, node: ToTNode) -> ToTNode:
        while node.children:
            # Choose a node with zero visits or select based on the value/visits ratio
            if any(child.visits == 0 for child in node.children):
                zero_visit_nodes = [child for child in node.children if child.visits == 0]
                selected_node = random.choice(zero_visit_nodes)
                logger.debug(f"Selected node with 0 visits: {selected_node.thought}")
                return selected_node
            else:
                selected_node = max(node.children, key=lambda child: (child.value / child.visits) if child.visits > 0 else float('-inf'))
                logger.debug(f"Selected node based on value/visits ratio: {selected_node.thought}, value: {selected_node.value}, visits: {selected_node.visits}")
                return selected_node
        return node


    def expand(self, node: ToTNode, query: str) -> ToTNode:
        if not node.children and len(node.thought.split()) > 2:
            expansions = self.expand_thought(node.thought)
            for expansion in expansions:
                node.add_child(expansion)
        return random.choice(node.children) if node.children else node

    @defer.inlineCallbacks
    def simulate(self, node: ToTNode, query: str):
        current_node = node
        depth = 0
        while depth < self.max_depth:
            if not current_node.children:
                break
            current_node = random.choice(current_node.children)
            depth += 1
        
        logger.debug(f"Simulating for thought: {current_node.thought}")
        
        search_results = yield self.search_and_augment(current_node.thought)
        current_node.search_results = search_results
        
        logger.debug(f"Search results count: {len(search_results)}")
        
        ranked_results = self.agent.calculate_reward(current_node.thought, query)
        logger.debug(f"Ranked results: {ranked_results}")
        
        mcts_node = MCTSNode(current_node.thought)
        current_node.mcts_node = mcts_node
        mcts_total_reward = 0
        
        for _ in range(self.num_simulations):
            mcts_reward = yield self.mcts.simulate(mcts_node)
            mcts_total_reward += mcts_reward
            self.mcts.backpropagate(mcts_node, mcts_reward)
        
        logger.debug(f"MCTS node visits: {mcts_node.visits}, total reward: {mcts_total_reward}")
        
        if mcts_node.visits == 0 or ranked_results == 0:
            logger.warning(f"Avoiding division by zero. MCTS visits: {mcts_node.visits}, Ranked results: {ranked_results}")
            combined_reward = 0
        else:
            combined_reward = (ranked_results + mcts_value) / 2

        if mcts_node.visits > 0:
            mcts_value = mcts_total_reward / mcts_node.visits
            logger.debug(f"MCTS value: {mcts_value}")
        else:
            mcts_value = 0
            logger.warning(f"MCTS node has 0 visits, assigning value 0")
        
        combined_reward = (ranked_results + mcts_value) / 2
        logger.debug(f"Combined reward: {combined_reward}")
        
        defer.returnValue(combined_reward)

    def backpropagate(self, node: ToTNode, reward: float):
        while node:
            node.update(reward)
            node = node.parent

    @defer.inlineCallbacks
    def tot_search(self, query: str) -> Generator[Deferred, Any, ToTNode]:
        root = ToTNode(query)
        for _ in range(self.num_simulations):
            node = self.select(root)
            node = self.expand(node, query)
            reward = yield self.simulate(node, query)
            self.backpropagate(node, reward)
            
            # Update agent's experience replay
            state = self.agent.extract_features(node.thought, query)
            next_state = self.agent.extract_features(node.children[0].thought if node.children else node.thought, query)
            self.agent.remember_worker(state, 0, reward, next_state, False)
            
            # Perform agent's replay to update RL models
            self.agent.replay_worker()
            self.agent.replay_manager()
        
        defer.returnValue(root)

    def get_best_path(self, root: ToTNode) -> List[str]:
        path = [root.thought]
        current = root
        while current.children:
            current = max(current.children, key=lambda child: child.value / child.visits if child.visits > 0 else float('-inf'))
            path.append(current.thought)
        return path

    @defer.inlineCallbacks
    def synthesize_results(self, root: ToTNode, query: str) -> Generator[Deferred, Any, str]:
        best_path = self.get_best_path(root)
        all_results = []
        
        def collect_results(node):
            all_results.extend(node.search_results)
            for child in node.children:
                collect_results(child)
        
        collect_results(root)
        
        # Sort results by relevance
        all_results.sort(key=lambda x: self.evaluate_thought(x['content'], query), reverse=True)
        
        # Generate a summary of the top results
        top_results = all_results[:5]  # Adjust the number as needed
        summary_prompt = f"Synthesize the following information into a coherent answer for the query '{query}':\n\n"
        summary_prompt += f"Thought path: {' -> '.join(best_path)}\n\n"
        for result in top_results:
            summary_prompt += f"- {result['content'][:200]}...\n"
        
        # Use the agent's RAG capabilities for final answer generation
        final_answer = yield self.agent.generate_rag_response(query, top_results)
        
        # Save the generated answer and thought path to the agent's knowledge base
        self.agent.add_document_to_kb(
            title=f"ToT Search Result: {query}",
            content=final_answer,
            metadata={"thought_path": best_path}
        )
        
        defer.returnValue(final_answer)

    @defer.inlineCallbacks
    def search(self, query: str) -> Generator[Deferred, Any, str]:
        logger.info(f"Starting ToT search for query: {query}")
        root = yield self.tot_search(query)
        final_answer = yield self.synthesize_results(root, query)
        logger.info(f"ToT search completed for query: {query}")
        defer.returnValue(final_answer)

# Usage example:
# tot_search = ToTSearch(agent)
# final_answer = yield tot_search.search("What are the latest advancements in renewable energy?")