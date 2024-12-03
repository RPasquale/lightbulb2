
# agent.py
# agent.py
import numpy as np
from mcts import MCTS
from ranking import train_ranking_model
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, OrderedDict
import random
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import hashlib
from twisted.internet import defer
import logging
import json
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ==========================
# Prioritized Experience Replay
# ==========================

class SumTree:
    """
    SumTree data structure where the parent’s value is the sum of its children.
    Leaf nodes contain the priorities of experiences.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[data_idx])

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # [0,1] convert the importance of TD error to priority
        self.epsilon = 1e-6  # small amount to avoid zero priority

    def add(self, error, sample):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, sample)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        total = self.tree.total()
        probs = priorities / total
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights /= weights.max()
        return batch, idxs, weights

    def update(self, idx, error):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.update(idx, p)

# ==========================
# Hierarchical Reinforcement Learning (HRL)
# ==========================

class ManagerModel(nn.Module):
    """
    High-level policy model (Manager) that decides which option to execute.
    """
    def __init__(self, input_size, hidden_size, num_options):
        super(ManagerModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_options)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a time dimension
        out, hidden = self.lstm(x, hidden)
        last_output = out[:, -1, :]
        last_output = self.layer_norm(last_output)
        option_scores = self.fc(last_output)
        return option_scores, hidden

class WorkerModel(nn.Module):
    """
    Low-level policy model (Worker) that executes actions based on the selected option.
    """
    def __init__(self, input_size, hidden_size, action_size):
        super(WorkerModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.action_size = action_size  # Store action_size for reference

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a time dimension
        out, hidden = self.lstm(x, hidden)
        last_output = out[:, -1, :]
        last_output = self.layer_norm(last_output)
        action_scores = self.fc(last_output)
        return action_scores, hidden

    def act(self, state, epsilon=0.1):
        """
        Selects an action using epsilon-greedy policy.
        """
        if random.random() < epsilon:
            action = random.randint(0, self.action_size - 1)
            return action
        state = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            action_scores, _ = self(state)
            action = torch.argmax(action_scores, dim=1).item()
        return action

# ==========================
# RAGSummarizer Class
# ==========================

class RAGSummarizer:
    def __init__(self, model_name='gpt2', embedding_model='all-MiniLM-L6-v2', 
                 max_length=150, cache_capacity=100, persistent_cache_path='rag_cache.json'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        # Explicitly set the device for SentenceTransformer
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        self.max_length = max_length
        self.cache = LRUCache(cache_capacity)
        self.persistent_cache_path = persistent_cache_path
        self.load_persistent_cache()

    def load_persistent_cache(self):
        if os.path.exists(self.persistent_cache_path):
            with open(self.persistent_cache_path, 'r', encoding='utf-8') as f:
                try:
                    persistent_data = json.load(f)
                    for key, value in persistent_data.items():
                        self.cache.put(key, value)
                    logger.info(f"Loaded persistent cache with {len(persistent_data)} entries.")
                except json.JSONDecodeError:
                    logger.warning("Persistent cache file is corrupted. Initializing empty cache.")
        else:
            logger.info("No persistent cache found. Starting with empty cache.")

    def save_persistent_cache(self):
        with open(self.persistent_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache.cache, f, indent=2)
        logger.info(f"Saved persistent cache with {len(self.cache.cache)} entries.")

    def save_rag_data(self, query, chunks, embeddings):
        data = {
            "query": query,
            "chunks": chunks,
            "embeddings": embeddings.tolist()
        }

        os.makedirs("rag_data", exist_ok=True)

        filename = f"rag_data/{hash(query)}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved RAG data to {filename}")

    def split_into_chunks(self, text, chunk_size=200):
        words = text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def retrieve_relevant_chunks(self, query, chunks, embeddings, top_k=3):
        if embeddings.size(0) == 0:
            logger.warning("Embeddings are empty. Cannot retrieve relevant chunks.")
            return []
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        cosine_scores = cosine_similarity(query_embedding.cpu().numpy(), embeddings.cpu().numpy())[0]
        top_indices = cosine_scores.argsort()[-top_k:][::-1]
        # Ensure indices are within bounds
        top_indices = [idx for idx in top_indices if idx < len(chunks)]
        return [chunks[i] for i in top_indices]
    
    def get_embeddings(self, chunks):
        # Implement batch processing
        batch_size = 32
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings)
        if embeddings:
            return torch.cat(embeddings, dim=0)
        else:
            return torch.tensor([])

    def generate_summary(self, query, relevant_chunks):
        cache_key = hashlib.md5((query + ''.join(relevant_chunks)).encode()).hexdigest()
        cached_summary = self.cache.get(cache_key)
        if cached_summary:
            return cached_summary

        context = " ".join(relevant_chunks)
        prompt = f"Summarize the following content in relation to '{query}': {context}\n\nSummary:"

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        try:
            output = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + self.max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                early_stopping=True
            )
        except Exception as e:
            logger.error(f"Error during summary generation: {str(e)}")
            return "Summary generation failed."

        self.save_rag_data(query, relevant_chunks, self.get_embeddings(relevant_chunks))

        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
        summary = summary.split("Summary:")[-1].strip()

        self.cache.put(cache_key, summary)
        self.save_persistent_cache()

        return summary

# ==========================
# WorldModel Class
# ==========================

class WorldModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3):
        super(WorldModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a time dimension
        out, hidden = self.lstm(x, hidden)
        last_output = out[:, -1, :]
        last_output = self.layer_norm(last_output)
        action_scores = self.fc(last_output)
        state_value = self.value_head(last_output)
        return action_scores, state_value, hidden

# ==========================
# Manager and Worker Classes for HRL
# ==========================

class Manager:
    def __init__(self, state_size, num_options, hidden_size=128, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_capacity=1000, device=torch.device("cpu")):
        self.state_size = state_size
        self.num_options = num_options
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device

        self.model = ManagerModel(state_size, hidden_size, num_options).to(self.device)
        self.target_model = ManagerModel(state_size, hidden_size, num_options).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5, verbose=True)

        self.memory = PrioritizedReplayMemory(capacity=memory_capacity, alpha=0.6)

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, option, reward, next_state, done, td_error):
        sample = (state, option, reward, next_state, done)
        self.memory.add(td_error, sample)

    def act(self, state):
        if random.random() < self.epsilon:
            option = random.randint(0, self.num_options - 1)
            return option
        state = torch.FloatTensor(state).unsqueeze(0).to(self.model.lstm.weight.device)
        with torch.no_grad():
            option_scores, _ = self.model(state)
            option = torch.argmax(option_scores).item()
        return option

    def replay(self, batch_size, beta=0.4):
        if self.memory.tree.n_entries < batch_size:
            return
        batch, idxs, weights = self.memory.sample(batch_size, beta)
        states, options, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.model.lstm.weight.device)
        next_states = torch.FloatTensor(next_states).to(self.model.lstm.weight.device)
        options = torch.LongTensor(options).unsqueeze(1).to(self.model.lstm.weight.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.model.lstm.weight.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.model.lstm.weight.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.model.lstm.weight.device)

        # Current Q values
        current_q_values, _ = self.model(states)
        current_q_values = current_q_values.gather(1, options)

        # Target Q values
        with torch.no_grad():
            next_q_values, _ = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Compute TD errors
        td_errors = target_q_values - current_q_values

        # Compute loss with importance-sampling weights
        loss = (td_errors.pow(2) * weights).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step(loss.item())

        # Update priorities
        td_errors_np = td_errors.detach().cpu().numpy().squeeze()
        for idx, td_error in zip(idxs, td_errors_np):
            self.memory.update(idx, np.abs(td_error))

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==========================
# AutonomousWebAgent Class
# ==========================

def truncate_text(text, max_length=1024):
    tokens = text.split()
    if len(tokens) > max_length:
        return ' '.join(tokens[:max_length])
    return text

class AutonomousWebAgent:
    def __init__(self, state_size, action_size, num_options, hidden_size=64, learning_rate=0.001,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 knowledge_base_path='knowledge_base.json'):
        self.state_size = state_size
        self.action_size = action_size
        self.num_options = num_options  # Number of high-level options for HRL
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize RAGSummarizer first to get the device
        self.summarizer = RAGSummarizer()
        self.device = self.summarizer.device

        # Initialize SentenceTransformer with the correct device
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        # Low-level (Worker) Model
        self.worker_model = WorldModel(state_size, hidden_size, action_size).to(self.device)
        self.worker_target_model = WorldModel(state_size, hidden_size, action_size).to(self.device)
        self.worker_optimizer = optim.AdamW(self.worker_model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.worker_loss_fn = nn.MSELoss()
        self.worker_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.worker_optimizer, 'min', patience=5, factor=0.5, verbose=True)
        self.worker_memory = PrioritizedReplayMemory(capacity=2000, alpha=0.6)
        self.update_worker_target_model()

        # High-level (Manager) Model
        self.manager = Manager(state_size, num_options, hidden_size=128, learning_rate=learning_rate, 
                               gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, 
                               epsilon_min=epsilon_min, memory_capacity=1000, device=self.device)

        self.mcts = MCTS(initial_state="")
        logger.info(f"Initialized AutonomousWebAgent with state_size={state_size}, action_size={action_size}, num_options={num_options}")

        self.site_performance = {}  # {(site, query): performance_score}

        # List of all search sites (base URLs without the query)
        self.all_search_sites = [
            "https://en.wikibooks.org/w/index.php?search=",
            "https://en.wikiversity.org/w/index.php?search=",
            "https://commons.wikimedia.org/w/index.php?search=",
            "https://stackexchange.com/search?q=",
            "https://arxiv.org/search/?query=",
            "https://www.ncbi.nlm.nih.gov/pmc/?term=",
            "https://www.gutenberg.org/ebooks/search/?query=",
            "https://openlibrary.org/search?q=",
            "https://doaj.org/search/articles?ref=homepage&q=",
            "https://www.ted.com/search?q=",
            "https://en.citizendium.org/wiki?search=",
            "https://www.jstor.org/action/doBasicSearch?Query=",
            "https://archive.org/search.php?query=",
            "https://search.scielo.org/?q=",
            "https://paperswithcode.com/search?q=",
            "https://www.reddit.com/search/?q=",
            "https://huggingface.co/models?search=",
            "https://huggingface.co/datasets?search=",
            "https://machinelearningmastery.com/?s=",
            "https://www.kaggle.com/search?q=",
            "https://towardsdatascience.com/search?q=",
            "https://github.com/search?q=",
            "https://stackoverflow.com/search?q=",
            "https://www.youtube.com/results?search_query=",
            "https://www.slideshare.net/search/slideshow?searchfrom=header&q="
        ]

        # Initialize Knowledge Base
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = []
        self.kb_embeddings = None
        self.load_knowledge_base()

        # Additional Features for State Representation
        self.additional_features = ['image_count', 'script_count', 'css_count']

    def save(self, filename):
        """Save the entire agent state."""
        state = {
            'worker_model': self.worker_model.state_dict(),
            'manager_model': self.manager.model.state_dict(),
            'worker_optimizer': self.worker_optimizer.state_dict(),
            'manager_optimizer': self.manager.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(state, filename)
        logger.info(f"Saved agent state to {filename}")

    def load(self, filename):
        """Load the entire agent state."""
        state = torch.load(filename, map_location=self.device)
        self.worker_model.load_state_dict(state['worker_model'])
        self.manager.model.load_state_dict(state['manager_model'])
        self.worker_optimizer.load_state_dict(state['worker_optimizer'])
        self.manager.optimizer.load_state_dict(state['manager_optimizer'])
        self.epsilon = state['epsilon']
        logger.info(f"Loaded agent state from {filename}")

    # ==========================
    # Text Generation
    # ==========================

    def generate_text(self, prompt):
        # Use the RAGSummarizer to generate text
        chunks = self.summarizer.split_into_chunks(prompt)
        embeddings = self.summarizer.get_embeddings(chunks)
        relevant_chunks = self.summarizer.retrieve_relevant_chunks(query=prompt, chunks=chunks, embeddings=embeddings)
        generated_text = self.summarizer.generate_summary(prompt, relevant_chunks)
        return generated_text

    # ==========================
    # Knowledge Base Management
    # ==========================

    def load_knowledge_base(self):
        if not os.path.exists(self.knowledge_base_path):
            logger.warning(f"Knowledge base file {self.knowledge_base_path} does not exist. Initializing empty KB.")
            self.knowledge_base = []
            self.kb_embeddings = torch.tensor([]).to(self.device)
            return
        
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        
        if self.knowledge_base:
            texts = [doc['content'] for doc in self.knowledge_base]
            self.kb_embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
            logger.info(f"Loaded {len(self.knowledge_base)} documents into the knowledge base.")
        else:
            self.kb_embeddings = torch.tensor([]).to(self.device)
            logger.info("Knowledge base is empty.")

    def save_knowledge_base(self):
        with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2)
        logger.info(f"Knowledge base saved with {len(self.knowledge_base)} documents.")

    def add_document_to_kb(self, title, content, metadata=None):
        document = {
            "title": title,
            "content": content,
            "metadata": metadata or {}
        }
        self.knowledge_base.append(document)
        # Update embeddings
        new_embedding = self.embedding_model.encode([content], convert_to_tensor=True).to(self.device)
        if self.kb_embeddings.numel() == 0:
            self.kb_embeddings = new_embedding
        else:
            self.kb_embeddings = torch.cat([self.kb_embeddings, new_embedding], dim=0)
        # Save to knowledge base
        self.save_knowledge_base()
        logger.info(f"Added new document to knowledge base: {title}")

    def retrieve_from_kb(self, query, top_k=5):
        if not self.knowledge_base:
            logger.warning("Knowledge base is empty. No documents to retrieve.")
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True).to(self.device)
        
        if self.kb_embeddings is None or self.kb_embeddings.numel() == 0:
            logger.warning("Knowledge base embeddings are empty. No documents to retrieve.")
            return []
        
        if query_embedding.size(1) != self.kb_embeddings.size(1):
            logger.error("Dimension mismatch between query embedding and KB embeddings.")
            return []
        
        cosine_scores = cosine_similarity(query_embedding.cpu().numpy(), self.kb_embeddings.cpu().numpy())[0]
        top_indices = cosine_scores.argsort()[-top_k:][::-1]
        
        # Ensure indices are within the knowledge_base length
        top_indices = [idx for idx in top_indices if idx < len(self.knowledge_base)]
        
        retrieved_docs = []
        for idx in top_indices:
            doc = self.knowledge_base[idx]
            doc['score'] = cosine_scores[idx]
            retrieved_docs.append(doc)
        
        logger.info(f"Retrieved top {len(retrieved_docs)} documents from Knowledge Base for the query.")
        return retrieved_docs

    # ==========================
    # RAG Integration
    # ==========================

    def retrieve_from_web(self, query, top_k=5):
        logger.info(f"Performing web search for query: {query}")
        mcts_iterations = self.calculate_mcts_iterations(np.zeros(self.state_size, dtype=np.float32))
        self.mcts = MCTS(initial_state=query, num_simulations=mcts_iterations)
        
        try:
            new_query = yield self.mcts.run()
            logger.debug(f"New query from MCTS: {new_query}")
            # Select search sites
            search_sites = self.select_search_sites(new_query)
            results = yield self.mcts.web_search(new_query, search_sites)
            logger.debug(f"Web search completed. Found {len(results)} results")
            return results[:top_k] if results else []
        except Exception as e:
            logger.error(f"Error during MCTS or web search: {str(e)}", exc_info=True)
            return []

    def combine_documents(self, kb_docs, web_docs):
        combined = kb_docs + web_docs
        logger.info(f"Combined {len(kb_docs)} KB documents and {len(web_docs)} Web documents.")
        return combined

    def save_llm_training_data(self, query, content, summary=None, link=None, title=None):
        data = {
            "query": query,
            "search_result": {
                "link": link,
                "title": title
            },
            "content": content,
            "description": summary
        }

        os.makedirs("llm_training_data", exist_ok=True)
        file_path = "llm_training_data/llm_training_data.jsonl"

        # Append the new data as a new line in the JSONL file
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(data, f)
            f.write('\n')

        logger.info(f"Appended LLM training data to {file_path}")

    # ==========================
    # Hierarchical RL Integration
    # ==========================

    def remember_manager(self, state, option, reward, next_state, done, td_error):
        self.manager.remember(state, option, reward, next_state, done, td_error)

    def remember_worker(self, state, action, reward, next_state, done):
        self.worker_memory.add(reward, (state, action, reward, next_state, done))

    # ==========================
    # Action Selection and Execution
    # ==========================

    def act_manager(self, state):
        option = self.manager.act(state)
        return option

    def act_worker(self, state):
        action = self.worker_model.act(state, epsilon=self.epsilon)
        return action

    # ==========================
    # Replay Methods
    # ==========================

    def replay_manager(self, batch_size=32, beta=0.4):
        self.manager.replay(batch_size, beta)

    def replay_worker(self, batch_size=32, beta=0.4):
        result = self.worker_memory.replay(batch_size, beta)
        if result is None:
            return
        batch, idxs, weights = result
        if len(self.worker_memory.tree.data) >= batch_size:
            batch, idxs, weights = self.worker_memory.sample(batch_size, beta)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(self.worker_model.lstm.weight.device)
            next_states = torch.FloatTensor(next_states).to(self.worker_model.lstm.weight.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.worker_model.lstm.weight.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.worker_model.lstm.weight.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.worker_model.lstm.weight.device)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.worker_model.lstm.weight.device)

            # Current Q values
            current_q_values, _ = self.worker_model(states)
            current_q_values = current_q_values.gather(1, actions)

            # Target Q values
            with torch.no_grad():
                next_q_values, _ = self.worker_target_model(next_states)
                max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
                target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

            # Compute TD errors
            td_errors = target_q_values - current_q_values

            # Compute loss with importance-sampling weights
            loss = (td_errors.pow(2) * weights).mean()

            # Optimize the model
            self.worker_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.worker_model.parameters(), max_norm=1.0)
            self.worker_optimizer.step()
            self.worker_scheduler.step(loss.item())

            # Update priorities
            td_errors_np = td_errors.detach().cpu().numpy().squeeze()
            for idx, td_error in zip(idxs, td_errors_np):
                self.worker_memory.update(idx, np.abs(td_error))

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                logger.debug(f"Updated epsilon to: {self.epsilon}")

    # ==========================
    # Load and Save Models
    # ==========================

    def load_worker_model(self, name):
        self.worker_model.load_state_dict(torch.load(name, map_location=self.device))
        logger.info(f"Loaded worker model weights from {name}")

    def save_worker_model(self, name):
        torch.save(self.worker_model.state_dict(), name)
        logger.info(f"Saved worker model weights to {name}")

    def load_manager_model(self, name):
        self.manager.model.load_state_dict(torch.load(name, map_location=self.device))
        self.manager.update_target_model()
        logger.info(f"Loaded manager model weights from {name}")

    def save_manager_model(self, name):
        torch.save(self.manager.model.state_dict(), name)
        logger.info(f"Saved manager model weights to {name}")

    # ==========================
    # Update Target Models
    # ==========================

    def update_worker_target_model(self):
        self.worker_target_model.load_state_dict(self.worker_model.state_dict())
        logger.info("Updated worker target model with current model weights")

    def update_manager_target_model(self):
        self.manager.update_target_model()
        logger.info("Updated manager target model with current model weights")

    # ==========================
    # Feature Extraction
    # ==========================

    def extract_features(self, content, query):
        content = truncate_text(content)
        query = truncate_text(query)
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        word_count = len(text.split())
        link_count = len(soup.find_all('a'))
        header_count = len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
        
        # Calculate semantic similarity
        text_embedding = self.embedding_model.encode([text], convert_to_tensor=True).to(self.device)
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True).to(self.device)
        semantic_similarity = cosine_similarity(text_embedding.cpu().numpy(), query_embedding.cpu().numpy())[0][0]
        
        # Additional Features
        image_count = len(soup.find_all('img'))
        script_count = len(soup.find_all('script'))
        css_count = len(soup.find_all('link', rel='stylesheet'))
        
        return np.array([word_count, link_count, header_count, semantic_similarity, image_count, script_count, css_count])

    # ==========================
    # Reward Calculation
    # ==========================

    def calculate_reward(self, content, query):
        try:
            ranked_results = train_ranking_model(query, [{'content': content}])
            logger.debug(f"Ranked results: {ranked_results}")
            if ranked_results and isinstance(ranked_results[0], dict) and 'predicted_score' in ranked_results[0]:
                reward = max(ranked_results[0]['predicted_score'], 0)
                logger.debug(f"Calculated reward: {reward}")
                return reward
            else:
                logger.warning(f"Invalid ranked results: {ranked_results}")
                return 0
        except Exception as e:
            logger.error(f"Error in calculate_reward: {str(e)}", exc_info=True)
            return 0

    # ==========================
    # Search Site Selection
    # ==========================

    def select_search_sites(self, query, num_sites=5):
        # Select top sites based on past performance for this query
        site_scores = {}
        for (site, q), score in self.site_performance.items():
            if q == query:
                site_scores[site] = site_scores.get(site, 0) + score
        if site_scores:
            sorted_sites = sorted(site_scores.items(), key=lambda x: x[1], reverse=True)
            top_sites = [site for site, score in sorted_sites[:num_sites]]
        else:
            # If no past data, select random sites
            top_sites = random.sample(self.all_search_sites, num_sites)
        # Construct full URLs with query
        search_sites = [site + query for site in top_sites]
        return search_sites

    # ==========================
    # Search Method with HRL
    # ==========================

    @defer.inlineCallbacks
    def search(self, query, max_steps=2):
        logger.info(f"Starting search for query: {query}")
        state = np.zeros(self.state_size, dtype=np.float32)
        total_reward = 0
        content = ""
        done = False
        results = None

        try:
            # High-Level: Manager selects an option
            option = self.act_manager(state)
            logger.debug(f"Manager selected option: {option}")

            # Execute the selected option
            if option == 0:  # Search Option
                logger.debug("Executing Search Option")
                results = yield self.retrieve_from_web(query)
                if results:
                    content = results[0]['content']
                    site = urlparse(results[0]['link']).netloc
                    self.save_llm_training_data(
                        query, 
                        content, 
                        summary=results[0].get('summary'),
                        link=results[0].get('link'),
                        title=results[0].get('title')
                    )
                    self.add_document_to_kb(title=results[0].get('title', 'No Title'), content=content, metadata=results[0].get('meta', {}))
                    next_state = self.extract_features(content, query)
                    reward = self.calculate_reward(content, query)
                    logger.debug(f"Extracted features: {next_state}, Reward: {reward}")
                    # Update site performance
                    key = (site, query)
                    self.site_performance[key] = self.site_performance.get(key, 0) + reward

                    # Remember Manager's experience
                    self.remember_manager(state, option, reward, next_state, done, td_error=reward)

                    # Remember Worker's experience
                    self.remember_worker(state, 0, reward, next_state, done)

                    state = next_state.astype(np.float32)
                    total_reward += reward

                else:
                    reward = -1
                    logger.warning(f"No results for query: {query}")
                    # Remember Manager's experience
                    self.remember_manager(state, option, reward, state, True, td_error=reward)

            elif option == 1:  # Summarize Option
                logger.debug("Executing Summarize Option")
                if content:
                    summary = self.summarizer.generate_summary(content, query)
                    self.save_llm_training_data(
                        query, 
                        content, 
                        summary=summary,
                        link=results[0].get('link') if results else None,
                        title=results[0].get('title') if results else None
                    )
                    reward = self.calculate_reward(summary, query)
                    next_state = self.extract_features(summary, query)
                    logger.info(f"Summary:\n{summary}")
                    logger.info(f"Summarized content. Reward: {reward}")

                    # Remember Manager's experience
                    self.remember_manager(state, option, reward, next_state, done, td_error=reward)

                    # Remember Worker's experience
                    self.remember_worker(state, 1, reward, next_state, done)

                    state = next_state.astype(np.float32)
                    total_reward += reward
                else:
                    reward = -1
                    logger.warning("No content to summarize")
                    # Remember Manager's experience
                    self.remember_manager(state, option, reward, state, True, td_error=reward)

            elif option == 2:  # RAG-based Generation Option
                logger.debug("Executing RAG-based Generation Option")
                kb_docs = self.retrieve_from_kb(query, top_k=5)
                web_docs = []  # Assuming web_docs are already retrieved
                combined_docs = self.combine_documents(kb_docs, web_docs)
                generated_output = self.generate_rag_response(query, combined_docs)
                logger.info(f"Generated Output:\n{generated_output}")
                self.save_llm_training_data(
                    query, 
                    generated_output, 
                    summary=None,
                    link=None,
                    title="RAG-generated response"
                )
                reward = self.calculate_reward(generated_output, query)
                next_state = self.extract_features(generated_output, query)

                # Remember Manager's experience
                self.remember_manager(state, option, reward, next_state, done, td_error=reward)

                # Remember Worker's experience
                self.remember_worker(state, 2, reward, next_state, done)

                state = next_state.astype(np.float32)
                total_reward += reward

            else:
                logger.warning(f"Unknown option selected by Manager: {option}")

            # Perform replay for both Manager and Worker
            self.replay_manager(batch_size=32, beta=0.4)
            self.replay_worker(batch_size=32, beta=0.4)

            # Update target models periodically
            self.update_worker_target_model()
            self.update_manager_target_model()

            logger.info(f"Search completed. Total reward: {total_reward}")
            defer.returnValue(total_reward)
        except Exception as e:
            logger.error(f"Error during search: {str(e)}", exc_info=True)
            defer.returnValue(-1)  # Return a negative reward on error

    # ==========================
    # Summarization Method
    # ==========================

    def summarize(self, content, query):
        chunks = self.summarizer.split_into_chunks(content)
        embeddings = self.summarizer.get_embeddings(chunks)
        relevant_chunks = self.summarizer.retrieve_relevant_chunks(query, chunks, embeddings)
        summary = self.summarizer.generate_summary(query, relevant_chunks)
        
        # Save RAG data
        self.summarizer.save_rag_data(query, chunks, embeddings)
        
        return summary

    # ==========================
    # MCTS Iterations Calculation
    # ==========================

    def calculate_mcts_iterations(self, state):
        # Calculate MCTS iterations based on state complexity
        base_iterations = 2
        complexity_factor = np.mean(state) / 100  # Normalize state values
        iterations = int(base_iterations * (1 + complexity_factor))
        max_iterations = 5  # Set a reasonable maximum
        return min(iterations, max_iterations)

    # ==========================
    # RAG-based Response Generation
    # ==========================

    def generate_rag_response(self, query, combined_docs):
        if not combined_docs:
            logger.warning("No documents available for RAG-based generation.")
            return "I'm sorry, I couldn't find any relevant information."

        # Prepare context for the generator
        context = "\n\n".join([f"Title: {doc.get('title', 'No Title')}\nContent: {doc.get('content', '')}" for doc in combined_docs])
        prompt = f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"
        
        # Check cache first
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        cached_response = self.summarizer.cache.get(cache_key)
        if cached_response:
            logger.debug("Using cached RAG response.")
            return cached_response

        # Generate response
        input_ids = self.summarizer.tokenizer.encode(prompt, return_tensors='pt').to(self.summarizer.device)
        try:
            output = self.summarizer.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + self.summarizer.max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                early_stopping=True
            )
        except Exception as e:
            logger.error(f"Error during RAG response generation: {str(e)}")
            return "RAG response generation failed."

        response = self.summarizer.tokenizer.decode(output[0], skip_special_tokens=True)
        answer = response.split("Answer:")[-1].strip()
        
        # Cache the response
        self.summarizer.cache.put(cache_key, answer)
        self.summarizer.save_persistent_cache()
        return answer

    # ==========================
    # Manager and Worker Interaction
    # ==========================

    def select_option(self, option):

        """
        Define the mapping of options to their corresponding actions.
        """
        # This can be expanded based on the number of options
        option_actions = {
            0: self.perform_search,
            1: self.perform_summarization,
            2: self.perform_rag_generation
        }
        action = option_actions.get(option, None)
        if action:
            return action
        else:
            logger.error(f"No action defined for option: {option}")
            return None

    def perform_search(self, query):
        """
        Perform the search action.
        """
        # Implementation is handled in the 'search' method
        pass

    def perform_summarization(self, content, query):
        """
        Perform the summarization action.
        """
        # Implementation is handled in the 'summarize' method
        pass

    def perform_rag_generation(self, query, combined_docs):
        """
        Perform the RAG-based generation action.
        """
        # Implementation is handled in the 'generate_rag_response' method
        pass

# ==========================
# LRUCache Class
# ==========================

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


