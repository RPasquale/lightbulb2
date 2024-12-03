'''import os
import random
import json
import pandas as pd
import pdfplumber
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
import torch.optim as optim
from tabular_ml import TabularMLProcessor
import logging
import time
from torch.multiprocessing import Pool, cpu_count
import torch.nn.functional as F
import math 
from concurrent.futures import ThreadPoolExecutor
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MuZeroModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(MuZeroModel, self).__init__()
        self.representation = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.to(device)

    def forward(self, state, action):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        if action.dim() == 2:
            action = action.unsqueeze(0)
        
        assert state.dim() == 3 and action.dim() == 3, f"State dim: {state.dim()}, Action dim: {action.dim()}"

        state_rep = self.representation(state.squeeze(1))
        action_rep = torch.cat([state_rep, action.squeeze(1)], dim=-1)
        next_state_rep = self.dynamics(action_rep)
        reward = self.reward_head(next_state_rep)
        value = self.value_head(state_rep)
        return next_state_rep, reward, value

class RLHFDataset(Dataset):
    def __init__(self, contexts, cache_dir):
        self.contexts = contexts
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return RLHFState(context=self.contexts[idx], cache_dir=self.cache_dir)

class RLHFState:
    def __init__(self, context, prompt=None, accepted=None, rejected=None, cache_dir=None):
        self.context = context
        self.prompt = prompt
        self.accepted = accepted
        self.rejected = rejected

        # Initialize these pipelines on CPU to avoid multiprocessing issues
        self.qa_pipeline = pipeline(
            "question-answering", 
            model=AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad", cache_dir=cache_dir),
            tokenizer=AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad", cache_dir=cache_dir),
            device=-1  # Run on CPU
        )
        self.qg_pipeline = pipeline(
            "text2text-generation",
            model=T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-hl", cache_dir=cache_dir),
            tokenizer=AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl", cache_dir=cache_dir),
            device=-1  # Run on CPU
        )
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=cache_dir, device="cpu")

        self.visits = 0
        self.total_reward = 0
        self.parent = None
        self.children = []

    def ucb_score(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.total_reward / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def generate_prompt(self):
        input_text = f"generate question: {self.context}"
        result = self.qg_pipeline(input_text, max_new_tokens=50)
        return result[0]['generated_text']

    def generate_accepted(self):
        if not self.prompt:
            raise ValueError("No prompt to answer")
        result = self.qa_pipeline(question=self.prompt, context=self.context)
        return result['answer']

    def generate_rejected(self):
        if not self.prompt:
            raise ValueError("No prompt to answer")
        incorrect_context = self.context[:len(self.context)//2] + self.context[len(self.context)//2:][::-1]
        result = self.qa_pipeline(question=self.prompt, context=incorrect_context)
        return result['answer']

    def is_terminal(self):
        return self.prompt is not None and self.accepted is not None and self.rejected is not None

    def get_reward(self):
        if self.prompt and self.accepted and self.rejected:
            return (len(self.prompt) + len(self.accepted) + len(self.rejected) + 
                    self.semantic_similarity(self.context, self.accepted) - 
                    self.semantic_similarity(self.context, self.rejected))
        return 0.0
    
    def semantic_similarity(self, text1, text2):
        embedding1 = self.embedding_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.embedding_model.encode(text2, convert_to_tensor=True)
        
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return similarity

def custom_collate(batch):
    # This function will create a batch from a list of RLHFState objects
    contexts = [item.context for item in batch]
    return contexts 

class RLHFDataGenerator:
    def __init__(self, pdf_folder, csv_folder, output_file, cache_dir, num_samples=10, batch_size=2):
        self.pdf_folder = pdf_folder
        self.csv_folder = csv_folder
        self.output_file = output_file
        self.cache_dir = cache_dir
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.tabular_processor = TabularMLProcessor()
        self.model = MuZeroModel(state_dim=2, action_dim=1, hidden_dim=64).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.examples = self.load_examples()
        self.logger = logging.getLogger(__name__)
        self.scaler = GradScaler()

    def load_examples(self):
        examples = [
            {"prompt": "Summarize the main argument presented in paragraph 3 of the PDF.", "accepted": "A concise, accurate summary of the key points from the specified paragraph, using objective language.", "rejected": "A summary that includes personal opinions, information from other paragraphs, or misrepresents the argument."},
            {"prompt": "Calculate the average value in column B of the CSV file.", "accepted": "The correct mathematical average of the values in column B, presented with appropriate precision.", "rejected": "An incorrect calculation, or an average from the wrong column or dataset."},
            {"prompt": "Identify the trend in data points 15-30 in the CSV file.", "accepted": "An accurate description of the trend (e.g., 'increasing linearly', 'fluctuating with an overall downward trend') based on the specified data points.", "rejected": "A trend description that contradicts the data or describes points outside the specified range."},
            {"prompt": "Compare the outcomes described in scenarios 2 and 4 of the PDF.", "accepted": "A balanced comparison highlighting key similarities and differences between the two scenarios, referencing specific details from each.", "rejected": "A comparison that misrepresents one or both scenarios, or that introduces information not present in the original text."},
            {"prompt": "What correlation exists between variables X and Y in the CSV?", "accepted": "An accurate statement about the correlation (positive, negative, or none) supported by the data, potentially including the correlation coefficient if calculable.", "rejected": "A claim about correlation that contradicts the data or assumes a relationship where none is evident."},
            {"prompt": "Explain how the concept introduced in section 2 of the PDF relates to the data in columns C and D of the CSV.", "accepted": "A logical explanation of the relationship, citing specific examples from both the PDF concept and the CSV data to support the connection.", "rejected": "An explanation that forces a connection where none exists, or that misinterprets either the concept or the data."},
            {"prompt": "What is the main limitation of the methodology described in the PDF's introduction?", "accepted": "An accurate identification of a key limitation, based on information provided in the introduction, with a brief explanation of its potential impact.", "rejected": "Claiming there are no limitations, or identifying a limitation not mentioned or implied in the text."},
            {"prompt": "Generate a hypothesis based on the findings in section 4 of the PDF and rows 50-75 of the CSV.", "accepted": "A plausible hypothesis that logically combines information from both sources, presented as a testable statement.", "rejected": "A hypothesis that contradicts the given information or makes unfounded leaps beyond the data provided."},
            {"prompt": "Identify any outliers in the dataset presented in the CSV file.", "accepted": "Accurate identification of data points that significantly deviate from the overall pattern, with brief justification for why they're considered outliers.", "rejected": "Labeling normal data variations as outliers, or failing to recognize clear anomalies in the dataset."},
            {"prompt": "Summarize the key differences between the approaches outlined in sections 1, 3, and 5 of the PDF.", "accepted": "A clear, concise comparison of the main points of each approach, highlighting key differentiators without bias.", "rejected": "A summary that blends the approaches, misrepresents their key features, or shows preference for one without justification from the text."},
            {"prompt": "Based on the CSV data, predict the likely value for variable Z if X increases by 10%.", "accepted": "A reasonable prediction based on the relationship between variables X and Z in the existing data, showing the calculation or logic used.", "rejected": "A prediction that ignores the established relationship in the data or uses incorrect mathematical reasoning."},
            {"prompt": "How does the case study in section 6 of the PDF exemplify the trend seen in column E of the CSV?", "accepted": "An explanation that accurately links the qualitative information from the case study to the quantitative trend in the CSV, providing specific examples.", "rejected": "Forcing a connection that isn't supported by the data, or misinterpreting either the case study or the trend."},
            {"prompt": "What ethical considerations are raised by the methodology described in the PDF, in light of the outcomes shown in the CSV?", "accepted": "A thoughtful discussion of potential ethical issues, referencing specific aspects of the methodology and data outcomes that could be problematic.", "rejected": "Claiming there are no ethical considerations, or raising ethical issues not related to the given information."},
            {"prompt": "Propose an alternative interpretation of the results presented in section 7 of the PDF, using the data from columns F and G of the CSV to support your argument.", "accepted": "A logical alternative interpretation that doesn't contradict the given data, supported by specific references to both the PDF content and CSV data.", "rejected": "An interpretation that ignores key information from either source, or that isn't substantively different from the original interpretation."},
            {"prompt": "Identify any inconsistencies between the conclusions drawn in the PDF and the data presented in the CSV.", "accepted": "Accurate identification of any areas where the PDF's conclusions aren't fully supported by the CSV data, with specific examples.", "rejected": "Claiming inconsistencies where none exist, or failing to recognize clear discrepancies between the two sources."},
            {"prompt": "Suggest a follow-up study based on the findings presented in both the PDF and CSV.", "accepted": "A well-reasoned proposal for further research that addresses gaps or questions raised by the current data, with clear objectives and methodology.", "rejected": "A suggestion that ignores the current findings, or proposes a study that would merely replicate the existing research without adding new insights."},
            {"prompt": "Evaluate the strength of the evidence presented in section 5 of the PDF using the statistical data from the CSV.", "accepted": "A balanced assessment that considers both qualitative arguments from the PDF and quantitative data from the CSV, noting areas of strong support and potential weaknesses.", "rejected": "An evaluation that misrepresents the strength of evidence, either by overstating weak connections or understating strong correlations."},
            {"prompt": "How might the findings from the CSV data affect the practical applications discussed in the conclusion of the PDF?", "accepted": "A thoughtful analysis of how the quantitative results could impact the proposed applications, considering both supporting and potentially challenging data points.", "rejected": "Ignoring relevant data from the CSV, or making sweeping claims about applications without considering the limitations of the findings."},
            {"prompt": "Create a brief executive summary that integrates key points from both the PDF report and the CSV data analysis.", "accepted": "A concise, well-structured summary that accurately represents the main findings from both sources, highlighting their interrelations and overall significance.", "rejected": "A summary that favors one source over the other, misses critical points, or fails to show how the PDF and CSV data complement each other."},
            {"prompt": "Based on the methodologies described in the PDF and the results in the CSV, what potential confounding variables should be considered in future research?", "accepted": "Identification of relevant confounding variables that could affect the results, explaining their potential impact and suggesting ways to control for them in future studies.", "rejected": "Listing irrelevant variables, or failing to explain how identified confounders relate to the specific methodologies and results presented."}
        ]
        return examples

    def generate_sample(self, _):
        context = self.get_context_from_files()
        self.logger.info(f"Context generated: {len(context)} characters")
        state = RLHFState(context=context, cache_dir=self.cache_dir)
        return self.mcts(state)

    def get_context_from_files(self, _):
        self.logger.info("Generating context from files")
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        csv_files = [f for f in os.listdir(self.csv_folder) if f.endswith('.csv')]

        pdf_texts = [self.extract_text_from_pdf(os.path.join(self.pdf_folder, f)) for f in pdf_files]
        csv_summaries = [self.process_csv_file(os.path.join(self.csv_folder, f)) for f in csv_files]

        combined_context = " ".join(pdf_texts + csv_summaries)
        self.logger.info(f"Combined context length: {len(combined_context)} characters")
        return combined_context[:1000]

    def process_batch(self, batch):
        results = []
        for context in batch:
            state = RLHFState(context=context, cache_dir=self.cache_dir)
            result = self.mcts(state)
            results.append(result)
        return results

    def mcts(self, root, iterations=10):
        self.logger.info(f"Starting parallel MCTS with {iterations} iterations")
        
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [executor.submit(self.mcts_iteration, root) for _ in range(iterations)]
            results = [future.result() for future in futures]
        
        best_node = max(results, key=lambda x: x['reward'])
        self.logger.info("MCTS completed")
        return {
            "prompt": best_node['node'].prompt,
            "accepted": best_node['node'].accepted,
            "rejected": best_node['node'].rejected
        }

    def mcts_iteration(self, root):
        node = self.select(root)
        if not node.is_terminal():
            node = self.expand(node)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        else:
            reward = node.get_reward()
        return {'node': node, 'reward': reward}

    def select(self, node):
        while not node.is_terminal():
            if not node.children:
                return node
            node = max(node.children, key=lambda n: n.ucb_score())
        return node

    def expand(self, node):
        if not node.prompt:
            node.prompt = node.generate_prompt()
        elif not node.accepted:
            node.accepted = node.generate_accepted()
        elif not node.rejected:
            node.rejected = node.generate_rejected()
        return node

    def simulate(self, node):
        example = random.choice(self.examples)
        simulated_state = RLHFState(context=node.context, prompt=example['prompt'], 
                                    accepted=example['accepted'], rejected=example['rejected'], 
                                    cache_dir=self.cache_dir)
        return simulated_state.get_reward()

    def backpropagate(self, node, reward):
        states = []
        actions = []
        rewards = []
        
        while node:
            state = [1.0 if node.prompt else 0.0, 1.0 if node.accepted else 0.0]
            action = [0.0 if node.rejected else 1.0]
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            node.visits += 1
            node.total_reward += reward
            node = node.parent

        self.batch_update(states, actions, rewards)

    def batch_update(self, states, actions, rewards):
        if len(states) < self.batch_size:
            return

        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1)

        with autocast():
            _, predicted_rewards, _ = self.model(states_tensor, actions_tensor)
            loss = F.mse_loss(predicted_rewards, rewards_tensor)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def generate_rlhf_data(self):
        self.logger.info("Starting RLHF data generation")
        start_time = time.time()

        with Pool(processes=cpu_count()) as pool:
            contexts = pool.map(self.get_context_from_files, range(self.num_samples))

        dataset = RLHFDataset(contexts, self.cache_dir)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, 
                                pin_memory=True, collate_fn=custom_collate)

        rlhf_data = []
        for batch in dataloader:
            batch_results = self.process_batch(batch)
            rlhf_data.extend(batch_results)

        with open(self.output_file, 'w') as f:
            for item in rlhf_data:
                json.dump(item, f)
                f.write('\n')
        
        end_time = time.time()
        self.logger.info(f"RLHF data saved to {self.output_file}")
        self.logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")

    def process_csv_file(self, csv_path):
        df = pd.read_csv(csv_path)
        return self.tabular_processor.summarize_dataframe(df)

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    
if __name__ == "__main__":
    pdf_folder = "C:/Users/Admin/Desktop/ai-ml-web-app/backend/uploads/pdf"
    csv_folder = "C:/Users/Admin/Desktop/ai-ml-web-app/backend/uploads/csv"
    output_file = "rlhf_data.jsonl"
    cache_dir = "C:/Users/Admin/.cache/transformers"
    num_samples = 10

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    generator = RLHFDataGenerator(pdf_folder, csv_folder, output_file, cache_dir, num_samples)
    generator.generate_rlhf_data()'''
    




import os
import random
import json
import pandas as pd
import pdfplumber
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
import torch.optim as optim
from tabular_ml import TabularMLProcessor
import logging
import time
from torch.multiprocessing import Pool, cpu_count
import torch.nn.functional as F
import math 
from concurrent.futures import ThreadPoolExecutor
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MuZeroModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(MuZeroModel, self).__init__()
        self.representation = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.to(device)

    def forward(self, state, action):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        if action.dim() == 2:
            action = action.unsqueeze(0)
        
        assert state.dim() == 3 and action.dim() == 3, f"State dim: {state.dim()}, Action dim: {action.dim()}"

        state_rep = self.representation(state.squeeze(1))
        action_rep = torch.cat([state_rep, action.squeeze(1)], dim=-1)
        next_state_rep = self.dynamics(action_rep)
        reward = self.reward_head(next_state_rep)
        value = self.value_head(state_rep)
        return next_state_rep, reward, value

class RLHFDataset(Dataset):
    def __init__(self, contexts, examples, cache_dir):
        self.contexts = contexts
        self.examples = examples
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        example = self.examples[idx % len(self.examples)]
        return RLHFState(context=self.contexts[idx], example=example, cache_dir=self.cache_dir)


class RLHFState:
    def __init__(self, context, example, cache_dir=None):
        self.context = context
        self.prompt = example['prompt']
        self.accepted = None
        self.rejected = None
        self.cache_dir = cache_dir

        # Initialize these pipelines on CPU to avoid multiprocessing issues
        self.qa_pipeline = pipeline(
            "question-answering", 
            model=AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad", cache_dir=cache_dir),
            tokenizer=AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad", cache_dir=cache_dir),
            device=-1  # Run on CPU
        )
        self.qg_pipeline = pipeline(
            "text2text-generation",
            model=T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-hl", cache_dir=cache_dir),
            tokenizer=AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl", cache_dir=cache_dir),
            device=-1  # Run on CPU
        )
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=cache_dir, device="cpu")

        self.visits = 0
        self.total_reward = 0
        self.parent = None
        self.children = []

    def ucb_score(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.total_reward / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def generate_accepted(self):
        if not self.prompt:
            raise ValueError("No prompt to answer")
        result = self.qa_pipeline(question=self.prompt, context=self.context)
        self.accepted = result['answer']

    def generate_rejected(self):
        if not self.prompt:
            raise ValueError("No prompt to answer")
        incorrect_context = self.context[:len(self.context)//2] + self.context[len(self.context)//2:][::-1]
        result = self.qa_pipeline(question=self.prompt, context=incorrect_context)
        self.rejected = result['answer']

    def is_terminal(self):
        return self.accepted is not None and self.rejected is not None

    def get_reward(self):
        if self.accepted and self.rejected:
            return (len(self.prompt) + len(self.accepted) + len(self.rejected) + 
                    self.semantic_similarity(self.context, self.accepted) - 
                    self.semantic_similarity(self.context, self.rejected))
        return 0.0
    
    def semantic_similarity(self, text1, text2):
        embedding1 = self.embedding_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.embedding_model.encode(text2, convert_to_tensor=True)
        
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return similarity

def custom_collate(batch):
    contexts = [item.context for item in batch]
    return contexts 

class RLHFDataGenerator:
    def __init__(self, pdf_folder, csv_folder, output_file, cache_dir, num_samples=10, batch_size=2):
        self.pdf_folder = pdf_folder
        self.csv_folder = csv_folder
        self.output_file = output_file
        self.cache_dir = cache_dir
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.tabular_processor = TabularMLProcessor()
        self.model = MuZeroModel(state_dim=2, action_dim=1, hidden_dim=128).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.examples = self.load_examples()
        self.logger = logging.getLogger(__name__)
        self.scaler = GradScaler()

    def load_examples(self):
        examples = [
            {"prompt": "Summarize the main argument presented in paragraph 3 of the PDF.", "accepted": "A concise, accurate summary of the key points from the specified paragraph, using objective language.", "rejected": "A summary that includes personal opinions, information from other paragraphs, or misrepresents the argument."},
            {"prompt": "Calculate the average value in column B of the CSV file.", "accepted": "The correct mathematical average of the values in column B, presented with appropriate precision.", "rejected": "An incorrect calculation, or an average from the wrong column or dataset."},
            {"prompt": "Identify the trend in data points 15-30 in the CSV file.", "accepted": "An accurate description of the trend (e.g., 'increasing linearly', 'fluctuating with an overall downward trend') based on the specified data points.", "rejected": "A trend description that contradicts the data or describes points outside the specified range."},
            {"prompt": "Compare the outcomes described in scenarios 2 and 4 of the PDF.", "accepted": "A balanced comparison highlighting key similarities and differences between the two scenarios, referencing specific details from each.", "rejected": "A comparison that misrepresents one or both scenarios, or that introduces information not present in the original text."},
            {"prompt": "What correlation exists between variables X and Y in the CSV?", "accepted": "An accurate statement about the correlation (positive, negative, or none) supported by the data, potentially including the correlation coefficient if calculable.", "rejected": "A claim about correlation that contradicts the data or assumes a relationship where none is evident."},
            {"prompt": "Explain how the concept introduced in section 2 of the PDF relates to the data in columns C and D of the CSV.", "accepted": "A logical explanation of the relationship, citing specific examples from both the PDF concept and the CSV data to support the connection.", "rejected": "An explanation that forces a connection where none exists, or that misinterprets either the concept or the data."},
            {"prompt": "What is the main limitation of the methodology described in the PDF's introduction?", "accepted": "An accurate identification of a key limitation, based on information provided in the introduction, with a brief explanation of its potential impact.", "rejected": "Claiming there are no limitations, or identifying a limitation not mentioned or implied in the text."},
            {"prompt": "Generate a hypothesis based on the findings in section 4 of the PDF and rows 50-75 of the CSV.", "accepted": "A plausible hypothesis that logically combines information from both sources, presented as a testable statement.", "rejected": "A hypothesis that contradicts the given information or makes unfounded leaps beyond the data provided."},
            {"prompt": "Identify any outliers in the dataset presented in the CSV file.", "accepted": "Accurate identification of data points that significantly deviate from the overall pattern, with brief justification for why they're considered outliers.", "rejected": "Labeling normal data variations as outliers, or failing to recognize clear anomalies in the dataset."},
            {"prompt": "Summarize the key differences between the approaches outlined in sections 1, 3, and 5 of the PDF.", "accepted": "A clear, concise comparison of the main points of each approach, highlighting key differentiators without bias.", "rejected": "A summary that blends the approaches, misrepresents their key features, or shows preference for one without justification from the text."},
            {"prompt": "Based on the CSV data, predict the likely value for variable Z if X increases by 10%.", "accepted": "A reasonable prediction based on the relationship between variables X and Z in the existing data, showing the calculation or logic used.", "rejected": "A prediction that ignores the established relationship in the data or uses incorrect mathematical reasoning."},
            {"prompt": "How does the case study in section 6 of the PDF exemplify the trend seen in column E of the CSV?", "accepted": "An explanation that accurately links the qualitative information from the case study to the quantitative trend in the CSV, providing specific examples.", "rejected": "Forcing a connection that isn't supported by the data, or misinterpreting either the case study or the trend."},
            {"prompt": "What ethical considerations are raised by the methodology described in the PDF, in light of the outcomes shown in the CSV?", "accepted": "A thoughtful discussion of potential ethical issues, referencing specific aspects of the methodology and data outcomes that could be problematic.", "rejected": "Claiming there are no ethical considerations, or raising ethical issues not related to the given information."},
            {"prompt": "Propose an alternative interpretation of the results presented in section 7 of the PDF, using the data from columns F and G of the CSV to support your argument.", "accepted": "A logical alternative interpretation that doesn't contradict the given data, supported by specific references to both the PDF content and CSV data.", "rejected": "An interpretation that ignores key information from either source, or that isn't substantively different from the original interpretation."},
            {"prompt": "Identify any inconsistencies between the conclusions drawn in the PDF and the data presented in the CSV.", "accepted": "Accurate identification of any areas where the PDF's conclusions aren't fully supported by the CSV data, with specific examples.", "rejected": "Claiming inconsistencies where none exist, or failing to recognize clear discrepancies between the two sources."},
            {"prompt": "Suggest a follow-up study based on the findings presented in both the PDF and CSV.", "accepted": "A well-reasoned proposal for further research that addresses gaps or questions raised by the current data, with clear objectives and methodology.", "rejected": "A suggestion that ignores the current findings, or proposes a study that would merely replicate the existing research without adding new insights."},
            {"prompt": "Evaluate the strength of the evidence presented in section 5 of the PDF using the statistical data from the CSV.", "accepted": "A balanced assessment that considers both qualitative arguments from the PDF and quantitative data from the CSV, noting areas of strong support and potential weaknesses.", "rejected": "An evaluation that misrepresents the strength of evidence, either by overstating weak connections or understating strong correlations."},
            {"prompt": "How might the findings from the CSV data affect the practical applications discussed in the conclusion of the PDF?", "accepted": "A thoughtful analysis of how the quantitative results could impact the proposed applications, considering both supporting and potentially challenging data points.", "rejected": "Ignoring relevant data from the CSV, or making sweeping claims about applications without considering the limitations of the findings."},
            {"prompt": "Create a brief executive summary that integrates key points from both the PDF report and the CSV data analysis.", "accepted": "A concise, well-structured summary that accurately represents the main findings from both sources, highlighting their interrelations and overall significance.", "rejected": "A summary that favors one source over the other, misses critical points, or fails to show how the PDF and CSV data complement each other."},
            {"prompt": "Based on the methodologies described in the PDF and the results in the CSV, what potential confounding variables should be considered in future research?", "accepted": "Identification of relevant confounding variables that could affect the results, explaining their potential impact and suggesting ways to control for them in future studies.", "rejected": "Listing irrelevant variables, or failing to explain how identified confounders relate to the specific methodologies and results presented."}
        ]
        return examples

    def generate_sample(self, idx):
        context = self.get_context_from_files()
        example = self.examples[idx % len(self.examples)]
        self.logger.info(f"Context generated: {len(context)} characters")
        state = RLHFState(context=context, example=example, cache_dir=self.cache_dir)
        return self.mcts(state)

    def get_context_from_files(self, _):
        self.logger.info("Generating context from files")
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        csv_files = [f for f in os.listdir(self.csv_folder) if f.endswith('.csv')]

        pdf_texts = [self.extract_text_from_pdf(os.path.join(self.pdf_folder, f)) for f in pdf_files]
        csv_summaries = [self.process_csv_file(os.path.join(self.csv_folder, f)) for f in csv_files]

        combined_context = " ".join(pdf_texts + csv_summaries)
        self.logger.info(f"Combined context length: {len(combined_context)} characters")
        return combined_context[:1000]

    def process_batch(self, batch):
        results = []
        for idx, context in enumerate(batch):
            state = RLHFState(context=context, example=self.examples[idx % len(self.examples)], cache_dir=self.cache_dir)
            result = self.mcts(state)
            results.append(result)
        return results

    def mcts(self, root, iterations=10):
        self.logger.info(f"Starting parallel MCTS with {iterations} iterations")
        
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [executor.submit(self.mcts_iteration, root) for _ in range(iterations)]
            results = [future.result() for future in futures]
        
        best_node = max(results, key=lambda x: x['reward'])
        self.logger.info("MCTS completed")
        return {
            "prompt": best_node['node'].prompt,
            "accepted": best_node['node'].accepted,
            "rejected": best_node['node'].rejected
        }

    def mcts_iteration(self, root):
        node = self.select(root)
        if not node.is_terminal():
            node.generate_accepted()
            node.generate_rejected()
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        else:
            reward = node.get_reward()
        return {'node': node, 'reward': reward}

    def select(self, node):
        while not node.is_terminal():
            if not node.children:
                return node
            node = max(node.children, key=lambda n: n.ucb_score())
        return node

    def expand(self, node):
        if not node.prompt:
            node.prompt = node.generate_prompt()
        elif not node.accepted:
            node.generate_accepted()
        elif not node.rejected:
            node.generate_rejected()
        return node

    def simulate(self, node):
        return node.get_reward()

    def backpropagate(self, node, reward):
        states = []
        actions = []
        rewards = []
        
        while node:
            state = [1.0 if node.prompt else 0.0, 1.0 if node.accepted else 0.0]
            action = [0.0 if node.rejected else 1.0]
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            node.visits += 1
            node.total_reward += reward
            node = node.parent

        self.batch_update(states, actions, rewards)

    def batch_update(self, states, actions, rewards):
        if len(states) < self.batch_size:
            return

        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1)

        with autocast():
            _, predicted_rewards, _ = self.model(states_tensor, actions_tensor)
            loss = F.mse_loss(predicted_rewards, rewards_tensor)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def generate_rlhf_data(self):
        self.logger.info("Starting RLHF data generation")
        start_time = time.time()

        with Pool(processes=cpu_count()) as pool:
            contexts = pool.map(self.get_context_from_files, range(self.num_samples))

        dataset = RLHFDataset(contexts, self.examples, self.cache_dir)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, 
                                pin_memory=True, collate_fn=custom_collate)

        rlhf_data = []
        for batch in dataloader:
            batch_results = self.process_batch(batch)
            rlhf_data.extend(batch_results)

        with open(self.output_file, 'w') as f:
            for item in rlhf_data:
                json.dump(item, f)
                f.write('\n')
        
        end_time = time.time()
        self.logger.info(f"RLHF data saved to {self.output_file}")
        self.logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")


    def process_csv_file(self, csv_path):
        df = pd.read_csv(csv_path)
        return self.tabular_processor.summarize_dataframe(df)

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    
if __name__ == "__main__":
    pdf_folder = "C:/Users/Admin/Desktop/ai-ml-web-app/backend/uploads/pdf"
    csv_folder = "C:/Users/Admin/Desktop/ai-ml-web-app/backend/uploads/csv"
    output_file = "rlhf_data.jsonl"
    cache_dir = "C:/Users/Admin/.cache/transformers"
    num_samples = 20

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    generator = RLHFDataGenerator(pdf_folder, csv_folder, output_file, cache_dir, num_samples)
    generator.generate_rlhf_data()





