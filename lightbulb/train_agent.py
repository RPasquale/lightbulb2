# train_agent.py
import sys
import os
from pathlib import Path

IS_COLAB = 'google.colab' in sys.modules

# Get the current file's directory
current_dir = Path(__file__).parent.absolute()

# Search for agent.py in the current directory and its parent directories
agent_path = None
search_dir = current_dir
while search_dir != search_dir.parent:  # Stop at root directory
    possible_path = search_dir / 'agent.py'
    if possible_path.exists():
        agent_path = str(search_dir)
        break
    search_dir = search_dir.parent

if agent_path:
    sys.path.insert(0, agent_path)
    print(f"Added {agent_path} to Python path")
else:
    print("Could not find agent.py")

# Now try to import AutonomousWebAgent
try:
    from lightbulb.agent import AutonomousWebAgent
    print("Successfully imported AutonomousWebAgent")
except ImportError as e:
    print(f"Error importing AutonomousWebAgent: {e}")
    sys.exit(1)

# Rest of your imports
from twisted.internet import reactor, defer, task
import random
import logging
import time
import codecs
# Configure logging
if IS_COLAB:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("agent_training.log", encoding='utf-8'),
                            logging.StreamHandler(codecs.getwriter('utf-8')(sys.stdout.buffer))
                        ])

logger = logging.getLogger(__name__)

# List of diverse queries
QUERIES = [
    "machine learning", "climate change", "renewable energy", "artificial intelligence",
    "quantum computing", "blockchain technology", "gene editing", "virtual reality",
    "space exploration", "cybersecurity", "autonomous vehicles", "Internet of Things",
    "3D printing", "nanotechnology", "bioinformatics", "augmented reality", "robotics",
    "data science", "neural networks", "cloud computing", "edge computing", "5G technology",
    "cryptocurrency", "natural language processing", "computer vision"
]

@defer.inlineCallbacks
def train_agent():
    # Updated state_size to 7 to match the feature extraction in AutonomousWebAgent
    state_size = 7  # word_count, link_count, header_count, semantic_similarity, image_count, script_count, css_count
    action_size = 3  # 0: Click Link, 1: Summarize, 2: RAG Generate
    num_options = 3  # 0: Search, 1: Summarize, 2: RAG Generate

    # Initialize the AutonomousWebAgent with the required arguments
    agent = AutonomousWebAgent(
        state_size=state_size,
        action_size=action_size,
        num_options=num_options,  # Added parameter for HRL
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        knowledge_base_path='knowledge_base.json'
    )
    logger.info(f"Initialized AutonomousWebAgent with state_size={state_size}, action_size={action_size}, num_options={num_options}")

    num_episodes = 10  # Adjust as needed
    total_training_reward = 0
    start_time = time.time()

    for episode in range(num_episodes):
        query = random.choice(QUERIES)
        logger.info(f"Starting episode {episode + 1}/{num_episodes} with query: {query}")
        episode_start_time = time.time()
        
        try:
            # Initiate the search process
            search_deferred = agent.search(query)
            search_deferred.addTimeout(300, reactor)  # 5-minute timeout
            total_reward = yield search_deferred
            total_training_reward += total_reward
            episode_duration = time.time() - episode_start_time
            logger.info(f"Episode {episode + 1}/{num_episodes}, Query: {query}, Total Reward: {total_reward}, Duration: {episode_duration:.2f} seconds")
        except defer.TimeoutError:
            logger.error(f"Episode {episode + 1} timed out")
            total_reward = -1  # Assign a negative reward for timeout
            total_training_reward += total_reward
        except Exception as e:
            logger.error(f"Error in episode {episode + 1}: {str(e)}", exc_info=True)
            total_reward = -1  # Assign a negative reward for errors
            total_training_reward += total_reward

        # Update target models periodically
        if (episode + 1) % 10 == 0:
            logger.info(f"Updating target models at episode {episode + 1}")
            agent.update_worker_target_model()
            agent.update_manager_target_model()
            agent.manager.update_target_model()

        # Log overall progress
        progress = (episode + 1) / num_episodes
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total_time - elapsed_time
        logger.info(f"Overall progress: {progress:.2%}, Elapsed time: {elapsed_time:.2f}s, Estimated remaining time: {remaining_time:.2f}s")

    total_training_time = time.time() - start_time
    average_reward = total_training_reward / num_episodes
    logger.info(f"Training completed. Total reward: {total_training_reward}, Average reward per episode: {average_reward:.2f}")
    logger.info(f"Total training time: {total_training_time:.2f} seconds")
    logger.info("Saving models.")

    # Save both Worker and Manager models
    agent.save_worker_model("worker_model.pth")
    agent.save_manager_model("manager_model.pth")
    agent.save("web_agent_model.pth")  # Assuming this saves additional components if needed
    
    if reactor.running:
        logger.info("Stopping reactor")
        reactor.stop()

def main(is_colab=False):
    global IS_COLAB
    IS_COLAB = is_colab
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Contents of current directory:")
    for item in os.listdir():
        print(f"  {item}")
    logger.info("Starting agent training")
    d = task.deferLater(reactor, 0, train_agent)
    d.addErrback(lambda failure: logger.error(f"An error occurred: {failure}", exc_info=True))
    d.addBoth(lambda _: reactor.stop())
    reactor.run()

if __name__ == "__main__":
    main(IS_COLAB)
