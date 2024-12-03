---
license: apache-2.0

---

![image/png](https://cdn-uploads.huggingface.co/production/uploads/650268d0e373323dabb308c8/-u2AtTFfctjiQHoFQcWm9.png)

# Use in Colab

from huggingface_hub import snapshot_download

# Download the repository
repo_path = snapshot_download("RobbiePasquale/lightbulb")

print(f"Repository downloaded to: {repo_path}")

!PYTHONPATH=$PYTHONPATH:/root/.cache/huggingface/hub/models--RobbiePasquale--lightbulb/snapshots/3d255ef87272610b055f67937014c0b0f69a4b84 python main_menu.py --task advanced_inference --query "Analyze the economic effects of artificial intelligence in the next decade."

## Installation

To install the necessary dependencies, run:

```bash
pip install huggingface_hub torch transformers datasets argparse
```


### Download the Repository

Use the `huggingface_hub` to download the repository:

```python
from huggingface_hub import snapshot_download

# Download the repository
repo_path = snapshot_download("RobbiePasquale/lightbulb")

print(f"Repository downloaded to: {repo_path}")
```
### 0. Distill Large model into your own small model

## Minimal quick testing

```bash
python main_menu_new.py \
    --task distill_full_model \
    --teacher_model_name gpt2 \
    --student_model_name distilgpt2 \
    --dataset_name wikitext
```

## Full Distillation

```bash
python main_menu_new.py \
    --task distill_full_model \
    --teacher_model_name gpt2 \
    --student_model_name distilgpt2 \
    --dataset_name wikitext \
    --config wikitext-2-raw-v1 \
    --num_epochs 5 \
    --batch_size 8 \
    --max_length 256 \
    --learning_rate 3e-5 \
    --temperature 2.0 \
    --save_path ./distilled_full_model \
    --log_dir ./logs/full_distillation \
    --checkpoint_dir ./checkpoints/full_distillation \
    --early_stopping_patience 2
```

## Domain Specific Distillation

Use domain specific distillation to distill the part of the model relevant for you- if you like how llama 3.1 7B responds to healthcare prompts for example, you could use:

```bash
python main_menu_new.py \
    --task distill_domain_specific \
    --teacher_model_name gpt2 \
    --student_model_name distilgpt2 \
    --dataset_name wikitext \
    --config wikitext-2-raw-v1 \
    --query_terms healthcare medicine pharmacology \
    --num_epochs 5 \
    --batch_size 8 \
    --max_length 256 \
    --learning_rate 3e-5 \
    --temperature 2.0 \
    --save_path ./distilled_healthcare_model \
    --log_dir ./logs/healthcare_distillation \
    --checkpoint_dir ./checkpoints/healthcare_distillation \
    --early_stopping_patience 2


```

### 1. Train a Web Search Agent


**Usage:**
```bash
python main_menu.py --task train_agent
```

### 2. Use a Web Search Agent (Inference)

**Description:**  
Utilizes the trained web search agent to process queries, perform web searches, and generate summarized responses.

**Usage:**
```bash
python main_menu.py --task test_agent
```

**Options:**
- **Interactive Mode:**
  ```bash
  python main_menu.py --task test_agent
  ```
- **Single Query Mode:**
  ```bash
  python main_menu.py --task test_agent --query "Your query here"
  ```

### 3. Train Language Model

**Usage:**
```bash
python main_menu.py --task train_llm_world --model_name gpt2 --dataset_name wikitext --num_epochs 5 --batch_size 8 --max_length 256
```

**Key Arguments:**
- `--model_name`: Pretrained model (e.g., `gpt2`, `bert`).
- `--dataset_name`: Dataset from Hugging Face (e.g., `wikitext`).
- `--num_epochs`: Number of training epochs.
- `--batch_size`: Number of samples per batch.
- `--max_length`: Maximum sequence length.

### 4. Inference Using Language Model

**Usage:**
```bash
python main_menu.py --task inference_llm --query "Your query here"
```

### 5. Train World Model

**Description:**  
Develops a comprehensive World Model that encapsulates state representations, dynamics, and prediction networks to simulate and predict state transitions within the Tree of Thought framework.

**Usage:**
```bash
python main_menu.py --task train_world_model --additional_args
```

### 6. Inference with Language World Model

**Usage:**
```bash
python main_menu.py --task inference_world_model --query "Your query here"
```

### 7. Advanced Inference


**Usage:**
```bash
python main_menu.py --task advanced_inference --query "Your complex query here"
```


### Training the World Model

```bash
python main_menu.py --task train_llm_world --model_name gpt2 --dataset_name wikitext --num_epochs 5 --batch_size 8 --max_length 256
```

### Training the Web Search Agent

```bash
python main_menu.py --task train_agent
```

### Use the Web Search Agent in Interactive Mode

```bash
python main_menu.py --task test_agent
```

### Use the Web Search Agent with a Single Query

```bash
python main_menu.py --task test_agent --query "What are the impacts of renewable energy on global sustainability?"
```

### Inference with World Model and Tree of Thought

```bash
python main_menu.py --task advanced_inference --query "Analyze the economic effects of artificial intelligence in the next decade."
```


# Explanation:

World Model Optimisation:
-------------------------------------------------------------------
Input: I_i
-------------------------------------------------------------------
Rotary Positional Encoding:

emb_i = RoPE (Input)
-------------------------------------------------------------------
Token_i = t_i transformer(, k_beams = k, n_tokens = j)

CE_Loss = CE_loss(token_i , true tokens)

-------------------------------------------------------------------
Variance of the next token + Entropy of the sequence =  State Score
-------------------------------------------------------------------
Representation Network: 
GAN/VAE/SAE (o_t -> s_t)

If the final hidden layer of the transformer outputs o_t of size S

h_t = GELU(sum(W.o_t + b))

Reconstruction Loss (o_t , h_t)


-------------------------------------------------------------------
Dynamics Network (s_t -> s_t+1)

... -> LSTM(s_t)  -> LSTM(s_t+1) -> ...

min MSE (s_t+1 , z_t+1 )

State mapping
-------------------------------------------------------------------
Utilise dynamics influence:

Action_i = a_i = t_1 , ... , t_n

Prediction Network : mcts( Q(s,a) , gamma * LSTM(s_t) , delta * State Score (s_t),  tree_depth = m, num_simulations) -> Q(s_t+1)

action search / selection
-------------------------------------------------------------------
Optimise the KL divergence between the policy of actions (and the tokens that were selected in those actions) and the actual sequences in the training data.

Policy_i = p_i =  a_1, ... ,a_n

min - KL(p_i / true_sequences)

-------------------------------------------------------------------
Inference: 

Thought_i = p_i , ... , p_n

Tree of Thought : 
Example:

-----------------
1
-----------------
121
122
123
-----------------
12131
12132
12133

12231
12232
12233

12331
12332
12333
---------------

= Graph(system prompt, children = 3, depth = 4, min - KL(p_i / true_sequences))

Graph(Thought_i -> Thought i+1)

Min ThoughtLoss()

-------------------------------------------------------------------

Backpropagate back through each section, get gradients for:
-------------------------------------------------------------------
for thought batch size = b_t:

d ThoughtLoss
______________
d Graph(Thought_i -> Thought_i+1)

-------------------------------------------------------------------
for policy batch size = b_p:

d KL(p_i / true_sequences)
_______________________
d Prediction_Network

-------------------------------------------------------------------
for state batch size: b_s:

d MSE(s_t+1 , z_t+1 )
_________________
d Dynamics Network

-------------------------------------------------------------------
for state batch size: b_s:

d Contrastive Loss
________________
d Representation Network

-------------------------------------------------------------------
for token batch_size: b_to

d Multi token beam search Transformer CE Loss
__________________________________
d transformer


+++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++

Inference:

1. Input User Query
2. The model's goal is to generate a thought, which contains a set of policies, which contains sequences of actions, and an action is a sequence of tokens. 
3. The sequence of tokens is chosen using multi token prediction.
4. The Thought size is defined based on the user prompt, if the user prompt is in depth, then given the text in the input query, a larger output tree of thought. 
5. Perform the multi token beam search, depending on the action size, for each action will contain a multi token beam search (so an action will contain the state score of k beams for n tokens each time step, for a batch size of b_to).
6. PPO agent selects the actions given a mcts over actions using their state scores
7. Based on the tree of thought prompt tree, and given the sequence of actions selected for the policy, feed the chosen policy into the tree of thought, and get the Transformer Language Model to output token sequences based on the tree of thought prompts. There is an actor critic RL agent that selects the next child node in the tree of thought that is used, therefore learning to control how it responds to different user queries. The tree of thought should contain logic for decision making or solving problems in different ways. 
8. Update world model given external evaluation datasets

+++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++


Web Search Agent:

1. Given a user prompt, search N websites, using the input search query.
2. Given meta charactistics of he webpages, use FFN to rank the web pages
3. Utilise RAG to retrieve and summarize the content from the k highest ranking web pages given the user search query.
4. Extract and formulate the retrieved information into a custom dataset.
5. Feed the LLM and World Model the custom search dataset.


## Citation

If you use LightBulb in your research, please cite the author:

```
@misc{RobbiePasquale_lightbulb,
  author       = {Robbie Pasquale},
  title        = {LightBulb: An Autonomous Web Search and Language Model Framework},
  year         = {2024},
  publisher    = {Huggingface},
  howpublished = {\url{https://huggingface.co/RobbiePasquale/lightbulb}},
}
```



## License

This project is licensed under the Apache 2.0 License.

---
