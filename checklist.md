| **Category**      | **Decision**         | **Details**                                                      |
| ----------------- | -------------------- | ---------------------------------------------------------------- |
| **LLM Algorithm** | **Dataset** | Anthropic HH-RLHF                   |
|                   | **Update Mechanism**            |  - PPO + RLHF <br> - DPO + SFT (for comparison)                    |
|                   | **Base Model**       | - DeepSeek-R1 <br> - LLaMA                                       |
|                   | **Framework**        | Huggingface Transformers + Trainer                               |
| **Reward Model**  | **Dataset**          | - lmsys/toxic-chat <br> - AllenAI RealToxicityPrompts           |
|                   | **Input / Output**   | **Input**: Prompt + LLM Response <br> **Output**: Score (0 \~ 1) |
|                   | **Base Model**       | DeBERTa-V3 (Encoder-based LM)                                    |
|                   | **Model**       | - OpenAssistant/reward-model-deberta-v3-large-v2     <br> - Skywork/Skywork-VL-Reward-7B                               |
|                   | **Loss Function**    | Pairwise Ranking Loss                                            |
| **Evaluation**    | **Metrics**          | Toxicity Scores                                                  |
|                   | **Tools**            | - OpenAI Moderation API <br> - Detoxify API                      |

### **LLM Algorithm**

* **Dataset**
    * Anthropic HH-RLHF
    * Kaggle Toxic Comment Classification Challenge
* **Update Mechanism**
  * PPO + RLHF
  * DPO + SFT
* **Base Model**
  * DeepSeek-R1
  * LLaMA
* **Framework**
    * Huggingface Transformers + Trainer

---

### **Reward Model**

* **Dataset**
  * lmsys/toxic-chat
  * AllenAI RealToxicityPrompts
* **Input / Output**
  * **Input**：Prompt + LLM Response
  * **Output**：Score（0 \~ 1）
* **Model**
    * **Train from scratch**: DeBERTa-V3（Encoder-based LM）
    * **FineTune**: Skywork/Skywork-Reward-V2-Qwen3-1.7B
* **Loss Function**
    * Pairwise Ranking Loss

---

### **Evaluation**

* **Metrics**
  * Toxicity Scores
* **Tools**
  * Detoxify API
  * OpenAI Moderation API

## Plan1

| **Category**      | **Decision**         | **Details**                                                      |
| ----------------- | -------------------- | ---------------------------------------------------------------- |
| **LLM Algorithm** | **Dataset** | Anthropic HH-RLHF                   |
|                   | **Update Mechanism**            |  PPO + RLHF                |
|                   | **Base Model**       | DeepSeek-R1            |
|                   | **Framework**        | Huggingface Transformers + Trainer                               |
| **Reward Model**  | **Required FT**          | False          |
|| **Dataset**          | AllenAI RealToxicityPrompts           |
|                   | **Input / Output**   | **Input**: Prompt + LLM Response <br> **Output**: Score (0 \~ 1) |
|                   | **Model**       |Skywork-Reward-V2-Qwen3-1.7B                                    |
| **Evaluation**    | **Metrics**          | Toxicity Scores                                                  |
|                   | **Tools**            | Detoxify                  |