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
|                   | **Tools**            | OpenAI Moderation API                    |

Here is your **Plan 1 Checklist** based on your table:

---

## ✅ Checklist

### 🔹 **LLM Algorithm Setup**

* [x] Download **Anthropic HH-RLHF Dataset**
* [ ] Fine-tune **DeepSeek-R1** using **SFT with HH Dataset**
* [ ] Set up PPO + RLHF Training Loop with Huggingface TRL
* [ ] Integrate Reward Scoring in PPO Training (reward from reward model inference)
* [ ] Complete PPO-based RLHF fine-tuning

---

### 🔹 **Reward Model (Inference Only)**

* [ ] Download **Skywork-Reward-V2-Qwen3-1.7B**
* [ ] Load and integrate reward model into PPO loop (inference only, no fine-tuning)
* [ ] Validate reward model scoring logic on sample prompts

---

### 🔹 **Evaluation & Validation**

* [ ] Download **AllenAI RealToxicityPrompts**
* [ ] Set up **OpenAI Moderation API** for output filtering
* [ ] Test RLHF-trained LLM on RealToxicityPrompts
* [ ] Measure Toxicity Scores on generated outputs
* [ ] Compare baseline (pre-RLHF) vs. RLHF-tuned model on toxicity metrics

---

### 🔹 **Final Deliverables**

* [ ] PPO + RLHF-tuned **DeepSeek-R1 Checkpoint**
* [ ] Toxicity Evaluation Report (with OpenAI Moderation & RealToxicityPrompts)
* [ ] Summary of Toxicity Reduction and Model Behavior

---

### 🔹 Optional but Recommended

* [ ] Set up Detoxify for secondary toxicity scoring
* [ ] Human-in-the-loop evaluation on sensitive prompts
