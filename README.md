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






## System Overview
```
                     ┌──────────────────────┐
                     │   Human Preference   │
                     │     Collection       │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   Reward Model       │
                     │   (Preference Model) │
                     └──────────┬───────────┘
                                │
 ┌──────────────────┐           ▼            ┌──────────────────┐
 │   Input Prompt   │───────▶  LLM  ───────▶ │  Generated Text  │
 └──────────────────┘         (Policy)        └──────────────────┘
                                │
                                ▼
                      ┌──────────────────────┐
                      │  Reward Model Score  │
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │      PPO Update      │
                      │  (Policy Optimization)│
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │   Updated LLM        │
                      └──────────┬───────────┘
                                 │
                   (Repeat Loop with New Prompts)
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │   Evaluation & Test  │
                      └──────────────────────┘
```