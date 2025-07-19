

## Dataset
* format: json
* #training: 160K; #testing: 8k




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