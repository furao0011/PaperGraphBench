# StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns
## Abstract

Long-term memory (LTM) is essential for large language models (LLMs) to achieve autonomous intelligence in complex, evolving environments. Despite increasing efforts in memory-augmented and retrieval-based architectures, there remains a lack of standardized benchmarks to systematically evaluate LLMs' long-term memory abilities. Existing benchmarks still face challenges in evaluating knowledge retention and dynamic sequential reasoning, and in their own flexibility, all of which limit their effectiveness in assessing models' LTM capabilities. To address these gaps, we propose a novel benchmark framework based on interactive fiction games, featuring dynamically branching storylines with complex reasoning structures. These structures simulate real-world scenarios by requiring LLMs to navigate hierarchical decision trees, where each choice triggers cascading dependencies across multi-turn interactions. Our benchmark emphasizes two distinct settings to test reasoning complexity: one with immediate feedback upon incorrect decisions, and the other requiring models to independently trace back and revise earlier choices after failure. As part of this benchmark, we also construct a new dataset designed to test LLMs' LTM within narrative-driven environments. We further validate the effectiveness of our approach through detailed experiments. Experimental results demonstrate the benchmark's ability to robustly and reliably assess LTM in LLMs.

## 1 Introduction

In the field of artificial intelligence, the pursuit of true intelligence in large language models (LLMs) has prompted researchers to look to biology for inspiration [Gutiérrez et al., 2024, Wu et al., 2025]. Just as organisms gradually accumulate knowledge through experience over time, LLMs need to possess long-term memory (LTM) capabilities to achieve self-evolution and strategic optimization in ever-changing environments [Shan et al., 2025]. Moreover, as LLMs are increasingly applied in scenarios such as multi-session dialogue [Zhang et al., 2025], task planning, and lifelong learning, the need for models to retain, update, and leverage prior knowledge dynamically becomes critical. Without robust LTM, AI systems are limited to short-term reasoning and static knowledge use, failing to achieve sustained, autonomous intelligence.

Given the importance of LTM in enabling advanced behaviors, it is crucial to evaluate these capabilities reliably and systematically. However, current benchmarks face challenges in adequately

evaluating LTM capabilities in two critical dimensions: 1) Knowledge Retention: the capacity to absorb, integrate, and preserve information across extended texts, maintaining contextual continuity beyond mere fact retrieval or local recall [Guo et al., 2025, EducateMe, 2024]; and 2) Sequential Reasoning: the ability to understand and reason about sequences of events, which involves inferring latent state changes, causal dependencies, and goal shifts across complex, dynamic, and multi-turn interactions rather than simply locating pre-stated answers within static text. 3) Flexibility: previous benchmarks often face challenges in adjusting and evaluating in different contexts.

To address these limitations, we propose a dynamic benchmark framework inspired by interactive fiction games, where LLMs engage in branching narratives with multi-turns that simulate long-term sequential decision-making. In our benchmark, the model continuously receives scene descriptions, dialogues, and options, and must make choices based on its understanding. We design two modes: Immediate Feedback provides immediate feedback when the model makes a wrong choice, while Self Recovery allows the story to continue toward a failure ending without any hint, requiring the model to identify and revise past decisions on its own. Through this setup, our benchmark effectively evaluates the model's ability to remember key information (knowledge retention) and reason over event sequences (sequential reasoning). Furthermore, our benchmark demonstrates excellent flexibility in accommodating diverse scenarios.

To further illustrate the advantages of our benchmark, we comprehensively evaluate the differences between existing benchmarks and ours (Table 1) based on the following aspects:

Knowledge Retention. Long-context (L-ctx) evaluates whether the task requires long-term memory of earlier context to succeed. Continuity (Conty) measures whether the benchmark requires the model to maintain a coherent understanding of entities, events, and their relationships across interactions.

Sequential Reasoning. Complexity (Comp.) indicates whether the benchmark features nonlinear reasoning tasks, where multiple interdependent events or decisions must be jointly considered, requiring the model to reason beyond sequential context. Dynamics (Dyn.) refers to whether the model's actions or responses influence future tasks or states in the environment. Multi-turn (M-turn) evaluates whether the task involves multiple sequential interactions, where each turn is temporally connected to the previous ones.

Flexibility. Multi-solution (M-sol) indicates whether the benchmark includes tasks or questions with multiple valid answers or approaches, rather than a single fixed solution. LTM+STM evaluates the combined usage of long-term memory (LTM) and short-term memory (STM), i.e., whether the task requires reasoning over both recent and distant information.

[Table 1 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 1: Comparison of Existing Benchmarks across Multiple Dimensions.

Summary: This table compares existing benchmarks (NeedleInHaystack, RULER, LTM Benchmark, BABILong, L-Eval, LongBench, LooGLE, InfiniteBench, and StoryBench) across multiple dimensions including type (Synthetic, Realistic, Hybrid), knowledge retention (Long-context and Continual), sequential reasoning (Compositional, Dynamic, Multi-turn), and flexibility (Multiple-solution and Long-term+Short-term memory). It shows that the authors' benchmark, StoryBench, supports all listed dimensions, while other benchmarks support only subsets.

Table LaTeX:

```latex
\begin{tabular}{lllllllll}
\hline
Benchmark & Type & Knowledge Retention & Knowledge Retention & Sequential Reasoning & Sequential Reasoning & Sequential Reasoning & Flexibility & Flexibility \\
\hline
Benchmark & Type & L-ctx & Conty & Comp. & Dyn. & M-turn & M-sol & LTM+STM \\
NeedleInHaystack & Synthetic & ✓ & ✗ & ✗ & ✗ & ✗ & ✗ & ✗ \\
RULER & Synthetic & ✓ & ✗ & ✓ & ✗ & ✗ & ✗ & ✗ \\
LTM Benchmark & Synthetic & ✓ & ✗ & ✓ & ✗ & ✓ & ✗ & ✓ \\
BABILong & Synthetic & ✓ & ✗ & ✗ & ✗ & ✗ & ✗ & ✓ \\
L-Eval & Realistic & ✓ & ✗ & ✗ & ✗ & ✗ & ✓ & ✓ \\
LongBench & Hybrid & ✓ & ✗ & ✗ & ✗ & ✗ & ✗ & ✓ \\
LooGLE & Hybrid & ✓ & ✗ & ✓ & ✗ & ✗ & ✗ & ✗ \\
InfiniteBench & Hybrid & ✓ & ✗ & ✗ & ✗ & ✗ & ✓ & ✓ \\
Ours(StoryBench) & Hybrid & ✓ & ✓ & ✓ & ✓ & ✓ & ✓ & ✓ \\
\hline
\end{tabular}
```

To validate the effectiveness of our benchmark, we conduct systematic evaluations on advanced four LLMs. Each model is tested under both evaluation modes across 80+ branching story paths, with performance measured in terms of correct decision rates, task success counts, etc. Results show that while GPT-4o [OpenAI, 2024] and Claude 3.5 Sonnet [Anthropic, 2024] demonstrate relatively stronger long-term knowledge retention and sequential reasoning, all models struggle with self-recovery and fail to consistently revise earlier mistakes. In-depth failure analysis further reveals distinct memory bottlenecks, which existing benchmarks could be enhanced to expose. These findings confirm the utility of StoryBench in capturing LTM deficiencies, offering a more granular and realistic assessment than prior benchmarks.

Our contributions are as follows:

[Figure 6 was here. The original paper contained a figure at this position. Brief visual description: The figure presents two radar charts comparing the multidimensional performance of five AI models (Doubao1.5-pro, QFI-Mo, Claude 3.5 Sonnet, Deepseek-R1) across two modes: 'Immediate Feedback' (left) and 'Self Recovery' (right). The charts evaluate metrics including First-Try Accuracy, Overall Accuracy, Hard Accuracy, Success Count, and Reversed Retry Count. The visual data highlights differences in model capabilities, such as Claude 3.5 Sonnet achieving high success counts and Doubao1.5-pro (Improved) showing strong accuracy in self-recovery scenarios.]
Caption: Figure 6: Model multidimensional performance in Immediate Feedback and Self Recovery modes.
Key visible elements:
- Immediate Feedback Chart: Left radar chart displaying model performance under immediate feedback conditions.
- Self Recovery Chart: Right radar chart displaying model performance under self-recovery conditions.
- Performance Axes: Five radial axes representing metrics: First-Try Acc, Overall Acc, Hard Acc, Success Count, and Reversed Retry Count.
- Model Legends: Color-coded keys identifying specific models and their variants (e.g., Original vs. Improved).
- Data Polygons: Colored shapes representing the performance profile of each model on the radar chart.

[Figure 7 was here. The original paper contained a figure at this position. Brief visual description: The figure consists of two bar charts comparing the accuracy of various AI models (Doubao, GPT-4o, Claude 3.5 Sonnet, Deepseek-V3) across three task categories: Overall Accuracy, Easy Accuracy, and Hard Accuracy. The first chart displays performance under 'Immediate Feedback', while the second chart shows performance under 'Self Recovery' conditions, including specific variations like 'improved' or 'original' versions of the models.]
Caption: Figure 7: Accuracy disparities across Models: overall, easy & hard tasks.
Key visible elements:
- Immediate Feedback Chart: Left panel showing baseline model performance with immediate feedback
- Self Recovery Chart: Right panel showing model performance with self-recovery mechanisms
- X-axis Categories: Labels for different AI models (e.g., Doubao-1.5-v3, GPT-4o)
- Y-axis (Accuracy %): Vertical scale measuring performance percentage from 50 to over 90
- Legend: Color coding for Overall Acc (blue), Easy Acc (green), and Hard Acc (orange)
- Data Bars: Visual representation of accuracy scores for each model and task type

#### 5.2.2 Insights of Distinctions Between Two Modes

To investigate how short-term and long-term memory settings affect model behavior, we compare performance under two task modes. Immediate Feedback mode provides corrective signals after each wrong choice, effectively mimicking short-term memory and aiding models in adjusting quickly. In contrast, Self Recovery better simulates real long-term memory scenarios by removing such signals, requiring the model to navigate the narrative without external guidance.

Unsurprisingly, all models perform worse under Self Recovery mode, as shown by the consistent drop in Overall Accuracy and Success Count. This highlights the increased difficulty of sustained sequential reasoning and knowledge retention without short-term feedback. To alleviate task failure in extreme cases, we introduce an auxiliary intervention metric: Number of Choices Reaching Error Threshold (we set the threshold to 9). If a model makes the same mistake more than 9 times, it is prompted with the correct answer. Only Claude 3.5 and GPT-4o never reach this threshold, suggesting that their task completions in Self Recovery mode are entirely due to self-correction and internal reasoning without any artificial hints. This contrasts sharply with other models, indicating that they excel in sustained sequential reasoning and knowledge retention.

Surprisingly, despite the overall decline in performance across models in Self Recovery, two metrics: Longest Consecutive Correct Sequence and First-Try Accuracy actually increase for several models (Figure 8). This amazing trend emphasizes that while short-term feedback aids local correction, it may also disrupt long-horizon coherence. By removing it, models foster a deeper narrative understanding (knowledge retention) and more coherent reasoning (sequential reasoning) and we better expose the true limitations and strengths of long-term memory in different models.

[Figure 8 was here. The original paper contained a figure at this position. Brief visual description: Figure 8 consists of two horizontal bar charts comparing the performance of four language models (Deepseek-R1, Claude 3.5 Sonnet, GPT-4o, Doubao1.5-pro) under two modes: 'Immediate Feedback' (blue bars) and 'Self Recovery' (orange bars). The left chart displays the 'Longest Consecutive Correct Sequence', while the right chart displays 'First-Try Accuracy'.]
Caption: Figure 8: Mode impact on models: First-Try Accuracy & Longest Consecutive Correct Sequence metrics.
Key visible elements:
- Left Chart (Longest Consecutive Correct Sequence): Displays the average length of consecutive correct responses for each model and mode.
- Right Chart (First-Try Accuracy): Displays the percentage of correct answers on the first attempt for each model and mode.
- Immediate Feedback Mode: Baseline condition represented by blue bars.
- Self Recovery Mode: Experimental condition represented by orange bars.
- Model Labels: Y-axis categories identifying the specific LLMs being tested.

A notable case is Deepseek-R1. While it does not lead in most individual metrics, it demonstrates remarkable consistency across both Immediate Feedback and Self Recovery modes. This stable performance suggests that the model is capable of making accurate revisions during backtracking.

### 5.3 Failure Case Study

In evaluating long-term memory capabilities with StoryBench, we identified two principal types of failure that reflect limitations in current language models, corresponding to the core dimensions of knowledge retention and sequential reasoning.

The most prominent issue in knowledge retention was the failure to preserve contextual consistency over extended narratives. Models frequently made decisions that contradicted earlier story events, character motivations, or established world logic. This suggests difficulty in integrating and maintaining distributed information over long spans of interaction, especially when the necessary context spans dozens of turns. Even when the relevant facts appeared in the prompt, models struggled to apply them coherently, indicating limitations beyond simple factual recall.

In terms of sequential reasoning, a critical failure case was the inability to repair long-term or multi-error decisions. In Self Recovery mode, successful completion often required models to trace errors back across multi-step causal chains and revise earlier decisions (even multiple choices in combination) that affected downstream outcomes. However, most models exhibited shallow search strategies, typically backtracking only one or two steps rather than engaging in deeper reasoning about the narrative structure or goal shifts. This myopic behavior led to persistent failure when task success depended on understanding and correcting long-term dependencies. We retained such failures to reflect the true upper-bound difficulty of long-term memory reasoning.

Other failures such as format mismatches (e.g., returning option indices instead of decision point IDs), content filtering blocks, server timeouts, or rare instances of hallucinated explanations were also observed but were comparatively infrequent. These were retained in evaluation for completeness but are not the focus of our analysis.

These diverse failure cases underscore the challenge of StoryBench and emphasize the need for more robust memory integration, format alignment, and long-range error correction in current foundation models.

## 6 Limitations

While our benchmark provides a comprehensive evaluation of long-term memory capabilities in large language models through complex, branching narrative tasks, it has several limitations. First, the scenarios are derived from a single interactive fiction domain and the interactive environment is

text-based, both of which may limit the benchmark's generalizability to other knowledge-intensive or task-oriented contexts that require multimodal support. Second, the number of turns and the length of the context are still limited. The current interactive fiction dataset consists of only 6 chapters, which may not fully capture the long-term dependencies and complex reasoning required in more extensive narratives. Future work could expand the dataset by adding subsequent chapters to provide a more comprehensive evaluation of long-term memory. Third, due to API constraints and cost, we primarily evaluate a limited number of mainstream models. The performance of other models under similar conditions remains unexplored. Fourth, although we include a self-recovery setting to simulate real-world error correction, the evaluation remains scripted and cannot capture all forms of natural feedback.

## 7 Conclusions

We introduce StoryBench, a novel benchmark designed to systematically evaluate long-term memory capabilities in complex, dynamic, and multi-turn narrative environments. By simulating realistic memory demands across story understanding, sequential inference, and flexible correction, our benchmark assesses current mainstream models in knowledge retention and sequential reasoning. Through comprehensive experiments on representative models and detailed failure case analyses, we demonstrate that current models exhibit significant performance gaps on our benchmark, highlighting StoryBench's difficulty and effectiveness in evaluating long-term memory capabilities. Our findings underscore the importance of developing more robust memory mechanisms, laying the groundwork for future research toward memory-augmented, context-aware language agents.