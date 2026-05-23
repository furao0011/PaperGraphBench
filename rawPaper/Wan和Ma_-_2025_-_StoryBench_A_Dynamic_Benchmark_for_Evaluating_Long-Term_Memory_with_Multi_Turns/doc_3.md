has been proposed. For long-range discourse modeling in multi-session conversations, Multi-Session Chats [Xu et al., 2022] has been introduced. Agent-based evaluations such as AgentBench [Liu et al., 2024], WebArena [Zhou et al., 2023], and LLF-Bench [Cheng et al., 2024] offer dynamic environments for long-term interactions, focusing on multi-turn reasoning, real-world task completion, and learning from language feedback, respectively. While most of these works evaluate functional behavior, few explicitly isolate long-memory capabilities. A notable exception is LTM benchmark [Castillo-Bolado et al., 2024], which targets long-term memory in multi-turn conversations. However, existing benchmarks still face challenges in several aspects, especially in the evaluation of knowledge retention, sequential reasoning, and flexibility.

## 3 StoryBench

### 3.1 Motivation and Overview

Existing benchmarks apply static tasks (factual recall or isolated chain of thought tasks) that do not fully capture the dynamic nature of real-world interactions [Chang et al., 2024], suggesting there is room for improvement in evaluating LTM abilities in two critical dimensions: knowledge retention and sequential reasoning, as well as in their own flexibility. The limitation of current benchmarks results from their inability to simulate the dynamic, sequential nature of real-world decision-making, where memory must be actively updated, integrated with new information, and adapted to evolving contexts through multi-turn interactions.

To address this, we introduce StoryBench. The core design principle of StoryBench is to conduct memory stress-tests within a dynamic and sequentially structured environment grounded in interactive fiction multi-turn game-play. Unlike traditional benchmarks relying on static inputs or isolated memory recalls, StoryBench simulates realistic decision-making by embedding models in evolving narratives where each choice not only compels models to integrate information across short-term and long-term contexts (knowledge retention) but also tracks changing relationships between story elements and resolves contradictions arising from prior decisions in multi-turn interactions (sequential reasoning). In summary, StoryBench provides a more comprehensive and dynamic framework for evaluating long-term memory capabilities, effectively enhancing the assessment of knowledge retention and sequential reasoning, as well as improving the flexibility of the evaluation process.

### 3.2 Dynamic Narrative and Multi-Turn Decision-Making

StoryBench leverages the inherently dynamic and multi-turn nature of interactive fiction games to assess memory in realistic decision-making trajectories. Each run through the benchmark involves a sequence of interconnected choices, where past actions shape future outcomes. The model must continuously track character states, causal dependencies, and branching outcomes over extended contexts. This setup naturally embodies several key properties:

• Long-term: Many decisions require recalling events or facts introduced a few turns earlier. Concrete examples of such dependencies are provided in Section 4.2.

• Continuity: The benchmark follows a coherent plot, ensuring semantic continuity across interactions.

• Complex: Consecutive decisions are not isolated, but closely linked. One choice may directly affect the conditions or outcomes of several subsequent ones. We provide detailed illustrations of such dependencies in Section 4.2.

• Dynamic: Incorrect or suboptimal decisions dynamically alter the story path or trigger failure endings, requiring the model to adapt in real-time.

• Multi-turn: The task unfolds over many turns, demanding sustained memory and reasoning across sequentially extended interactions.

• Multi-solution: Many decision points allow for multiple acceptable paths, rather than a single fixed correct answer, better reflecting the uncertainty and variability of real-world scenarios. Specific examples demonstrating the multi-solution nature of the benchmark are provided in Section 4.2.