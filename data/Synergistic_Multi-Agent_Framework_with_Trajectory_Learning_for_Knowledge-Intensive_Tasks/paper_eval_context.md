# Synergistic Multi-Agent Framework with Trajectory Learning for Knowledge-Intensive Tasks
## Abstract

Recent advancements in Large Language Models (LLMs) have led to significant breakthroughs in various natural language processing tasks. However, generating factually consistent responses in knowledge-intensive scenarios remains a challenge due to issues such as hallucination, difficulty in acquiring long-tailed knowledge, and limited memory expansion. This paper introduces SMART, a novel multi-agent framework that leverages external knowledge to enhance the interpretability and factual consistency of LLM-generated responses. SMART comprises four specialized agents, each performing a specific sub-trajectory action to navigate complex knowledge-intensive tasks. We propose a multi-agent co-training paradigm, Long Short-Trajectory Learning, which ensures synergistic collaboration among agents while maintaining fine-grained execution by each agent. Extensive experiments on five knowledge-intensive tasks demonstrate SMART's superior performance compared to widely adopted knowledge internalization and knowledge enhancement methods. Our framework can extend beyond knowledge-intensive tasks to more complex scenarios.

## Introduction

Researchers continue to pursue empowering intelligent systems to generate factually consistent responses in knowledge-intensive tasks (Singhal et al. 2022; Yue et al. 2023a; Wang et al. 2022a). Although Large Language Models (LLMs) internalize substantial world knowledge within their parameter memory, they still suffer from fabricating facts, due to their inherent drawbacks, e.g., hallucination (Ji et al. 2023), trouble in acquiring long-tailed knowledge (Kandpal et al. 2023) and struggle to expand their memory (De Cao, Aziz, and Titov 2021). These issues significantly underscore the necessity of incorporating external knowledge from non-parametric (i.e., retrieval-based) memories.

Current methods typically augment LLMs with retrieved knowledge to generate responses, which face three main challenges. (1) Complex query intent: the diverse nature (semantics and form) of instructions (e.g., multiple choice,

[Figure 1 was here. The original paper contained a figure at this position. Brief visual description: The figure illustrates a multi-agent framework for knowledge-intensive tasks, divided into two main sections. The top section depicts a workflow trajectory starting from a task instruction, moving through intent reconstruction and knowledge access, then to fact identification and relevance discrimination, finally producing a response with citations. The bottom section compares three optimization strategies for this framework: (a) Modular Optimization, (b) End-to-end Optimization, and (c) the proposed 'Our Optimization' method, using robot icons and arrows to denote agent interactions and optimization paths.]
Caption: Figure 1: Example of our long trajectory for knowledge-intensive scenarios (Top) and optimization comparison of multi-agent frameworks (Bottom). Solid and dashed arrows indicate inference and optimization paths, respectively.
Key visible elements:
- Task Instruction: Input component of the workflow
- Reconstructing Intent: Intermediate processing step converting instructions to intent
- Accessing Knowledge: Retrieval component fetching relevant information
- Identifying Facts: Extraction component listing potential facts
- Discriminating Relevance: Filtering component selecting relevant facts
- Response&Cites: Output component generating the final answer
- (a) Modular Optimization: Baseline comparison showing separate optimization blocks
- (b) End-to-end Optimization: Baseline comparison showing joint optimization of all blocks

multi-turn dialogue, and complex questions) leads to confusion regarding the query intent of knowledge. (2) Distractors in retrieved knowledge: knowledge retrieval inevitably introduces noises of varying granularity (document and sentence), with irrelevant documents and superfluous spans distracting the response and resulting in more severe hallucinations. (3) Insufficient knowledge utilization: LLMs tend to rely more on their implicit knowledge (parameter memory) rather than fully exploiting provided external facts (Huang et al. 2023). This fact-following disloyalty invalidates the knowledge incorporation process. Existing knowledge enhancement efforts (Shi et al. 2023; Ma et al. 2023; Asai et al. 2023) do not comprehensively address these multi-stage challenges. To this end, we propose a multi-agent framework, SMART, to integrate different actions to tackle all challenges within complex knowledge-intensive tasks, where each agent performs a specific action. This comprises an Intent Reconstructor to clarify knowledge intents, a Knowledge Retriever to access external knowledge based on intent, a Fact Locator to evaluate retrieved knowledge and

identify factual spans, and a Response Generator that faithfully utilizes and cites available facts. This process can enhance the knowledge interpretability and response factuality.

However, a major concern remains in how to equip each agent with the necessary capability for corresponding actions while minimizing errors during agent streamline for better overall knowledge-intensive performance. This has been a longstanding challenge in improving multi-agent frameworks, especially as most (Yao et al. 2023; Hong et al. 2023) operate in a non-training manner. Specifically, On one hand, modular operations, where separate learned modules are pipelined with each dedicated to a specific agent, can streamline the processing. However, this can lead to error accumulation as mistakes in earlier modules propagate through the pipeline. On the other hand, encouraging LLM variants to imitate the entire trajectory, while mitigating the fragmentation and error propagation seen in modular systems, this long-term and global supervision cannot guarantee the precise fine-grained execution by each agent, as it fails to balance the attention each agent devotes to diverse input signals. Overall, maintaining synergy while ensuring the contribution of various stakeholders is essential.

To address this, we propose a multi-agent cooperative training method, namely Long Short-Trajectory Learning, which consists of two stages. In the first stage, short trajectory learning activates each specific agent in the framework. Next, long trajectory learning ensures synergy across multi-agents through trajectory skeleton learning. To establish a common supervisory signal for both phases while achieving different training objectives for each, we design special tokens (i.e., trajectory head-end tokens) to allow each agent to identify the attributed trajectories and learn interagent interaction signals during training. Specifically, the former phase learns the task output under the prompt of the trajectory-head token, so that the framework learns to distinguish between different agents and confirm the fine-grained information of interest. This independence enables more efficient training with the utilization of existing NLP datasets for pre-training and targeted optimization. The latter stage requires both predictions of task output and intermittent trajectory tokens throughout the process, i.e., establishing a navigation path from the previous agent to the next. Our learning approach enables multi-agent systems to collaboratively navigate a long and complex trajectory while concurrently upholding a nuanced representation of each agent.

We conduct experiments on five knowledge-intensive tasks, including fact verification, multiple-choice reasoning, open-domain question answering and long-form generation. Results demonstrate that our framework significantly outperforms pre-trained and instruction-tuned LLMs with more parameters (knowledge internalization methods), and widely adopted knowledge enhancement methods. Further analysis reveals that our long-short trajectory learning enables flexible plug-in combinations of agents while maintaining performance, which is beyond the reach of current end-to-end training systems. Additionally, the framework achieves impressive performance using only over 40% of long trajectory data, substantially reducing the cost and complexity of developing a high-performance multi-agent framework. We envision our framework as a general paradigm that extends beyond knowledge-intensive tasks to more complex scenarios, enabling any multi-agent framework to internalize tailored trajectories.

## Method

Figure 2 provides an overview of our co-framework. We first introduce our multi-agent framework with four key agents performing distinct trajectories. Next, we explain the data construction method and detail the Long-Short Trajectory Learning for optimizing framework synergies.

## Multi-Agent Framework

To address multi-stage complex challenges in knowledge-intensive scenarios, we design a multi-agent framework to execute complex long trajectories. This framework incorporates four key agents: intent reconstructor ( $ A_{i} $), knowledge retriever ( $ A_{r} $), fact locator ( $ A_{l} $), and response generator ( $ A_{g} $). Each agent serves a specific sub-trajectory, and the final response is obtained by synergizing these agents.

Intent Reconstructor. The $ A_i $ agent aims to clarify the knowledge query intent from user instructions. It possesses four primary capabilities: integrating contextual clues, identifying key query, unifying task formulation, and intent decomposition, to handle diverse instructions. For example, in multi-turn dialogues, $ A_i $ models long-term history for intent. For noisy instructions, it filters out irrelevant information to identify key queries. For various task formats such as multichoice QA, $ A_i $ formulate all inputs as a query format for subsequent processing. When handling multi-hop queries like “Who was born earlier, person A or person B?”, $ A_i $ breaks them down into multiple sub-intents, i.e., each person’s birth date. By flexibly applying these capabilities, this agent obtains clear query intent to access external knowledge.

Knowledge Retriever. The $ A_{r} $ agent accesses external knowledge bases (e.g., Wikipedia) and obtains relevant knowledge candidates based on reconstructed intents. Specifically, it is driven by an off-the-shelf retrieval model (Izacard et al. 2021) and acquires top-k knowledge document candidates from the knowledge base for each knowledge intent. Details of our knowledge retriever setup and the corpus are described in Appendix Sec. B.3.

Fact Locator. The $ A_{1} $ agent aims to locate factual evidence from knowledge candidate sets via document- and sentence-level assessments. Specifically, it assesses the relevance of each knowledge document to the given instruction to determine relevant ones. It then identifies the factual spans from relevant documents as evidence. The fact locator serves two primary purposes: 1) It enables the agent to check its relevance judgments to minimize the distraction of extraneous spans of the document, and allows the response phase to focus more on fact spans. 2) By explicitly learning to locate facts, it enhances the interpretability of the knowledge application process and bolsters user credibility.

[Figure 2 was here. The original paper contained a figure at this position. Brief visual description: Figure 2 illustrates a multi-agent framework for knowledge-intensive tasks, divided into a high-level architecture overview (top) and a detailed trajectory example (bottom). The top section shows four sequential agents: Intent Reconstructor, Knowledge Retriever, Fact Locator, and Response Generator. The bottom section demonstrates a 'Long Trajectory' using a specific question about Joe Colquhoun and Carlton Loewer, breaking the process into four 'Short Trajectories' corresponding to each agent's function.]
Caption: Figure 2: Overview of our multi-agent framework with long- and short-trajectory learning. This framework incorporates four agents: intent reconstructor, knowledge retriever, fac locator, and response generator.
Key visible elements:
- Knowledge-intensive tasks: Input category box listing task types like Fact Verification and Multiple-choice Reasoning
- Intent Reconstructor: First agent that processes instructions into retrieval intents
- Knowledge Retriever: Second agent that fetches external knowledge based on intents
- Fact Locator: Third agent that judges relevance and locates specific facts in paragraphs
- Response Generator: Final agent that generates the task response with citations
- Long Trajectory: The complete workflow example shown in the bottom half
- Short Trajectory 1-4: Individual steps within the example workflow corresponding to each agent

Response Generator. The $ A_{g} $ agent finally generates responses to user instructions. When facts are provided, it adjusts its knowledge preferences to adhere to them, and ultimately outputs citations to validate loyalty further. In the absence of such information, the response generator relies on its knowledge memory to formulate responses.

Inference Overview. The systematic procedure is delineated in the following steps: $ A_i $ first mines the explicit intent $ \bar{q} = \{q_1, q_2, ..., q_m\} $ from the instruction $ x $. Next, $ A_r $ retrieves top- $ k $ knowledge documents $ \bar{d} = \{d_1, d_2, ..., d_{k \times m}\} $ using each intent $ q_m $. Then, $ A_l $ determines each relevant knowledge passage and further locates the fact span $ f \subset d_{k \times m} $. Finally, $ A_g $ utilizes the previous execution trajectory to generate response $ y $ and citations when facts exist, otherwise $ A_g $ utilizes only $ x $. In the $ t $-th step, the Agent $ A $ generates a response $ r_t $ and a head token $ h_{t+1} $ of the next trajectory based on the current state of the system:

$$ r_{t},h_{t+1}=\mathcal{A}\left(x,\tau_{t-1}\right), $$

where $ \tau_{t-1} = \{h_1, r_1, e_1, ..., h_{t-1}, r_{t-1}, e_{t-1}\} $ denotes the previous execution trajectory. $ e $ denotes the trajectory end token. In addition, $ A_i $, $ A_l $ and $ A_g $ are built upon same LLMs to fulfill their roles. The pseudo-code for inference is referenced in Appendix.

## Trajectory Dataset Construction

To implement long-short trajectory learning to optimize our multi-agent framework, we construct the Trajectory dataset. We collect samples from over 12 knowledge-intensive tasks to ensure coverage of various instruction semantics and formats, such as fact verification (Thorne et al. 2018), dialogue (Dinan et al. 2018; Anantha et al. 2021), open-domain Q&A (Kwiatkowski et al. 2019; Stelmakh et al. 2022; Geva et al. 2021), and commonsense reasoning (Mihaylov et al. 2018; Huang et al. 2019). Detailed statistics are in Table 5 of Appendix. Our dataset contains two components: the long-trajectory subset and the short-trajectory subset. The data construction follows two distinct principles:

Long-trajectory subset. The long-trajectory subset aims to precisely mimic our multi-agent framework inference-time process, which emphasizes the synergy and logical interaction between agents. Existing work (Asai et al. 2023) has demonstrated the effectiveness of the powerful LLM (e.g., GPT3.5, GPT4 (Achiam et al. 2023)) as a critic model. Given an input-output pair $ (x, y) $, we create supervised data under the guide of the retrieval (R) and critic model (C). We enable C to unleash the knowledge intents $ \bar{q} $ in x according to the instruction type. Then, R retrieves the top-k knowledge documents based on every $ \bar{q} $. For each document, C further evaluates whether the passage is relevant based on $ (x, y) $. If a passage is relevant, C further locates and extracts the fact spans. Finally, we combine the data and insert the trajectory header and end token (e.g., ⟨Reconstructor⟩, ⟨/eor⟩) into each trajectory. Trajectory tokens are identifiers that serve as the skeleton of the multi-agent framework. In total, we construct 142,507 elaborated instances.

Short-trajectory subset. Unlike the long-trajectory subset, the short-trajectory subset facilitates the training of individual capabilities for each intelligent agent. This isolation allows us to acquire data directly from a huge amount of existing knowledge-intensive tasks through some simple processing. Thus, we sample from the established NLP and SFT datasets, appending the requisite trajectory header and

[Figure 3 was here. The original paper contained a figure at this position. Brief visual description: The figure illustrates a 'Long-Short Trajectory Learning' framework, contrasting short trajectory learning (top section) with long-trajectory learning (bottom section). The top section displays three distinct modules—Intent Reconstructor, Fact Locator, and Response Generator—each processing specific tokens. The bottom section shows how these individual components are combined into a single, continuous sequence of tokens for long-trajectory learning, connected by colored arrows indicating the mapping between short and long trajectories.]
Caption: Figure 3: Overview of Long-Short Trajectory Learning. It consists of two stages, for short trajectory learning, under a given trajectory head, requires insight into the various explicit and implicit signals in each particular task. For long-trajectory learning, LLM executes the entire process by predicting different trajectory tokens, ensuring the synergism of different short-trajectories.
Key visible elements:
- Intent Reconstructor: A module in the top section responsible for intent reconstruction, highlighted in green.
- Fact Locator: A module in the top section responsible for locating facts/retrieval, highlighted in orange.
- Response Generator: A module in the top section responsible for generating responses, highlighted in blue.
- Long-Trajectory Learning Sequence: A unified sequence in the bottom section combining tokens from the upper modules.
- Colored Arrows: Visual connectors mapping specific short-trajectory components to their corresponding positions in the long-trajectory sequence.

[Table 1 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 1: Four types of trajectory tokens. $ x $, $ \bar{q} $, $ \bar{d} $, $ \gamma $, $ \bar{f} $ and $ \bar{y} $ indicate instruction, intent, knowledge document, relevance tag, fact evidence and response, respectively.

Summary: The table defines four types of trajectory tokens used in a trajectory dataset construction: Reconstructor (A_i), Retrieval (A_r), Locator (A_l), and Generator (A_g). For each type, it specifies the head token (e.g., <Reconstructor>), end token (e.g., </eor>), input variables (e.g., x for Reconstructor), and output variables (e.g., \bar{q} for Reconstructor). Notation is provided in the caption: x=instruction, \bar{q}=intent, \bar{d}=knowledge document, \gamma=relevance tag, \bar{f}=fact evidence, and y=response.

Table LaTeX:

```latex
\begin{tabular}{lllll}
\hline
Type & Trajectory Tokens & Trajectory Tokens & Input & Output \\
\hline
Type & Head & End & Input & Output \\
\$ \textbackslash{}mathcal\{A\}\_i \$ & <Reconstructor> & </eor> & x & \$ \textbackslash{}bar\{q\} \$ \\
\$ \textbackslash{}mathcal\{A\}\_r \$ & <retrieval> & </retrieval> & \$ \textbackslash{}bar\{q\} \$ & \$ \textbackslash{}bar\{d\} \$ \\
\$ \textbackslash{}mathcal\{A\}\_l \$ & <Locator> & </eol> & x, \$ \textbackslash{}bar\{d\} \$ & \$ \textbackslash{}gamma,\textbackslash{}bar\{f\} \$ \\
\$ \textbackslash{}mathcal\{A\}\_g \$ & <Generator> & </eog> & x, \$ \textbackslash{}bar\{d\}/x \$ & y \\
\hline
\end{tabular}
```

end token. Note that the existing NLP datasets do not fulfill our requirements for intent reconstructing, we employ the methodology utilized in the long-trajectory subset collection. Table 1 exhibits the inputs and outputs of each short trajectory under the responsibility of each agent. In addition, the response generator contains two types of inputs to help adapt its knowledge preferences. We construct a total of 359,791 instances.

To summarize. Two keys are in the construction: the Long-trajectory subset is crafted to emphasize synergy, and the Short-trajectory subset can be easily accessed in large quantities to emphasize uniqueness. Refer to Appendix Sec.A for the detail of data construction.

## Long-Short Trajectory Learning

Effectively fine-tuning a trajectory system consisting of multi-agents is a complex task: on the one hand, each agent has its specific trajectory signals of attention. On the other hand, the transformation between different trajectories requires the collaboration of the agents. In addition, the cost of trajectory data construction for a multi-agent framework greatly hinders the development of such systems. To this end, we propose Long-Short Trajectory Learning for our multi-agent framework, which consists of two stages, Short Trajectory and Long Trajectory Learning. As shown in Figure 3, Under the guidance of the trajectory head-end token pairs, the intuition is that Short Trajectory Learning first delineates the responsibilities of each agent to develop their unique capabilities, and then Long Trajectory Learning learns the interactions between them. This can be understood as initially activating each agent that masters short trajectories within a broader trajectory framework, and then exploring the interconnections between those agents to navigate the full long trajectory.

Short Trajectory Learning. Short Trajectory Learning is the training of individual capabilities for a single agent. In the context of a long trajectory, it is important to note that short trajectories spanning multiple steps do not necessarily exhibit a strong dependence on preceding short trajectories. To illustrate this point, consider the case of a fact locator, which primarily relies on the original user query and the retrieved results, rather than having a strict dependence on the queries generated in Intent Reconstructor. Similarly, the Response Generator necessitates only the question itself or a combination of the question and the located facts. As shown in Figure 3, the short trajectory learning first activates each short agent in the framework to focus on the fine-grained signals. Given the short-trajectory subset $ \mathcal{D}_{\text{short}} = \{\mathcal{D}_{\text{intent}}, \mathcal{D}_{\text{locator}}, \mathcal{D}_{\text{generator}}\} $, we initialize a pre-trained LLM and train it on $ \mathcal{D}_{\text{short}} $. For each example $ \{(x_i; h_i), (y_i; e_i)\} \subset \mathcal{D}_{\text{short}} $, we use a standard conditional language modeling objective, maximizing likelihood:

$$ \mathcal{L}\left(\mathcal{D}_{s h o r t}\right)=\sum_{i}\log P_{L M}\left(y_{i};e_{i}\mid x_{i};h_{i}\right), $$

Given the inputs and trajectory header, the agent learns to predict the outputs, i.e., delineate different belonging trajectories for the agent to make them understand the fine-grained

[Table 2 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 2: Comparison results against knowledge internalization and knowledge enhancement methods. * denotes the method we reproduce based on the same base. * denotes re-implemented methods based on the same initial model. The $ \underline{\text{bold}} $ numbers represent the best results and the $ \underline{\text{underlined}} $ numbers represent the second.

Summary: Table 2 compares knowledge internalization and knowledge enhancement methods across multiple tasks (Health, ARC-C, PopQA, Squad1, ASQA) using metrics Acc, Str_EM, R-L, and Mauve. It reports the performance of various baselines and the proposed SMART method, with best results bold and second underlined. SMART achieves the highest scores on Health Acc (73.18), PopQA Acc (42.60), ASQA Str_EM (41.16), ASQA R-L (40.66), and ASQA Mauve (91.47) among all methods, but is not the best on ARC-C or Squad1.

Table LaTeX:

```latex
\begin{tabular}{llllllll}
\hline
Task & Health & ARC-C & PopQA & Squad1 & ASQA & ASQA & ASQA \\
\hline
Metric & Acc & Acc & Acc & Acc & Str\_EM & R-L & Mauve \\
Knowledge internalization methods & Knowledge internalization methods & Knowledge internalization methods & Knowledge internalization methods & Knowledge internalization methods & Knowledge internalization methods & Knowledge internalization methods & Knowledge internalization methods \\
Alpaca2 \$ \_\{7B*\} \$ & 44.78 & 36.43 & 25.58 & 11.50 & 14.42 & 28.72 & 51.24 \\
Mistral-Instruct \$ \_\{7B\} \$ & 65.45 & 57.84 & 22.37 & 14.97 & 20.80 & 32.20 & 33.47 \\
Llama-2-Chat \$ \_\{7B\} \$ & 47.95 & 47.95 & 25.44 & 14.13 & 16.79 & 32.35 & 24.21 \\
Vicuna-v1.5 \$ \_\{13B\} \$ & 63.01 & 57.59 & 17.94 & 15.25 & 31.95 & 22.99 & 68.41 \\
Llama-2-Chat \$ \_\{13B\} \$ & 62.20 & 48.72 & 21.22 & 15.97 & 19.97 & 30.37 & 40.23 \\
ChatGPT & 76.08 & 77.3 & 29.30 & 22.90 & 39.94 & 35.73 & 44.63 \\
Knowledge enhancement methods & Knowledge enhancement methods & Knowledge enhancement methods & Knowledge enhancement methods & Knowledge enhancement methods & Knowledge enhancement methods & Knowledge enhancement methods & Knowledge enhancement methods \\
Alpaca2 \$ \_\{7B*\} \$ & 26.44 & 35.15 & 33.38 & 21.41 & 23.59 & 27.21 & 50.09 \\
REPLUG \$ \_\{7B*\} \$ & 41.72 & 47.26 & 37.24 & 24.23 & 26.54 & 33.25 & 54.03 \\
VANILLA \$ \_\{7B*\} \$ & 29.52 & 42.74 & 37.52 & 25.92 & 32.25 & 34.93 & 39.54 \\
RAIT \$ \_\{7B*\} \$ & 52.98 & 62.10 & 38.02 & 23.86 & 25.68 & 15.99 & 12.35 \\
INTERACT \$ \_\{7B*\} \$ & 65.45 & 48.12 & 41.31 & 31.52 & 34.54 & 35.51 & 43.45 \\
SelfRag \$ \_\{7B\} \$ & 68.99 & 65.52 & 40.67 & 22.39 & 28.68 & 34.11 & 83.00 \\
MMAgent \$ \_\{3*7B*\} \$ & 70.82 & 63.99 & 36.88 & 23.79 & 33.04 & 36.49 & 88.98 \\
SMART (OURS) & 73.18 & 65.58 & 42.60 & 27.80 & 41.16 & 40.66 & 91.47 \\
\hline
\end{tabular}
```

representations of the corresponding tasks. This phase utilizes easily accessible and extensive data to build the basic capabilities of the trajectory, reducing the cost of such a framework while maintaining the creativity and versatility of the agent.

Long Trajectory Learning. After the above stage, the framework is equipped with four independent agents. Long Trajectory Learning further grooms the LLM to establish logical associations between agents in an end-to-end manner. We train based on the previous stage on the long-trajectory subset $ D_{long} $. Specifically, given instruction x, long trajectory learning forces the LLM to learn the long trajectory process:

$$ \begin{align*}\mathcal{L}\left(\mathcal{D}_{Long}\right)&=\sum_{i}\log P_{LM}\left(\tau_{i}^{R};\tau_{i}^{I};\tau_{i}^{G}\mid x_{i}\right),\\\tau_{i}^{T}&=\left[h_{i}^{T};y_{i}^{T};e_{i}^{T}\right],T\subset\left\{R,I,G\right\}.\end{align*} $$

where R, I and G denote the Intent Reconstructor, Fact Locator and Response Generator, respectively. Unlike short trajectory learning (Eq. 2), the framework learns both to predict the target output for each short trajectory as well as from the previous trajectory end $ e^{T} $ to the next trajectory head $ h^{T+1} $. In essence, the trajectory token serves as a skeleton in the learning process, guiding the agent not only to grasp a fine-grained representation of the intra-trajectory but also inter-trajectory interactions.

## Experiment Setting

## Setup

Task and Dataset. We evaluate our framework in a range of knowledge-intensive downstream tasks. Including (1) Fact verification: PubHealth (Akhtar, Cocarascu, and Simperl 2022) is a fact verification dataset about public health; (2) Multiple-choice reasoning: ARC-Challenge (Clark et al. 2018) is a multiple-choice question dataset about science exam. (3) Open-domain question answering: contains two short-form QA datasets, PopQA (Mallen et al. 2022), and SQuAD 1.1 (Rajpurkar et al. 2016). (4) Ambiguous question answering: ASQA (Gao et al. 2023) is ambiguous fact-to-data question of the long form response. Details of evaluation data, including size, and evaluation metrics are available in Appendix Sec. B.1.

Baselines. We compare our framework with a wide range of baseline methods in two categories. (1) Knowledge internalization methods (General-purpose LLMs): ChatGPT (gpt-3.5-turbo-0125) (Zheng et al. 2023) (Ouyang et al. 2022), Mistral-Instruct-v0.2-7B (Jiang et al. 2023), Llama2-Chat-7B/13B (Touvron et al. 2023), Vicuna-v1.5-13B (Zheng et al. 2023) and Alpaca2-7B (Zheng et al. 2023). (2) Knowledge enhancement methods: REPLUG-7 (Shi et al. 2023), VANILLA-7B (Gao et al. 2023), INTERACT-7B (Gao et al. 2023), RAIT-7B (Lin et al. 2023), SelfRAG-7B (Asai et al. 2023), MMAgent-3*7B (modular approach). More details are in Appendix Sec. B.2.

## Implementation Details

Due to page limitations, details of our training and evaluation are in Appendix Sec. B.3.

## Main Result

## Experiment Result

Comparison against knowledge internalization methods. As shown in Table 2, our framework shows a significant performance advantage over equivalently sized fine-tuned LLMs across all tasks. In comparison to larger LLMs (Vicuna-v1.5-13B and Llama-2-Chat-13B), which possess greater internalized knowledge, our SMART framework also exhibits superior performance in all metrics. Furthermore,

[Table 3 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 3: Training Ablation and inference ablation for the contribution of different agents. L and S denote long-trajectory and short-trajectory learning, respectively. w/o $ \mathcal{A}_{f} $, w/o $ \mathcal{A}_{i} $, and w/o All denote no fact Locator, no intent reconstructor, and only response generator.

Summary: This table presents training and inference ablation results for the SMART model on four metrics (Health, ARC-C, Pop, AS), comparing the full model against variants without the fact locator (w/o A_f), without the intent reconstructor (w/o A_i), and without both (w/o All). Training ablation uses only long-trajectory learning (L), while inference ablation uses both long and short trajectories (L+S). The results show that each component contributes to performance, with the full model achieving the highest scores in both settings.

Table LaTeX:

```latex
\begin{tabular}{lllll}
\hline
 & Health (Acc) & ARC-C (Acc) & Pop (Acc) & AS (Em) \\
\hline
Training ablation & Training ablation & Training ablation & Training ablation & Training ablation \\
SMART (L) & 72.15 & 60.22 & 37.27 & 36.10 \\
w/o \$ \textbackslash{}mathcal\{A\}\_\{\textbackslash{}mathrm\{f\}\} \$ & 70.13 & 58.95 & 34.31 & 34.77 \\
w/o \$ \textbackslash{}mathcal\{A\}\_\{\textbackslash{}mathrm\{i\}\} \$ & 69.82 & 54.94 & 35.17 & 34.41 \\
w/o All & 57.95 & 56.99 & 21.15 & 20.05 \\
Inference ablation & Inference ablation & Inference ablation & Inference ablation & Inference ablation \\
SMART (L+S) & 73.18 & 65.58 & 42.60 & 41.16 \\
w/o \$ \textbackslash{}mathcal\{A\}\_\{\textbackslash{}mathrm\{f\}\} \$ & 71.63 & 62.45 & 37.45 & 36.10 \\
w/o \$ \textbackslash{}mathcal\{A\}\_\{\textbackslash{}mathrm\{i\}\} \$ & 71.22 & 60.11 & 39.88 & 35.30 \\
w/o All & 69.32 & 58.81 & 16.79 & 31.32 \\
\hline
\end{tabular}
```

our framework surpasses ChatGPT in all evaluated metrics for PopQA (long-tail knowledge evaluation), Squad1, and ASQA. Experimental results indicate that our method more effectively addresses long-tail knowledge, delivering more accurate and fluent responses compared to knowledge internalization methods, which necessitate extensive fine-tuning and training on large volumes of private data.

Comparison against knowledge enhancement methods. Considering fairness and persuasiveness, we compared knowledge enhancement methods based on the same size as ours. As shown in Table 2, our SMART performs better on most tasks compared to other knowledge enhancement methods. Compared to the SOTA retrieval method, SelfRag (Asai et al. 2023), our model shows great superiority in both accuracy and fluency. Our method exceeds MMAgent (four independent agents coupled together) in all metrics. This demonstrates that our learning paradigm improves multi-agent collaboration, resulting in more accurate responses. Note that INTERACT (Gao et al. 2023) is better than us on Squad1, the reason is that INTERACT allows the response model to do more reasoning steps, which is beneficial for hitting answers in short-format generation tasks. RAIT (Lin et al. 2023) is trained with SMART same data and initialized model without fact location and intent reconstruction, lagging behind us. Overall, our SMART delivers excellent performance in a diverse range of knowledge-intensive tasks. This result indicates SMART gains are not solely from the multi-agent framework and demonstrate the effectiveness of the long-short trajectory learning.

## Ablation Studies

Training ablation of different agents. Training ablation aims to verify the superiority of the entire multi-agent combination setup. To save the experiment cost, we implement long-trajectory learning using 60,000 samples from the long-trajectory subset to evaluate the performance of the co-framework under different agent absence scenarios.

[Table 4 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 4: Ablation studies of long-trajectory (Long) and short-trajectory (Short) learning.

Summary: Table 4 reports ablation study results comparing four methods (Vanilla LLM, +Short, +Long, +Short & Long) on four metrics: Health accuracy, PopQA accuracy, ASQA exact match, and ASQA ROUGE-L. The results show that the combination of short and long trajectory learning (+Short & Long) achieves the highest scores on all metrics, with values 73.18, 42.60, 41.16, and 40.66 respectively.

Table LaTeX:

```latex
\begin{tabular}{lllll}
\hline
Methods & Health & PopQA & ASQA & ASQA \\
\hline
Methods & (Acc) & (Acc) & (Em) & (R-L) \\
Vanilla LLM & 9.80 & 22.69 & 14.11 & 6.45 \\
+ Short & 62.00 & 32.23 & 23.95 & 19.91 \\
+ Long & 72.9 & 37.66 & 39.86 & 39.51 \\
+ Short \& Long & 73.18 & 42.60 & 41.16 & 40.66 \\
\hline
\end{tabular}
```

As the top part of Table 3, the absence of the fact Locator and the intent reconstructor significantly degrades the framework's performance. The intent reconstructor provides substantial benefits for multiple-choice reasoning (ARC-C) and ambiguous questions (ASQA), while the fact Locator is crucial for long-tail knowledge Q&A (PopQA). The experiment proved the effectiveness of different agents in our SMART, especially the fact Locator and the intent reconstructor.

Inference ablation of different agents. We use the full version of SMART with short long-trajectory learning to ignore the trajectories of different agents during the inference phase. As the bottom part of Table 3, each agent plays an important role in the collaboration framework. The effect degradation of the fact-checking task (Health) was not severe, which may be related to the large amount of knowledge injected during the short trajectory learning. In addition, note that if the inference process is missing a particular agent, most multi-agent frameworks that use end-to-end training become terrible, due to the loss of signals from the missing agent. Benefiting from our Short-Trajectory Learning through the trajectory tokens, our SMART does not collapse in performance when an agent is missing, demonstrating flexibility while maintaining performance.

Effects of Long-Short Trajectory Learning. Long-Short Trajectory Learning optimising a Multi-agent framework through two-stage learning. We demonstrate its effectiveness progressively by training it on vanilla models, Llama2-7B-hf (Touvron et al. 2023). As shown in Table 4, short-trajectory learning and long-trajectory learning enable huge performance improvements in the framework for all tasks. Short-trajectory learning enhances the system by optimizing each agent's base capability, though its impact is not as substantial as that of long-trajectory learning. Long-trajectory learning, by optimizing agent synergy, underscores the importance of collaborative optimization in a multi-agent framework, despite the challenges posed by complex data construction. Overall, the combined approach of long-short trajectory learning yields the best performance, highlighting the significance of simultaneous collaboration and individual uniqueness.

Effects of training data size. To examine the impact of long-trajectory training data on long-short trajectory learning, we randomly selected subsets of 8k, 20k, 60k, and 121k instances from the initial 140k training instances and fine-tuned four SMART variants on these subsets. Subsequently, we compared the model performance on ARC-C, PopQA,

[Figure 4 was here. The original paper contained a figure at this position. Brief visual description: The figure consists of three line charts (likely panels a, b, and c corresponding to tasks ARC-C, PopQA, and ASQA) that illustrate the effect of increasing long-trajectory training data size (K) on model performance. Each chart plots a performance metric (accuracy 'acc' or exact match 'em') on the y-axis against the number of training samples in thousands ('Num of training(k)') on the x-axis. A solid green line with markers represents the primary method's performance at different data sizes (0, 20, 60, 120, 140k), while horizontal dashed lines represent fixed baseline performances for 'SelfRag-7B' (blue) and 'MMAgent' (orange).]
Caption: Figure 4: Effects of long-trajectory training data size (K) on three tasks, ARC-C, PopQA and ASQA.
Key visible elements:
- Panel 1 (Left): Line chart showing accuracy vs training data size, likely for task ARC-C
- Panel 2 (Middle): Line chart showing accuracy vs training data size, likely for task PopQA
- Panel 3 (Right): Line chart showing exact match (em) vs training data size, likely for task ASQA
- Green Line Series: Represents the performance of the proposed method across varying data sizes
- Blue Dashed Line: Baseline performance reference for SelfRag-7B
- Orange Dashed Line: Baseline performance reference for MMAgent
- X-Axis Label: Indicates 'Num of training(k)', ranging from 0 to 140
- Y-Axis Labels: Indicates metrics 'acc' (Panels 1 & 2) and 'em' (Panel 3)

and ASQA with our SelfRAG and MMagent models. As shown in Figure 4, an increase in data size generally leads to improved performance across all datasets. Notably, by utilizing 60k data instances, SMART outperformed SelfRAG, which employs 120k samples. This demonstrates the significant advantage of our learning approach in markedly enhancing the performance of multi-agent framework.

## Related Work

Trajectory Learning. Trajectory learning aims to allow agent systems to complete a complex task or scenario through a series of interconnected phases, which requires a profound understanding of both global and local dimensions. Some methods (Chen et al. 2023; Song et al. 2023; Kong et al. 2023; Asai et al. 2023; Sun et al. 2022; Mou, Wei, and Huang 2024) enable agent learning trajectory via providing crafted prompt or tuning, which may not consistently yield high performance in every phase. Moreover, independently modules (Liu et al. 2023; Shen et al. 2024; Ma et al. 2023; Xu, Shi, and Choi 2023; Wang et al. 2023) can be combined with agent to implement trajectory inference, while this integration confers robust isolated capabilities, the gap between modules might lead to cumulative errors throughout the trajectory process. In this paper, we introduce long-short trajectory learning, which equips multiagent systems with the ability to not only grasp the logic connecting steps but also to refine each step. Our approach is scalable to increasingly complex scenarios.

Knowledge Enhancement Methods. Ensuring fact-consistent responses is a core goal of intelligent systems research (Wang et al. 2022b; Tu et al. 2024b,a, 2023; Yue et al. 2024, 2023b; Gao et al. 2024). LLMs parameterize knowledge by training on gargantuan textual corpora. However, LLMs suffer from hallucination (Ji et al. 2023), trouble in acquiring long-tailed fact (Kandpal et al. 2023) and struggle to expand their parametric knowledge. For knowledge-intensive scenarios, existing methods (Izacard et al. 2023; Sun et al. 2020) usually assist LLMs by integrating non-parametric knowledge. Recent advances incorporated retrievers (Asai et al. 2023; Shi et al. 2023; Lin et al. 2023) to augment LLMs. The efficacy of non-parametric knowledge collaboration in improving task performance significantly relies on the relevance of the acquired knowledge and the level of knowledge utilization by the LLM itself. However, existing work has not comprehensively confronted these challenges. Some works (Xu, Shi, and Choi 2023; Ma et al. 2023) simply select relevant knowledge and demonstrate better intentions by combining separate modules. Self-RAG (Asai et al. 2023) integrates specialized feedback tokens into the language model to assess the necessity for retrieval and to verify the relevance, support, or completeness of the output. Unlike existing approaches, we introduce a novel multi-agent framework that addresses these challenges with trajectory learning.

## Conclusions

In this paper, we introduce SMART, a novel multi-agent framework that addresses the challenges of generating factually consistent responses in knowledge-intensive tasks. By leveraging external knowledge and employing specialized agents, SMART enhances the interpretability and factual consistency of LLMs generated responses. Our proposed Long- and Short-Trajectory Learning paradigm ensures synergistic collaboration among agents while maintaining fine-grained execution, enabling the framework to navigate complex knowledge-intensive tasks effectively. Empirical results on five diverse tasks demonstrate SMART's superior performance compared to SOTA pre-trained and instruction-tuned LLMs, as well as widely adopted methods. SMART highlights the importance of integrating external knowledge and employing multi-agent systems to tackle the limitations of LLMs in knowledge-intensive scenarios.

Future work. One is that our framework currently executes sequentially without iterative optimization, which may lead to insufficient knowledge retrieval for multi-hop problems. However, this can be addressed by adding loop arrows between the Fact Locator and Intent Reconstructor agents. Another is that our retriever is not trained in the whole process, although it can be incorporated into the training process using existing techniques. We envision our framework as a general paradigm that extends beyond knowledge-intensive tasks to more complex scenarios, enabling any multi-agent framework to internalize tailored trajectories.