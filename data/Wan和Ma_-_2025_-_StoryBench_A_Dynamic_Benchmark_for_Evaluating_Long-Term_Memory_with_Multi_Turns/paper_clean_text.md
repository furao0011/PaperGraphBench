<!-- page 1: doc_0.md -->

# StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns
## Abstract

Long-term memory (LTM) is essential for large language models (LLMs) to achieve autonomous intelligence in complex, evolving environments. Despite increasing efforts in memory-augmented and retrieval-based architectures, there remains a lack of standardized benchmarks to systematically evaluate LLMs' long-term memory abilities. Existing benchmarks still face challenges in evaluating knowledge retention and dynamic sequential reasoning, and in their own flexibility, all of which limit their effectiveness in assessing models' LTM capabilities. To address these gaps, we propose a novel benchmark framework based on interactive fiction games, featuring dynamically branching storylines with complex reasoning structures. These structures simulate real-world scenarios by requiring LLMs to navigate hierarchical decision trees, where each choice triggers cascading dependencies across multi-turn interactions. Our benchmark emphasizes two distinct settings to test reasoning complexity: one with immediate feedback upon incorrect decisions, and the other requiring models to independently trace back and revise earlier choices after failure. As part of this benchmark, we also construct a new dataset designed to test LLMs' LTM within narrative-driven environments. We further validate the effectiveness of our approach through detailed experiments. Experimental results demonstrate the benchmark's ability to robustly and reliably assess LTM in LLMs.

## 1 Introduction

In the field of artificial intelligence, the pursuit of true intelligence in large language models (LLMs) has prompted researchers to look to biology for inspiration [Gutiérrez et al., 2024, Wu et al., 2025]. Just as organisms gradually accumulate knowledge through experience over time, LLMs need to possess long-term memory (LTM) capabilities to achieve self-evolution and strategic optimization in ever-changing environments [Shan et al., 2025]. Moreover, as LLMs are increasingly applied in scenarios such as multi-session dialogue [Zhang et al., 2025], task planning, and lifelong learning, the need for models to retain, update, and leverage prior knowledge dynamically becomes critical. Without robust LTM, AI systems are limited to short-term reasoning and static knowledge use, failing to achieve sustained, autonomous intelligence.

Given the importance of LTM in enabling advanced behaviors, it is crucial to evaluate these capabilities reliably and systematically. However, current benchmarks face challenges in adequately

<!-- page 2: doc_1.md -->

evaluating LTM capabilities in two critical dimensions: 1) Knowledge Retention: the capacity to absorb, integrate, and preserve information across extended texts, maintaining contextual continuity beyond mere fact retrieval or local recall [Guo et al., 2025, EducateMe, 2024]; and 2) Sequential Reasoning: the ability to understand and reason about sequences of events, which involves inferring latent state changes, causal dependencies, and goal shifts across complex, dynamic, and multi-turn interactions rather than simply locating pre-stated answers within static text. 3) Flexibility: previous benchmarks often face challenges in adjusting and evaluating in different contexts.

To address these limitations, we propose a dynamic benchmark framework inspired by interactive fiction games, where LLMs engage in branching narratives with multi-turns that simulate long-term sequential decision-making. In our benchmark, the model continuously receives scene descriptions, dialogues, and options, and must make choices based on its understanding. We design two modes: Immediate Feedback provides immediate feedback when the model makes a wrong choice, while Self Recovery allows the story to continue toward a failure ending without any hint, requiring the model to identify and revise past decisions on its own. Through this setup, our benchmark effectively evaluates the model's ability to remember key information (knowledge retention) and reason over event sequences (sequential reasoning). Furthermore, our benchmark demonstrates excellent flexibility in accommodating diverse scenarios.

To further illustrate the advantages of our benchmark, we comprehensively evaluate the differences between existing benchmarks and ours (Table 1) based on the following aspects:

Knowledge Retention. Long-context (L-ctx) evaluates whether the task requires long-term memory of earlier context to succeed. Continuity (Conty) measures whether the benchmark requires the model to maintain a coherent understanding of entities, events, and their relationships across interactions.

Sequential Reasoning. Complexity (Comp.) indicates whether the benchmark features nonlinear reasoning tasks, where multiple interdependent events or decisions must be jointly considered, requiring the model to reason beyond sequential context. Dynamics (Dyn.) refers to whether the model's actions or responses influence future tasks or states in the environment. Multi-turn (M-turn) evaluates whether the task involves multiple sequential interactions, where each turn is temporally connected to the previous ones.

Flexibility. Multi-solution (M-sol) indicates whether the benchmark includes tasks or questions with multiple valid answers or approaches, rather than a single fixed solution. LTM+STM evaluates the combined usage of long-term memory (LTM) and short-term memory (STM), i.e., whether the task requires reasoning over both recent and distant information.

 

 

To validate the effectiveness of our benchmark, we conduct systematic evaluations on advanced four LLMs. Each model is tested under both evaluation modes across 80+ branching story paths, with performance measured in terms of correct decision rates, task success counts, etc. Results show that while GPT-4o [OpenAI, 2024] and Claude 3.5 Sonnet [Anthropic, 2024] demonstrate relatively stronger long-term knowledge retention and sequential reasoning, all models struggle with self-recovery and fail to consistently revise earlier mistakes. In-depth failure analysis further reveals distinct memory bottlenecks, which existing benchmarks could be enhanced to expose. These findings confirm the utility of StoryBench in capturing LTM deficiencies, offering a more granular and realistic assessment than prior benchmarks.

Our contributions are as follows:

<!-- page 3: doc_10.md -->

#### 5.2.2 Insights of Distinctions Between Two Modes

To investigate how short-term and long-term memory settings affect model behavior, we compare performance under two task modes. Immediate Feedback mode provides corrective signals after each wrong choice, effectively mimicking short-term memory and aiding models in adjusting quickly. In contrast, Self Recovery better simulates real long-term memory scenarios by removing such signals, requiring the model to navigate the narrative without external guidance.

Unsurprisingly, all models perform worse under Self Recovery mode, as shown by the consistent drop in Overall Accuracy and Success Count. This highlights the increased difficulty of sustained sequential reasoning and knowledge retention without short-term feedback. To alleviate task failure in extreme cases, we introduce an auxiliary intervention metric: Number of Choices Reaching Error Threshold (we set the threshold to 9). If a model makes the same mistake more than 9 times, it is prompted with the correct answer. Only Claude 3.5 and GPT-4o never reach this threshold, suggesting that their task completions in Self Recovery mode are entirely due to self-correction and internal reasoning without any artificial hints. This contrasts sharply with other models, indicating that they excel in sustained sequential reasoning and knowledge retention.

Surprisingly, despite the overall decline in performance across models in Self Recovery, two metrics: Longest Consecutive Correct Sequence and First-Try Accuracy actually increase for several models (Figure 8). This amazing trend emphasizes that while short-term feedback aids local correction, it may also disrupt long-horizon coherence. By removing it, models foster a deeper narrative understanding (knowledge retention) and more coherent reasoning (sequential reasoning) and we better expose the true limitations and strengths of long-term memory in different models.

<!-- page 4: doc_11.md -->

A notable case is Deepseek-R1. While it does not lead in most individual metrics, it demonstrates remarkable consistency across both Immediate Feedback and Self Recovery modes. This stable performance suggests that the model is capable of making accurate revisions during backtracking.

### 5.3 Failure Case Study

In evaluating long-term memory capabilities with StoryBench, we identified two principal types of failure that reflect limitations in current language models, corresponding to the core dimensions of knowledge retention and sequential reasoning.

The most prominent issue in knowledge retention was the failure to preserve contextual consistency over extended narratives. Models frequently made decisions that contradicted earlier story events, character motivations, or established world logic. This suggests difficulty in integrating and maintaining distributed information over long spans of interaction, especially when the necessary context spans dozens of turns. Even when the relevant facts appeared in the prompt, models struggled to apply them coherently, indicating limitations beyond simple factual recall.

In terms of sequential reasoning, a critical failure case was the inability to repair long-term or multi-error decisions. In Self Recovery mode, successful completion often required models to trace errors back across multi-step causal chains and revise earlier decisions (even multiple choices in combination) that affected downstream outcomes. However, most models exhibited shallow search strategies, typically backtracking only one or two steps rather than engaging in deeper reasoning about the narrative structure or goal shifts. This myopic behavior led to persistent failure when task success depended on understanding and correcting long-term dependencies. We retained such failures to reflect the true upper-bound difficulty of long-term memory reasoning.

Other failures such as format mismatches (e.g., returning option indices instead of decision point IDs), content filtering blocks, server timeouts, or rare instances of hallucinated explanations were also observed but were comparatively infrequent. These were retained in evaluation for completeness but are not the focus of our analysis.

These diverse failure cases underscore the challenge of StoryBench and emphasize the need for more robust memory integration, format alignment, and long-range error correction in current foundation models.

## 6 Limitations

While our benchmark provides a comprehensive evaluation of long-term memory capabilities in large language models through complex, branching narrative tasks, it has several limitations. First, the scenarios are derived from a single interactive fiction domain and the interactive environment is

<!-- page 5: doc_12.md -->

text-based, both of which may limit the benchmark's generalizability to other knowledge-intensive or task-oriented contexts that require multimodal support. Second, the number of turns and the length of the context are still limited. The current interactive fiction dataset consists of only 6 chapters, which may not fully capture the long-term dependencies and complex reasoning required in more extensive narratives. Future work could expand the dataset by adding subsequent chapters to provide a more comprehensive evaluation of long-term memory. Third, due to API constraints and cost, we primarily evaluate a limited number of mainstream models. The performance of other models under similar conditions remains unexplored. Fourth, although we include a self-recovery setting to simulate real-world error correction, the evaluation remains scripted and cannot capture all forms of natural feedback.

## 7 Conclusions

We introduce StoryBench, a novel benchmark designed to systematically evaluate long-term memory capabilities in complex, dynamic, and multi-turn narrative environments. By simulating realistic memory demands across story understanding, sequential inference, and flexible correction, our benchmark assesses current mainstream models in knowledge retention and sequential reasoning. Through comprehensive experiments on representative models and detailed failure case analyses, we demonstrate that current models exhibit significant performance gaps on our benchmark, highlighting StoryBench's difficulty and effectiveness in evaluating long-term memory capabilities. Our findings underscore the importance of developing more robust memory mechanisms, laying the groundwork for future research toward memory-augmented, context-aware language agents.

<!-- page 6: doc_13.md -->

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv e-prints, pages arXiv–1904, 2019.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.

Tri Dao, Daniel Y Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In Proceedings of the 35th Neural Information Processing Systems Conference (NeurIPS), 2022.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guantong Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zhihu Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv e-prints, art. arXiv:2501.12948, January 2025. doi: 10.48550/arXiv.2501.12948.

Yu Ding, Cong Sheng, Zhongyue Chen, Lingli Mao, Zhao Peng, Tingting Chen, Yiqun Liu, and Wanli Huo. Cervical cancer segmentation based on full-scale feature fusion with cascading-attention and dilated convolution. In Proc. of SPIE Vol, volume 12800, pages 1280033–1, 2023.

Zican Dong, Tianyi Tang, Junyi Li, Wayne Xin Zhao, and Ji-Rong Wen. BAMBOO: A comprehensive benchmark for evaluating long text modeling capacities of large language models. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, editors, Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 2086–2099, Torino, Italia, May 2024. ELRA and ICCL. URL https://aclanthology.org/2024.lrec-main.188/.

EducateMe. Knowledge retention: 8 main strategies to improve it, February 2024. URL https://www.educate-me.co/blog/knowledge-retention-strategies.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=tEYskw1VY2.

Linqiang Guo, Wei Liu, Yi Wen Heng, Tse-Hsun, Chen, and Yang Wang. MAPLE: A Mobile Agent with Persistent Finite State Machines for Structured Task Reasoning. arXiv e-prints, art. arXiv:2505.23596, May 2025. doi: 10.48550/arXiv.2505.23596.

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically inspired long-term memory for large language models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 59532–59569. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf.

Yi Han, Hang Chen, Jun Du, Chang-Qing Kong, Shi-fu Xiong, and Jia Pan. Layer-adaptive low-rank adaptation of large asr model for low-resource multilingual scenarios. In 2024 IEEE 14th International Symposium on Chinese Spoken Language Processing (ISCSLP), pages 696–700. IEEE, 2024.

<!-- page 7: doc_14.md -->

Shuang Hao, Wenfeng Han, Tao Jiang, Yiping Li, Haonan Wu, Chunlin Zhong, Zhangjun Zhou, and He Tang. Synthetic Data in AI: Challenges, Applications, and Ethical Implications. arXiv e-prints, art. arXiv:2401.01629, January 2024. doi: 10.48550/arXiv.2401.01629.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What's the real context size of your long-context language models? CoRR, 2024.

Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In International Conference on Learning Representations, 2020.

Yury Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, and Mikhail Burtsev. Babilong: Testing the limits of llms with long context reasoning-in-a-haystack. Advances in Neural Information Processing Systems, 37:106519–106554, 2024.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with paged attention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023.

Kuang-Huei Lee, Xinyun Chen, Hiroki Furuta, John Canny, and Ian Fischer. A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts. arXiv e-prints, art. arXiv:2402.09727, feb 2024. doi:10.48550/arXiv.2402.09727.

Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. Loogle: Can long-context language models understand long contexts? arXiv e-prints, pages arXiv–2311, 2023.

Hao Liu, Matei Zaharia, and Pieter Abbeel. Ringattention with blockwise transformers for near-infinite context. In The Twelfth International Conference on Learning Representations, 2023.

Ruibo Liu, Jerry Wei, Fangyu Liu, Chenglei Si, Yanzhe Zhang, Jinmeng Rao, Steven Zheng, Daiyi Peng, Diyi Yang, Denny Zhou, and Andrew M. Dai. Best Practices and Lessons Learned on Synthetic Data. arXiv e-prints, art. arXiv:2404.07503, April 2024. doi: 10.48550/arXiv.2404.07503.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. In ICLR, 2024.

OpenAI. Hello gpt-4o, 2024. URL https://openai.com/index/hello-gpt-4o/.

Charles Packer, Vivian Fang, Shishir_G Patil, Kevin Lin, Sarah Wooders, and Joseph_E Gonzalez. Memgpt: Towards llms as operating systems. arXiv, 2023.

Matteo Pagliardini, Daniele Paliotta, Martin Jaggi, and François Fleuret. Faster causal attention over large sequences through sparse flash attention. arXiv e-prints, pages arXiv–2306, 2023.

Bo Peng, Eric Alcaide, Quentin Gregory Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Nguyen Chung, Leon Deczynski, et al. Rwkv: Reinventing rnns for the transformer era. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

Jack Rae and Ali Razavi. Do transformers need deep long-range memory? In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7524–7529. Association for Computational Linguistics, 2020.

Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. Zerosέrolls: A zero-shot benchmark for long text understanding. In 2023 Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7977–7989. Association for Computational Linguistics (ACL), 2023.

Lianlei Shan, Shixian Luo, Zexhou Zhu, Yu Yuan, and Yong Wu. Cognitive Memory in Large Language Models. arXiv e-prints, art. arXiv:2504.02441, April 2025. doi: 10.48550/arXiv.2504.02441.

Simeng Sun, Katherine Thai, and Mohit Iyyer. Chapterbreak: A challenge dataset for long-range language models. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3704–3714, 2022.

Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming Tang, and Yong Liu. From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs. arXiv e-prints, art. arXiv:2504.15965, April 2025. doi: 10.48550/arXiv.2504.15965.

<!-- page 8: doc_15.md -->

Tao Xie, David Harel, Dezhi Ran, Zhenwen Li, Maoliang Li, Zhi Yang, Leye Wang, Xiang Chen, Ying Zhang, Wentao Zhang, Meng Li, Chen Zhang, Linyi Li, and Assaf Marron. Data and System Perspectives of Sustainable Artificial Intelligence. arXiv e-prints, art. arXiv:2501.07487, January 2025. doi: 10.48550/arXiv.2501.07487.

Xinchao Xu, Zhibin Gou, Wenquan Wu, Zheng-Yu Niu, Hua Wu, Haifeng Wang, and Shihang Wang. Long time no see! open-domain conversation with long-term persona memory. Findings of the Association for Computational Linguistics: ACL 2022, 2022.

Chen Zhang, Xinyi Dai, Yaxiong Wu, Qu Yang, Yasheng Wang, Ruiming Tang, and Yong Liu. A survey on multi-turn interaction capabilities of large language models. arXiv e-prints, pages arXiv–2501, 2025.

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, et al. bench: Extending long context evaluation beyond 100k tokens. CoRR, 2024.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. CoRR, 2023.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations, 2023.

<!-- page 9: doc_2.md -->

• A Dynamic Multi-turn Evaluation Framework: We introduce a novel dynamic multi-turn benchmark inspired by interactive fiction. Through branching narratives and two distinct modes (Immediate Feedback, Self Recovery), it assesses models' knowledge retention and sequential reasoning, while offering high flexibility across various scenarios.

- A Novel Dataset for Long-Term Memory Evaluation: We construct an annotated interactive fiction-based dataset to test LTM. It features cohesive narrative continuity, dynamic branching, complex interdependencies, and multi-solution mechanisms to emulate real-world memory challenges.

• Reliable and Robust Experimental Analysis: To ensure the credibility of our findings, we perform repeated trials, enhancing statistical robustness and supporting meaningful performance comparisons.

## 2 Related Work

### 2.1 Strategies and Techniques for Enhancing Long-Term Memory

Transformer models face inherent limitations in processing long sequences due to the quadratic complexity of self-attention mechanisms. To address these challenges, various architectural innovations have been proposed, including sparse attention mechanisms like Reformer [Kitaev et al., 2020], Longformer [Beltagy et al., 2020], Sparse Transformer [Child et al., 2019], and Sparse Flash Attention [Pagliardini et al., 2023], which reduce the number of token pairs in attention computation to improve speed and memory usage. Enhancements such as dilated convolution and cascading attention [Ding et al., 2023], sparse attention and HSR data structures [Chen et al., 2024], and Ring Attention [Liu et al., 2023] aid in handling long-range dependencies, while performance optimizations like FlashAttention [Dao et al., 2022] and PagedAttention [Kwon et al., 2023] further enhance efficiency through techniques like tiling, paging and flexible KV cache sharing. Context expansion techniques via recurrence, such as Transformer-XL [Dai et al., 2019], enable the retention and reuse of longer context windows and optimizations in Transformer-XL, such as reducing the number of long-range memories and limiting attention range in lower layers [Rae and Razavi, 2020], can achieve comparable or better performance. New architectures like Mamba [Gu and Dao, 2024] and RWKV [Peng et al., 2023] explore alternatives to traditional attention mechanisms. Beyond these architectural improvements, recent research has also focused on explicit memory mechanisms to address the limitations of fixed-size context windows and improve information retention. Memory modules like MemoryBank [Zhong et al., 2023] and Retrieval-Augmented Generation (RAG) leverage external storage and dynamic retrieval to enhance long-term memory and knowledge utilization in language models. Parameter-efficient fine-tuning techniques such as LoRA [Han et al., 2024] embed task-specific adjustments via low-rank matrices, preserving information without altering the full model architecture. Hybrid memory systems like MemGPT [Packer et al., 2023], Mem0 [Chhikara et al., 2025], and MemoryScope [Yu et al., 2024] integrate various memory modules and interfaces to enhance long-term retention and retrieval, while recursive approaches like Gist Memory [Lee et al., 2024] compress and retain key context fragments. These advancements collectively address both retention and adaptive management of information, enabling more effective long-term memory capabilities in language models.

### 2.2 Benchmarks for Evaluating Long-Term Memory

Recent evaluations of large language models' (LLMs) long-context and long-term memory capabilities have primarily relied on dedicated benchmarks. Existing benchmarks use prefilled contexts of varying lengths, such as up to 10k tokens in LongBench [Bai et al., 2024], $ \approx $24k in LooGLE [Li et al., 2023], and up to 100k in InfiniteBench [Zhang et al., 2024] and even longer contexts (10 million tokens or even more) in BabiLong [Kuratov et al., 2024]. Benchmarks like ZeroSCROLLS [Shaham et al., 2023], L-Eval [An et al., 2024], and LongBench diversify task types and cover various domains and sequence lengths. ZeroSCROLLS focuses on zero-shot evaluation, L-Eval provides diverse long-document tasks, and LongBench spans six major task categories across multiple languages. Synthetic retrieval-focused setups like Needle-in-a-Haystack [Kamradt, 2023] are popular for their controllability, but concerns remain about their ecological validity due to overly repetitive haystacks. Other benchmarks like RULER [Hsieh et al., 2024] and BAMBOO [Dong et al., 2024] assess reasoning under long contexts. For a holistic understanding of long texts, ChapterBreak [Sun et al., 2022]

<!-- page 10: doc_3.md -->

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

<!-- page 11: doc_4.md -->

### 3.3 Two Task Modes for Evaluating LTM

To explore different aspects of memory utilization, we design two complementary task modes. The dual-mode setup allows StoryBench to probe both short-horizon reactive memory and long-horizon strategic recall (LTM), offering a comprehensive view of how models navigate extended, decision-heavy interactions and revealing not just whether a model can remember facts, but whether it can strategically reason across time, self-correct, and navigate branching storylines over extended sequences.

Immediate Feedback: Designed to evaluate a model’s responsiveness to error signals, this mode simulates situations where feedback is available at each turn. After a wrong choice, the model is told the outcome and prompted to retry (Figure 1), allowing us to examine its short-term adjustment ability and interactive learning dynamics.

 

 

Self Recovery: This mode suppresses feedback, mimicking scenarios where incorrect decisions propagate through multiple scenes, potentially ending the game. The model is then challenged to trace back to the error's origin and recover (Figure 2). This stresses long-term causal reasoning and memory retention under uncertainty.

 

 

### 3.4 Tailored Metrics for Assessing LTM Models

To comprehensively evaluate long-term memory (LTM) capabilities in language models, StoryBench introduces a set of targeted metrics covering two essential cognitive dimensions: knowledge retention and sequential reasoning.

We define a decision sequence $\{c_1, c_2, ..., c_T\}$, where $c_t \in \{0, 1\}$ denotes whether the model selected the correct option (1) or not (0) at step $t$.

<!-- page 12: doc_5.md -->

#### 3.4.1 Metrics for Knowledge Retention

• Overall Accuracy (Overall Acc): The average correctness across all decisions, measuring how consistently the model maintains relevant knowledge and narrative coherence:

 $$ \mathrm{Accuracy}_{\mathrm{overall}}=\frac{1}{T}\sum_{t=1}^{T}c_{t}. $$ 

- First-Try Accuracy (First-Try Acc): The proportion of decision points at which the model selected the correct option on its first attempt. Let $ f_t \in \{0,1\} $ be 1 if the model is correct on the first try at step t, then:

 $$ \mathrm{Accuracy}_{\mathrm{first-try}}=\frac{1}{T}\sum_{t=1}^{T}f_{t}. $$ 

• Longest Consecutive Correct Sequence (Longest Corr): The length of the longest contiguous subsequence of correct decisions:

 $$ \mathrm{Longest Corr}=\max_{1\leq i\leq j\leq T}\left(j-i+1\mid c_{k}=1\forall k\in\left[i,j\right]\right). $$ 

This reflects the model’s ability to sustain contextual consistency over extended intervals, though less critical than the above metrics.

#### 3.4.2 Metrics for Sequential Reasoning

• Accuracy by Difficulty (Easy/Hard Acc): To account for varying levels of memory and reasoning demand, we classify decisions into easy and hard categories. A decision is labeled as hard if it requires recalling information from a distant context, tracking latent state changes, or performing multi-step sequential reasoning; otherwise, it is considered easy. Let $ \mathcal{E}_t $ and $ \mathcal{H}_t $ denote easy and hard decision sets up to step $ t $ (including retries), then:

 $$ \mathrm{A c c u r a c y}_{\mathrm{e a s y}}^{(t)}=\frac{1}{|\mathcal{E}_{t}|}\sum_{i\in\mathcal{E}_{t}}c_{i},\quad\mathrm{A c c u r a c y}_{\mathrm{h a r d}}^{(t)}=\frac{1}{|\mathcal{H}_{t}|}\sum_{i\in\mathcal{H}_{t}}c_{i}. $$ 

These metrics assess how well the model adapts to sequentially distributed and cognitively demanding decisions.

• Retry Count: Let $ r_{t} $ denote the number of retries required before reaching a correct decision at step t. The total number of retries across the trajectory is:

 $$ \mathrm{Retry_{total}}=\sum_{t=1}^{T}r_{t}. $$ 

- Max Error per Choice (Max Err/Choice) and Thresholded Error Count: These metrics capture the worst-case and accumulated difficulty for the model in terms of repeated failures:

 $$ MaxError=\max_{1\leq t\leq T}r_{t},\quad ErrorCount\geq r_{thres}=\sum_{t=1}^{T}\mathbb{I}(r_{t}\geq r_{thres}), $$ 

Where $ \mathbb{I}(\cdot) $ is the indicator function and $ r_{\mathrm{thres}} $ is a predefined retry threshold (e.g., 9 in our experiments).

Finally, while not directly measuring memory accuracy, two auxiliary metrics provide additional perspective on the model's efficiency in handling long-horizon tasks: Runtime Cost reflects the inference efficiency of the memory system, while Token Consumption (Token Cons) indicates the model's reliance on contextual information.

Together, these metrics form a multi-faceted evaluation framework that jointly targets both the persistence of stored information and the model's ability to apply it dynamically within complex, sequentially structured environments. This ensures that memory is not only retained but also meaningfully used to navigate and reason through realistic multi-turn interactions.

<!-- page 13: doc_6.md -->

## 4 Dataset Construction

### 4.1 Overview

To evaluate long-term memory (LTM) capabilities of large language models (LLMs), we construct a narrative dataset based on the interactive fiction game The Invisible Guardian, encompassing 311 scene nodes and 86 choice nodes as captured in our structured JSON format.

We chose to use an interactive fiction game as the basis for our dataset rather than synthetic data or real-world data for several reasons. First, it is arguable that all publicly available benchmark test cases might occasionally be included in LLM pre-training data [Liu et al., 2024]. Consequently, to mitigate potential data overlap issues, we opted to independently construct a dataset of interactive fiction games. Second, synthetic data is often overly simplistic and lacks the nuanced coherence of real human narratives [Hao et al., 2024]. It relies on predefined templates, resulting in repetitive scenarios that fail to capture the complex interdependencies crucial for evaluating long-term reasoning. In contrast, the interactive fiction game The Invisible Guardian offers a rich, evolving storyline that naturally tests long-term dependencies. Third, real-world data is messy and difficult to control [Xie et al., 2025, Behr et al., 2025]. It is influenced by numerous external factors, making it hard to isolate causal relationships and define clear “success” or “failure” paths. The structured and controlled environment of an interactive fiction game provides a clear framework for evaluating long-term memory and decision-making in a repeatable manner.

Our design incorporates several distinctive features for evaluating LTM. First, unlike conventional QA or dialogue datasets that consist of isolated or short-context samples, our dataset presents a continuous and evolving story world that unfolds over multiple interactive turns, offering a naturalistic setting for evaluating long-horizon reasoning. Second, many long-term choices depend on events or facts introduced several turns earlier, thereby testing models' long-term dependency tracking. Third, the story dynamically evolves based on the model's choices, allowing branching into different paths, including success or failure endings. Fourth, the benchmark reflects realistic decision-making complexity: consecutive choices are often interdependent, requiring models to maintain logical consistency across transitions. Finally, the dataset is multi-solution: multiple choice paths may lead to successful conclusions, emphasizing adaptability rather than rigid answer matching.

### 4.2 Structural Organization

The dataset is organized as a directed acyclic graph (DAG) composed of two types of nodes: scene nodes, which represent narrative fragments, and choice nodes, which define branching decision points. Edges denote transitions between these nodes, forming a tree-like structure that allows non-linear progression through the story. This organization not only captures the dynamic and interactive nature while enabling clear tracing of causal dependencies but also allows flexible nuanced evaluation of LTM in knowledge retention and sequential reasoning.

<!-- page 14: doc_7.md -->

To illustrate the complexity and diversity of our dataset's structure, we categorize representative graph patterns in 

### 4.3 Data Source and Annotation Process

We construct our dataset based on the interactive fiction game The Invisible Guardian from the game's prologue to Chapter 5 by far. Manual annotation preserves the game's branching logic and causal relationships, ensures chronological ordering with memory checkpoints, and annotates metadata on transitions, dynamics, and ethics to retain sequential depth for evaluating LLMs' long-term reasoning. All content is meticulously transcribed from the original game, encompassing dialogues, narrative descriptions, character interactions, and player decision points, with each entry structured as a JSON object annotated with granular details according to its type. Scene nodes (311 entries) include unique identifiers, location, characters with descriptive attributes, sequential dialogues with speaker labels, and flags for narrative endings (where applicable), such as ending (Figure 4). Choice nodes (86 entries) feature unique identifiers, decision context descriptions, and branching options with distinct IDs and text (Figure 5).

<!-- page 15: doc_8.md -->

## 5 Experiments & Results

### 5.1 Experimental Setup

We conduct experiments on four representative foundation models: Doubao 1.5-pro-256k [ByteDance, 2025], GPT-4o [OpenAI, 2024], Claude 3.5 Sonnet [Anthropic, 2024], and Deepseek-R1 [DeepSeek-AI et al., 2025]. These models are chosen based on both their broad real-world usage and competitive performance. Doubao 1.5-pro-256k excels in handling extremely long contexts with its 256k-token support, making it ideal for tasks requiring extensive context retention. GPT-4o, as a leading closed-source commercial model, demonstrates strong language understanding and reasoning abilities. Claude 3.5 Sonnet excels in long-context understanding and knowledge reasoning (supporting 200k+ tokens), maintaining stable performance in long-text reasoning and structural analysis tasks. Deepseek-R1 employs pure reinforcement learning, which gives it excellent logical reasoning and structured thinking capabilities. It shows strong performance in multi-step reasoning and planning tasks. Their diverse features make them ideal for evaluating long-term memory across different technical approaches and application scenarios. While several memory-augmented approaches [Chhikara et al., 2025, Yu et al., 2024] have RAG-style architectures or external memory buffers, we exclude them from our evaluation because their memory utility centers on retrieving isolated factual content. However, StoryBench emphasizes long-term sequential reasoning, where memory must support inference, self-correction, and causal tracking.

For each of the two task modes in StoryBench, we run 10 trials per model. Inputs are carefully formatted to encourage structured reasoning, and we adopt a Chain-of-Thought (CoT) prompting strategy to stimulate stepwise deliberation. In Immediate Feedback mode (results in Table 2), we observe that GPT-4o is more sensitive to content filtering issues (e.g., mentioning weapon-related terms) and frequently interrupts completion due to server overload. To ensure smooth evaluation, we filter potentially problematic vocabulary and limit single-turn inputs to 5,000 tokens for GPT-4o. In Self Recovery mode, models often repeatedly select the same wrong option more than ten times without real-time feedback, therefore stalling the task. So we implement a soft intervention by revealing the correct answer if a model failed at the same decision point for nine consecutive attempts. In the initial evaluation phase (results in Table 3), we retain the original unfiltered dataset and deliberately remove token limits to simulate high-pressure, long-horizon conditions, then conduct five trials. The performance of all models decreases significantly, reflecting the intrinsic difficulty of the task. In response, we launch a second phase of five-trial experiments (Table 4) with improved input handling: sensitive vocabulary is filtered and a 5,000-token per turn limit is applied.

### 5.2 Main Results of Long-Term Memory Performance

 

 

#### 5.2.1 Model Analysis

To better understand the performance differences among models, we analyze five core metrics illustrated in

<!-- page 16: doc_9.md -->

the model's capacity to navigate complex, dynamic, and multi-step decision paths involving long-range dependencies. Success Count captures the overall task-completion ability. Among all models, Doubao1.5-pro achieves the highest scores in knowledge-related metrics such as Overall Accuracy and First-Try Accuracy, suggesting strong capabilities in knowledge retention. Doubao effectively absorbs and integrates contextual information across extended texts. However, long-term memory evaluation must prioritize not accuracy but the ability to complete extended decision paths. That is because the premise for evaluation is the model can complete the story chain, otherwise, no matter how high the local accuracy is, it will lose significance to the evaluation. Doubao's Success Count is significantly lower than Claude 3.5 Sonnet, indicating that it often "dies in details" when dealing with complex reasoning chains and long-term interactive tasks despite its solid knowledge base. In contrast, Claude 3.5 Sonnet maintains a solid balance: it trails slightly in accuracy, but excels in degree of completion, achieving the highest Success Count. This suggests Claude is more robust in multi-turn sequential reasoning, which is a critical factor in long-term memory evaluation.

Interestingly, most models show large gaps between Easy and Hard Accuracy, 

From an efficiency perspective, GPT-4o and Doubao1.5-pro offer excellent cost-performance tradeoffs. Their Runtime Cost and Token Consumption are significantly lower than Claude 3.5 and Deepseek-R1.