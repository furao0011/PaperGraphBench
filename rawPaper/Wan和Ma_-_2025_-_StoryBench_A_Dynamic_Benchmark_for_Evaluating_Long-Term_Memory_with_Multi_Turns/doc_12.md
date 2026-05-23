text-based, both of which may limit the benchmark's generalizability to other knowledge-intensive or task-oriented contexts that require multimodal support. Second, the number of turns and the length of the context are still limited. The current interactive fiction dataset consists of only 6 chapters, which may not fully capture the long-term dependencies and complex reasoning required in more extensive narratives. Future work could expand the dataset by adding subsequent chapters to provide a more comprehensive evaluation of long-term memory. Third, due to API constraints and cost, we primarily evaluate a limited number of mainstream models. The performance of other models under similar conditions remains unexplored. Fourth, although we include a self-recovery setting to simulate real-world error correction, the evaluation remains scripted and cannot capture all forms of natural feedback.

## 7 Conclusions

We introduce StoryBench, a novel benchmark designed to systematically evaluate long-term memory capabilities in complex, dynamic, and multi-turn narrative environments. By simulating realistic memory demands across story understanding, sequential inference, and flexible correction, our benchmark assesses current mainstream models in knowledge retention and sequential reasoning. Through comprehensive experiments on representative models and detailed failure case analyses, we demonstrate that current models exhibit significant performance gaps on our benchmark, highlighting StoryBench's difficulty and effectiveness in evaluating long-term memory capabilities. Our findings underscore the importance of developing more robust memory mechanisms, laying the groundwork for future research toward memory-augmented, context-aware language agents.

## References

Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. L-eval: Instituting standardized evaluation for long context language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14388–14411, 2024.

Anthropic. Claude 3.5 sonnet model card addendum, 2024. URL https://www.anthropic.com/news/claude-3-5-sonnet.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3119–3137, 2024.

Matthias Behr, Ralf Burghaus, Christoph Diedrich, et al. Opportunities and challenges for ai-based analysis of rwd in pharmaceutical r&d: A practical perspective. Künstliche Intelligenz, 39(1):7–18, 2025. doi:10.1007/s13218-023-00809-6. URL https://doi.org/10.1007/s13218-023-00809-6.

Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

ByteDance. Doubao-1.5-pro, 2025. URL https://seed.bytedance.com/zh/special/doubao_1_5_pro.

David Castillo-Bolado, Joseph Davidson, Finlay Gray, and Marek Rosa. Beyond prompts: Dynamic conversational benchmarking of large language models. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. ACM transactions on intelligent systems and technology, 15(3):1–45, 2024.

Bo Chen, Yingyu Liang, Zhizhou Sha, Zhenmei Shi, and Zhao Song. Hsr-enhanced sparse attention acceleration. arXiv preprint arXiv:2410.10165, 2024.

Ching-An Cheng, Andrey Kolobov, Dipendra Misra, Allen Nie, and Adith Swaminathan. Llf-bench: Benchmark for interactive learning from language feedback. In ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.