Llama-3-8B-Inst model and our proposed model. The evaluation was conducted under the "Scaling with the Number of Relevant Passages" setting, focusing on scenarios with 40 passages.

Table 4 presents the evaluation results of various models regarding accuracy and KF1. The Llama-3-8B-Inst model serves as the baseline, while Llama-3-8B-Inst (chosen) and Llama-3-8B-Inst (rejected) are fine-tuned variants trained on the chosen and rejected synthetic data, respectively. Our model demonstrates the highest performance, achieving 65.76 in accuracy and 43.78 in KF1, surpassing all other models.

## 5 Conclusion

The improved accuracy and KF1 scores of Llama-3-8B-Inst (chosen), compared to Llama-3-8B-Inst, indicate that the chosen synthetic data enhances accuracy and reduces confirmation bias. In contrast, the lower accuracy and KF1 scores of Llama-3-8B-Inst (rejected) suggest that the rejected data increases confirmation bias and contains less reliable answers. These results highlight that the synthetic data we generated exhibit high quality in both its chosen and rejected subsets, supporting our approach to reducing confirmation bias.

In this work, we identified confirmation bias as a key factor contributing to hallucination in question answering under long-context scenarios. To address this, we proposed a dataset construction pipeline aimed at mitigating confirmation bias in both the chosen and rejected responses. Our method ensures the integration and effective utilization of knowledge segments that are closely related to the given question within the context. We then trained our model using the DPO-based approach.

In this paper, we have demonstrated the occurrence of confirmation bias in language models and proposed methodologies to mitigate it. For future work, it will be essential to investigate the underlying causes of confirmation bias in language models and provide empirical evidence to substantiate them. Additionally, we plan to explore the presence of confirmation bias in other tasks beyond question answering and investigate the application of our proposed methodology to these scenarios.

## 6 Limitations

The proposed method in the paper generates a DPO dataset for cases with gold knowledge. However, this is generated based on the assumption that the question is unconditionally answerable, so it can cause hallucinations in the unanswerable case. Depending on the performance of the RAG, only irrelevant knowledge may be retrieved, and generating data that accounts for this can further improve the model's ability to comprehend the retrieved knowledge. In addition, you can also consider the case where the knowledge contains contradictions.



Our proposed synthetic data distribution optimization methodology for mitigating confirmation bias in LLM models can be applied to an infinite number of tasks. We have verified its effectiveness in the multi-document question answering task, which is the easiest task to verify. We leave the experimentation of applying our method to various tasks such as summarization, document writing, editing, and rewriting as future work.

While the proposed method in the paper proved to be effective in mitigating confirmation bias, some issues may persist due to inherent vulnerabilities that naturally exist in humans.

## Acknowledgments

This work was supported by the Ministry of Education of the Republic of Korea and the National Research Foundation of Korea(NRF-2022S1A5A2A03052880).

## References

Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, and Jian-Guang Lou. 2024. Make your llm fully utilize the context. arXiv preprint arXiv:2404.16811.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. 2024. Longalign: A recipe for long context alignment of large language models. arXiv preprint arXiv:2401.18058.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards retrieval-augmented large language models. In Proceedings of the 30th ACM SIGKDD Conference on