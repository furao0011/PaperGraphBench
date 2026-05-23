
<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2">Model</td><td colspan="2">CaseLaw</td><td colspan="2">CAIL18</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Acc</td><td style='text-align: center; word-wrap: break-word;'>F1</td><td style='text-align: center; word-wrap: break-word;'>Acc</td><td style='text-align: center; word-wrap: break-word;'>F1</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Few-shot</td><td style='text-align: center; word-wrap: break-word;'>63.8%</td><td style='text-align: center; word-wrap: break-word;'>64.1%</td><td style='text-align: center; word-wrap: break-word;'>29.7%</td><td style='text-align: center; word-wrap: break-word;'>5.03%</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>CoT (4-steps)</td><td style='text-align: center; word-wrap: break-word;'>63.7%</td><td style='text-align: center; word-wrap: break-word;'>64.0%</td><td style='text-align: center; word-wrap: break-word;'>31.2%</td><td style='text-align: center; word-wrap: break-word;'>6.17%</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Reflexion</td><td style='text-align: center; word-wrap: break-word;'>64.5%</td><td style='text-align: center; word-wrap: break-word;'>65.0%</td><td style='text-align: center; word-wrap: break-word;'>31.8%</td><td style='text-align: center; word-wrap: break-word;'>8.12%</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Debate-Feedback (single)</td><td style='text-align: center; word-wrap: break-word;'>66.2%</td><td style='text-align: center; word-wrap: break-word;'>65.7%</td><td style='text-align: center; word-wrap: break-word;'>41.9%</td><td style='text-align: center; word-wrap: break-word;'>16.1%</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Debate-Feedback (assistant)</td><td style='text-align: center; word-wrap: break-word;'>67.1%</td><td style='text-align: center; word-wrap: break-word;'>66.1%</td><td style='text-align: center; word-wrap: break-word;'>44.8%</td><td style='text-align: center; word-wrap: break-word;'>16.3%</td></tr></table>

<div style="text-align: center;">Table 3: Performance comparison of different reasoning methods on CaseLaw and CAIL18 datasets.</div>


As shown in table[3], Debate-Feedback structure achieves significant advantages in comparison with several basic reasoning frameworks. The results show that Chain-of-Thought and Reflection perform only marginally better than Zeroshot, while our Debate-feedback framework consistently demonstrates superior performance, reinforcing the conclusions of our original experiments.

We believe there are two primary reasons why standard reasoning techniques like CoT and Reflection are less effective for this type of legal prediction problem:

Complexity of Legal Texts: The legal text itself is lengthy and logically complex, and simple prompts are difficult to be effective.

## 5 Conclusion

Nature of Legal Prediction: Legal prediction is always different from logical reasoning. It is not a step-by-step thinking toward the correct answer, but usually a discussion to unify or compromise the views of multiple parties. This is precisely why we designed the Debate-feedback framework, which is tailored to handle such tasks.

We propose a debate-feedback model based on LLMs for legal judgment prediction and demonstrated its feasibility through experiments. The inclusion of an assistant model and reliability analysis enhances prediction robustness. Future work could explore the application of debate models in other fields or further integrate them with LLMs.

## 6 Limitations

Our work currently has the following limitations: framework, their individual contributions to the overall performance were not deeply investigated.



(a) The experiments were limited to two datasets and two specific tasks, broader evaluations across additional datasets and tasks are necessary to fully validate the model's robustness and generalizability in different legal contexts.

(c) This work does not integrate retrieval argument techniques, which presents a promising direction for future research to enhance the model's performance.



(b) While the smoothing technique and assistant model (reliability analysis) were included in the

## References

Nikolaos Aletras, Dimitrios Tsarapatsanis, Daniel Preotiuc-Pietro, and Vasileios Lampos. 2016. Predicting judicial decisions of the european court of human rights: A natural language processing perspective. PeerJ Computer Science, 2:e93.

Ilias Chalkidis, Michael Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2019. Legalbert: The muppets straight out of law school. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings, pages 2898–2904.

Junyun Cui, Xiaoyu Shen, and Shaochun Wen. 2023. A survey on legal judgment prediction: Datasets, metrics, models and challenges. IEEE Access, 11:102050–102071.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. Preprint, arXiv:1810.04805.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. 2024. A survey on in-context learning. Preprint, arXiv:2301.00234.

Daniel A. Gutierrez-Pachas, Eduardo F. Costa, and Alessandro N. Vargas. 2022. Distribution of a markov chain in reverse-time with cluster observations in the extremes of a finite time window. Preprint, arXiv:2206.05607.

Zihan Huang, Charles Low, Mengqiu Teng, Hongyi Zhang, Daniel E. Ho, Mark S. Krass, and Matthias Grabmair. 2021. Context-aware legal citation recommendation using deep learning. In Proceedings of the Eighteenth International Conference on Artificial Intelligence and Law, ICAIL '21, page 79–88, New York, NY, USA. Association for Computing Machinery.

Geoffrey Irving, Paul Christiano, and Dario Amodei. 2018. Ai safety via debate. arXiv preprint arXiv:1805.00899.

Daniel Katz, Michael Bommarito, and Josh Blackman. 2017. A general approach for predicting the behavior of the supreme court of the united states. PLOS ONE, 12.