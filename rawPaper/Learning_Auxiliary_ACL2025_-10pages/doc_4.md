
<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2">Dataset</td><td colspan="3">Method</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Fine-Tuning</td><td style='text-align: center; word-wrap: break-word;'>w.o. aux</td><td style='text-align: center; word-wrap: break-word;'>RATE-FT</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>LongFact</td><td style='text-align: center; word-wrap: break-word;'>76.1</td><td style='text-align: center; word-wrap: break-word;'>77.5</td><td style='text-align: center; word-wrap: break-word;'>79.6</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Biography</td><td style='text-align: center; word-wrap: break-word;'>78.2</td><td style='text-align: center; word-wrap: break-word;'>79.4</td><td style='text-align: center; word-wrap: break-word;'>80.9</td></tr></table>

<div style="text-align: center;">Table 3: Results of different ablations.</div>


in Table 2, which demonstrates the superiority of RATE-FT across different datasets (see Appendix A.6 for an analysis of the effect of additional data augmentation compared to the auxiliary QA task).

### 5.1 Further Analysis

Ablation Study We analyze the contribution of different components of RATE-FT by investigating the variant of RATE-FT without the auxiliary task (w.o. aux). Table 3 presents the performance of different methods, highlighting that each component plays an important role in achieving the overall performance.

Generalization to Different Models Our experiments and analysis so far use Llama-3-8B-Instruct as the backbone model. To verify whether the performance gain of RATE-FT is consistent across different backbone models, we extend the experiments to Llama-3.1-70B-Instruct (Dubey et al., 2024), Mistral-7B-Instruct (Jiang et al., 2023), and Qwen2.5-7B-Instruct (Yang et al., 2024) on LongFact (see Appendix A.7 for details on data collection). From the results shown in Table 4, we can observe that RATE-FT consistently outperforms baseline approaches across all models, demonstrating its robustness and generalizability to diverse model architectures and scales.

In addition, we provide results of incorporating uncertainty for hallucination detection, all prompts used in our experiments, and implementation details in Appendix A.8 ~ A.12, respectively.

## 6 Conclusion

In this work, we systematically investigate reference-free hallucination detection in opendomain long-form generation. Our study begins with an analysis of the model's internal states, demonstrating that these states alone cannot reliably detect hallucinations. We then evaluate several existing approaches, including prompting, probing, and fine-tuning, with fine-tuning emerging as the most effective method. Building on these findings, we introduce Rationale and Auxiliary Task Enhanced Fine-Tuning (RATE-FT), a novel approach


<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2">Model</td><td colspan="5">Method</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Prompt_{TF}</td><td style='text-align: center; word-wrap: break-word;'>Prompt_{CoT-TF}</td><td style='text-align: center; word-wrap: break-word;'>Probing</td><td style='text-align: center; word-wrap: break-word;'>Fine-Tuning</td><td style='text-align: center; word-wrap: break-word;'>RATE-FT</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Llama-3.1-70B-Instruct</td><td style='text-align: center; word-wrap: break-word;'>73.2</td><td style='text-align: center; word-wrap: break-word;'>76.8</td><td style='text-align: center; word-wrap: break-word;'>79.4</td><td style='text-align: center; word-wrap: break-word;'>80.6</td><td style='text-align: center; word-wrap: break-word;'>83.8</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Mistral-7B-Instruct</td><td style='text-align: center; word-wrap: break-word;'>61.8</td><td style='text-align: center; word-wrap: break-word;'>64.1</td><td style='text-align: center; word-wrap: break-word;'>68.4</td><td style='text-align: center; word-wrap: break-word;'>70.8</td><td style='text-align: center; word-wrap: break-word;'>73.4</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Qwen2.5-7B-Instruct</td><td style='text-align: center; word-wrap: break-word;'>72.8</td><td style='text-align: center; word-wrap: break-word;'>75.5</td><td style='text-align: center; word-wrap: break-word;'>77.0</td><td style='text-align: center; word-wrap: break-word;'>78.4</td><td style='text-align: center; word-wrap: break-word;'>81.1</td></tr></table>

<div style="text-align: center;">Table 4: Results using different models.</div>


that leverages rationales and an auxiliary task to achieve significant improvements in detection performance across two datasets and various LLMs.

## Limitations

One limitation of our work is its focus solely on improving the performance of the hallucination detector. A potential improvement could be to explore leveraging the detector's feedback as a reward signal to guide LLMs to generate more factual responses. Additionally, developing a more comprehensive benchmark for hallucination detection in open-domain long-form generation that covers a broader range of domains would further enhance its applicability.

## References

David Paul Ausubel. 2012. The acquisition and retention of knowledge: A cognitive view. Springer Science & Business Media.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Jifan Chen, Grace Kim, Aniruddh Sriram, Greg Durrett, and Eunsol Choi. 2024a. Complex claim verification with evidence retrieved in the wild. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3569–3587, Mexico City, Mexico. Association for Computational Linguistics.

Lida Chen, Zujie Liang, Xintao Wang, Jiaqing Liang, Yanghua Xiao, Feng Wei, Jinglei Chen, Zhenghong Hao, Bing Han, and Wei Wang. 2024b. Teaching large language models to express knowledge boundary from their own signals. arXiv preprint arXiv:2406.10881.