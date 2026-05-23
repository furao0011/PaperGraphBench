#### 5.4. Findings and Directions

Based on the aforementioned experimental results and analysis, several significant findings have been identified on the evaluation of hallucinations. Finding 1: LLMs typically detect hallucinations by evaluating the logical coherence of the sentence context. However, their ability to identify false information, such as manipulated drug names and treatment plans, is limited. Finding 2: LLMs exhibit strong performance in environments devoid of interfering information. However, their performance tends to deteriorate in noisy environments, such as when patients provide a substantial amount of invalid information. Finding 3: LLMs that possess a deeper understanding of medical concepts exhibit improved performance in noisy environments.

Due to the poor performance of LLMs in noisy environments, exploring ways to enhance the robustness of LLMs during inference, especially when LLMs are aware of their errors but tend to perpetuate their previous falsehoods, will be an intriguing avenue for future investigation.

### 6. Conclusion

Hallucination evaluation is a major challenge for LLMs' application in the Chinese medical domain, especially snowballing hallucination problems. Most existing studies rely on automatic indicators and lack an intuitive evaluation of hallucinations. To appreciate LLM's ability on hallucination perception sufficiently, we need to decompose this problem into several aspects, e.g., identifying medical hallucinations, making accurate diagnoses in noisy conditions, providing plausible explanations, etc. CMHE specifically targets the assessment of comprehensive hallucinations in Chinese medical chat scenarios. This involves evaluating various processes that could potentially lead to hallucinations, such as errors in identification, reasoning, diagnosis, concept explanation, and exploitation. Our findings demonstrate that LLMs excel at detecting inconsistencies but struggle in noisy environments with redundant information. However, when LLMs possess a solid understanding of concepts, their performance can be greatly enhanced. In conclusion, researchers can easily locate the type of hallucinations and identify the lack of understanding in LLM through failures of CMHE, and CMHE can serve as a valuable benchmark to assess hallucinations in Chinese medical contexts.

### 7. Ethics and Limitations

The CMHE benchmark is constructed using a widely used public corpus. All information in the corpus has been anonymized and excludes any personal data, and it is publicly accessible online. Additionally, during the annotation process, we require annotators to manually filter and screen sensitive information to ensure the protection of personal privacy. While W2W shows great potential, it is essential to assess its ethical and societal implications. Our task definition and research models rely on pre-trained language models and public datasets, which may contain hidden biases leading to fairness issues within the algorithms. By acknowledging and actively addressing these implications, our aim is to raise awareness among practitioners if the model is deployed as a language-learning agent in the future.



### 8. Acknowledgement

Our work is supported by the National Key Research and Development Program of China (Project Number: 2020AAA0109400). We kindly appreciate all the researchers who provide valuable insights, discussions, and comments on this work.

### 9. Bibliographical References

Amos Azaria and Tom Mitchell. 2023. The internal state of an llm knows when its lying. arXiv preprint arXiv:2304.13734.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Jiaxi Cui, Zongjian Li, Yang Yan, Bohua Chen, and Li Yuan. 2023. Chatlaw: Open-source legal large language model with integrated external knowledge bases. arXiv preprint arXiv:2306.16092.

Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. 2023. Chain-of-verification reduces hallucination in large language models. arXiv preprint arXiv:2309.11495.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2021. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421.