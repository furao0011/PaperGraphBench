(LLMs) can resist snowball hallucination, we created a Chinese Medical Hallucination Evaluation Dataset (CMHE) with 42,198 samples. This dataset aims to assess LLMs' ability to identify misinformation, perform accurate reasoning in noisy environments, and minimize the generation of erroneous content. The dataset encompasses 2,000 questions related to hallucination detection, 1,622 questions for diagnosis, and 38,576 questions for concept explanation, allowing a comprehensive assessment of each of the aforementioned aspects. In contrast to previous investigations, our dataset does not offer predetermined response options for the model to select from. Instead, we simulate conversational scenarios by prompting the model to generate responses freely.

In order to ensure the specificity of the examination, we employed various construction strategies when creating the three types of tasks. For the hallucination detection, we create test samples using generation and tampering based approaches. These samples assess the model's ability to identify hallucinations that contradict medical knowledge and those that defy contextual logic. Samples for diagnosis tasks include standard medical exam questions and manually crafted test questions extracted from web-based consultation data. This allows us to evaluate the model's reasoning ability in scenarios with and without interfering information. Samples for concept explanation are created using Medical Subject Headings (Lipcomb, 2000, MeSH) with specific rules. This ensures that a broad and exhaustive spectrum of concepts are included in the assessment. Furthermore, we structure the task as a self-familiar test (Luo et al., 2023) to evaluate the extent of hallucinatory phenomena present in the model responses.

The contributions of our work can be summarized as follows:

• We propose a comprehensive benchmark to evaluate Chinese medical hallucination in LLMs. This benchmark includes three tasks: identifying hallucinations, diagnosing disease in noisy environments, and explaining specific concepts.

• We constructed three brand-new data sets for the proposed benchmark, which covers various hallucinations, all kinds of disease categories, and various medical concepts.

• We use our benchmark to evaluate three popular LLMs for Chinese medical purposes. The experimental results reveal the following findings: First, LLMs are better at recognizing hallucinations caused by logic errors rather than knowledgeable errors. Second, redundancy information can lower the accuracy of LLMs in disease diagnosis. Third, understanding of concepts by large models can impact their performance in noisy environments.



### 2. Related Work

#### 2.1. Hallucination of LLMs

Hallucination is when language generation models produce unreliable or nonsensical text (Ji et al., 2023; Zhang et al., 2023c). It can be classified based on presentation: contradicting instructions (Ji et al., 2023), contradicting context (Maynez et al., 2020), and contradicting facts.

In recent years, researchers have focused extensively on identifying the causes of hallucinations with the aim of eliminating them. Studies conducted by Li et al. (2023); McKenna et al. (2023) found a strong connection between the hallucination of LLMs and the distribution of training data. Azaria and Mitchell (2023); Lee et al. (2022) argue that flawed decoding strategies are responsible for the occurrence of hallucinations. Moreover, LLMs exhibit a proclivity for producing a higher volume of inaccurate information by building upon previously generated erroneous sentences, a phenomenon commonly known as "hallucination snowballing" (Zhang et al., 2023b). Researchers like Schulman (2023) have found that the preference alignment process in LLMs often results in these models becoming overconfident when dealing with unfamiliar tasks. Kadavath et al. (2022); Yin et al. (2023) have also observed that this overconfidence can result in the production of error information.

Based on the these findings, researchers try to eliminate hallucination of LLMs in pre-training (Touvron et al., 2023), supervised fine-tuning (Zhou et al., 2023, SFT), reinforcement learning with human feedbacks (Schulman, 2023, RLHF), inference (Mialon et al., 2023) stages. Although these studies have attracted a lot of attention, hallucination evaluation is still the main bottleneck of improving the elimination performance.

#### 2.2. Hallucination Evaluation

The existing benchmarks for evaluating hallucinations in language models (LLMs) primarily concentrate on two key abilities: generating factual statements and distinguishing between factual and nonfactual statements (Zhang et al., 2023c). The evaluation of the generation task typically employs metrics such as BLEU (Papineni et al., 2002), ROUGE (Lin, 2004), and FActScore (Min et al., 2023) to assess the similarity between the model's output and the reference answer. A higher similarity score indicates greater confidence in the model's performance. TruthfulQA (Lin et al., 2021) serves as an example of a commonly used dataset for this