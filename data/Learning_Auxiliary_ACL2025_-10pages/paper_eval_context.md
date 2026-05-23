# Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation
Chengwei Qin $ ^{\dagger,*} $, Wenxuan Zhou $ ^{\dagger} $, Karthik Abinav Sankararaman $ ^{\dagger} $, Nanshu Wang $ ^{\dagger} $, Tengyu Xu $ ^{\dagger} $, Alexander Radovic $ ^{\dagger} $, Eryk Helenowski $ ^{\dagger} $, Arya Talebzadeh $ ^{\dagger} $, Aditya Tayade $ ^{\dagger} $, Sinong Wang $ ^{\dagger} $, Shafiq Joty $ ^{\dagger} $, Han Fang $ ^{\dagger} $, Hao Ma $ ^{\dagger} $
## Abstract

Hallucination, the generation of factually incorrect information, remains a significant challenge for large language models (LLMs), especially in open-domain long-form generation. Existing approaches for detecting hallucination in long-form tasks either focus on limited domains or rely heavily on external fact-checking tools, which may not always be available.

In this work, we systematically investigate reference-free hallucination detection in open-domain long-form responses. Our findings reveal that internal states (e.g., model's output probability and entropy) alone are insufficient for reliably (i.e., better than random guessing) distinguishing between factual and hallucinated content. To enhance detection, we explore various existing approaches, including prompting-based methods, probing, and fine-tuning, with fine-tuning proving the most effective. To further improve the accuracy, we introduce a new paradigm, named RATE-FT, that augments fine-tuning with an auxiliary task for the model to jointly learn with the main task of hallucination detection. With extensive experiments and analysis using a variety of model families & datasets, we demonstrate the effectiveness and generalizability of our method, e.g., +3% over general fine-tuning methods on LongFact.

## 1 Introduction

With the recent advancements in model scale and pretraining data, large language models (LLMs) have demonstrated remarkable capabilities in various natural language processing (NLP) tasks (Brown et al., 2020). Despite these successes, hallucination, where models tend to produce content that conflicts with real-world facts, remains a significant challenge (Zhang et al., 2023). Most existing research on hallucination detection has

[Figure 1 was here. The original paper contained a figure at this position. Brief visual description: The figure compares a standard Fine-Tuning process with the proposed RATE-FT method for hallucination detection. The top section illustrates standard training where a detector is trained directly on (Claim, Label) data. The bottom section, labeled RATE-FT, shows an enhanced pipeline where the initial (Claim, Label) data undergoes 'Rationale Augmentation' and 'Auxiliary Task Augmentation' to produce richer datasets containing rationales and question-answer pairs, which are then used to train the detector.]
Caption: Figure 1: Comparison between Fine-Tuning and RATE-FT for hallucination detection. RATE-FT improves Fine-Tuning by incorporating rationales and an auxiliary task (question answering) into the training process.
Key visible elements:
- Standard Training Setup: Top panel showing baseline fine-tuning
- RATE-FT Setup: Bottom panel showing the proposed augmented method
- Input Data (Claim, Label): Original dataset source in both panels
- Detector (Top): Model being trained in standard setup, marked with a question mark
- Detector (Bottom): Model being trained in RATE-FT, marked with a checkmark
- Rationale Augmentation: Process step adding rationales to the data
- Augmented Data 1: Dataset containing (Claim, Label, Rationale)
- Auxiliary Task Augmentation: Process step creating QA pairs with rationales

cused on short-form tasks, where the output consists of one or a few tokens. While these methods are effective for short-form content (Manakul et al., 2023; Mahaut et al., 2024; Yehuda et al., 2024; Zhang et al., 2024a), extending them to open-domain long-form generation presents additional complexities and new challenges. Unlike short-form tasks, long-form responses can span hundreds or even thousands of tokens, requiring models to generate detailed and nuanced answers to broad fact-seeking prompts (Wei et al., 2024). This necessitates that LLMs synthesize information across multiple knowledge domains, increasing the risk of generating content that sounds plausible yet is factually incorrect. For example, when answering 'What is the significance of Amber Room?', LLMs may generate responses that mix accurate historical information with fabricated details, complicating the task of distinguishing fact from hallucination.

Recent efforts have sought to address hallucination detection in long-form tasks. However, they either focus on limited domains, e.g., biography generation (Min et al., 2023; Fadeeva et al., 2024) or rely heavily on external fact-checking tools or

knowledge bases, e.g., Google Search (Wei et al., 2024). While these tools offer valuable support, they are not always available or scalable. This raises an important question: can we develop hallucination detectors that rely solely on the model itself, without the need for external fact-checking resources? So far, little attention has been given to systematically exploring how the model's own mechanisms can be used for detecting hallucinations in open-domain long-form generation.

To address this gap, we start by investigating hallucination detection in open-domain long-form responses using the model's internal states, e.g., output probability and entropy. Specifically, we decompose long-form responses into atomized claims using the model and verify each claim's correctness using Google Search to construct benchmark data following Wei et al. (2024). Our analysis reveals that these internal states alone are insufficient for reliably (i.e., better than random guessing) distinguishing between correct and incorrect claims, indicating that the mechanisms for detecting hallucinations in long-form outputs differ significantly from those in short-form tasks. To enhance detection, we explore several existing methods, including prompting, probing, and fine-tuning LLMs. Our experimental results show that fine-tuning LLMs is the most effective method to detect hallucinations.

Building on this, we introduce a novel method Rationale and Auxiliary Task Enhanced Fine-Tuning (RATE-FT) (Figure 1). Specifically, we convert the original claims into auxiliary question answering (QA) examples for augmentation, providing a complementary learning perspective for the model, which enables better generalization. Additionally, we incorporate collected rationales into the training process for better reasoning. Extensive experiments and analysis using different models demonstrate the effectiveness and generalizability of our approach. Furthermore, we investigate the integration of model uncertainty into hallucination detection in Appendix A.8. In summary, our main contributions are:

• We are the first to systematically investigate reference-free hallucination detection in open-domain long-form generation by analyzing a representative set of existing methods.

• We introduce a novel approach that incorporates rationales and an auxiliary question answering task into fine-tuning, achieving significant performance improvements.

## 2 Related Work

Large Language Models (LLMs) often generate content that appears plausible but is factually unsupported, a phenomenon commonly known as hallucination (Zhang et al., 2023). Based on whether the hallucinated content contradicts real-world facts or the input context, hallucination can be categorized into two main groups: factuality hallucination and faithfulness hallucination (Huang et al., 2023). Extensive research has been conducted on exploring the causes (Onoe et al., 2022; Kang and Choi, 2023; Wei et al., 2023; Liu et al., 2024), detection (Min et al., 2023; Zhao et al., 2023; Chen et al., 2024a; Fadeeva et al., 2024; Wei et al., 2024), and mitigation (Gao et al., 2023; Ji et al., 2023; Tian et al., 2024; Zhang et al., 2024b; Kang et al., 2024; Lin et al., 2024) of hallucination in LLMs. However, most existing hallucination detection methods have primarily focused on short-form tasks, where the output consists of one or a few tokens. In this work, we shift the focus to the more challenging problem of reference-free hallucination detection in open-domain long-form generation, where outputs are substantially longer and require a more nuanced evaluation of actuality.

## 3 Are LLMs’ Internal States Sufficient for Open-Domain Long-Form Generation?

The internal states of LLMs, such as output probability and entropy, have been shown to be effective in detecting hallucinations in short-form tasks, where outputs are typically limited to only a few tokens. By analyzing these signals, models can often differentiate between factual and hallucinated information. However, their applicability in open-domain long-form generation remains underexplored. A key question is whether LLMs can depend solely on their internal states to identify hallucinations in long-form generation, without using external fact-checking tools. To answer it, we conduct some pilot experiments on LongFact (Wei et al., 2024), a long-form generation dataset spanning 38 different domains. Specifically, for each prompt in the sampled subset (200 prompts), we obtain a long-form response from Llama-3-8B-Instruct with greedy decoding. Following Wei et al. (2024), we employ the model to decompose long-form responses into atomized claims and label them as ‘factual’ or ‘hallucinated’ together with the reasons (see Appendix A.1 for construction details).

For each claim, we mainly focus on two types

[Figure 2 was here. The original paper contained a figure at this position. Brief visual description: The figure presents eight histograms comparing the distribution of token probabilities for 'Normal' (blue) versus 'Hallucinated' (orange) tokens. The subplots represent different methods for calculating or selecting these probabilities, including averages (`arithmetic_average`, `geometric_average`) and lowest probability selections (`lowest_1` through `lowest_15%`). The x-axis represents probability values ranging from 0 to 1, while the y-axis indicates the count of tokens.]
Caption: Figure 2: Detection results based on token probability.
Key visible elements:
- Eight histogram subplots: Displaying frequency distributions of token probabilities
- Legend (Normal/Hallucinated): Distinguishing between correct and incorrect generation outcomes
- X-axis labels (method names): Identifying the specific probability calculation method used in each subplot
- Y-axis (Counts): Indicating the frequency of tokens falling into specific probability bins

of internal states to estimate factual confidence following SelfCheckGPT (Manakul et al., 2023): the probability or the entropy (uncertainty) of output tokens. Specifically, we examine the arithmetic and geometric $ {}^{1} $ averages of all tokens, the average of tokens with the top-K lowest probability or highest entropy ( $ K = 1, 3, 5 $), and the average of tokens with the top-P% lowest probability or highest entropy ( $ P = 5, 10, 15 $). The results in Figure 2 and Appendix A.2 suggest that neither internal state reliably, i.e., better than random guessing, predicts the correctness of a given claim, which may be due to the presence of numerous insignificant tokens within the claim, such as stop words. To address this, we consider variants that focus only on output tokens related to entities. The results, shown in Appendix A.2, reveal similar patterns (see Appendix A.3 for a detailed comparison with the findings in Manakul et al. (2023)). We analyze the underlying reasons as follows. In open-domain long-form generation, claims are not limited to a few tokens, which introduces multiple sources of uncertainty. Specifically, the probability or entropy reflects the model's confidence in how a claim is expressed, i.e., its confidence in the claim as a sequence of output tokens, rather than in the correctness of the claim. Different surface forms of the claim yield different confidence levels, leading to unreliable estimates.

Considering the unreliability of LLMs' internal states in hallucination detection, there are several promising alternative approaches, including prompting, probing and fine-tuning LLMs, which we explore in the next section.

## 4 Prompting, Probing and Fine-Tuning

Based on a review of the research area, we identify three groups of existing hallucination detection methods, which we discuss below.

Prompting Prompting-based approaches involve directly prompting LLMs to assess the correctness of a given claim without additional training. We investigate the following three different methods: (i) Prompting the model to output 'True' or 'False' for a given claim, referred to as Prompt $ _{TF} $. The probability assigned to the token 'True' represents $ P_{factual} $, while the probability assigned to 'False' represents $ P_{hallucinated} $. (ii) Prompting the model to output the probability that it considers the given claim to be correct, referred to as Prompt $ _{Prob} $. This number directly represents $ P_{factual} $. (iii) SelfCheckGPT, which detects hallucinations by sampling additional responses from the model and assessing inconsistencies between each response and the target claim. The proportion of responses that support the claim is taken as $ P_{factual} $. Following Manakul et al. (2023), we sample 20 responses for detection.

Probing Following Su et al. (2024a), we train a multilayer perceptron (MLP) on the contextualized embeddings of LLMs to perform binary classification for hallucination detection, while keeping the base LLM frozen. The trained MLP outputs $ P_{factual} $ as an indicator for classification.

Fine-Tuning We fine-tune the base LLM with LoRA to enhance its ability to output 'True' or 'False' for a given claim (Kapoor et al., 2024). Similar to Prompt $ _{TF} $, the probabilities assigned to the tokens 'True' and 'False' correspond to $ P_{factual} $ and $ P_{hallucinated} $, respectively. Note that LoRA fine-tuning allows us to easily use the original model for general tasks while applying the trained LoRA specifically for hallucination detection.

Following the data construction process outlined in Appendix A.1, we conduct experiments on the full set of LongFact using Llama-3-8B-Instruct. This process yields 2,711 factual and hallucinated claims, which are subsequently split into training (70%), validation (20%), and test (10%) sets. For all three types of methods, we use $ P_{\text{factual}} $ as the classification indicator. Specifically, a claim is classified as ‘factual’ if $ P_{\text{factual}} $ exceeds a predefined threshold; otherwise, it is classified as ‘hallucinated’. The optimal threshold is determined through a search on the validation set. Consistent with Tang et al. (2024); Chen et al. (2024b), we employ balanced accuracy (BAcc) as the evaluation metric: $ \text{BAcc} = \frac{1}{2} \left( \frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}} \right) $, where TP, TN, FP, and FN stand for true/false positives/negatives.

The results of different methods on the test set, as

[Table 1 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 1: BAcc (%) of existing hallucination detection methods on LongFact and biography generation.

Summary: Table 1 reports Balanced Accuracy (BAcc) percentages for five hallucination detection methods (Prompt_TF, Prompt_Prob, SelfCheckGPT, Probing, Fine-Tuning) on the LongFact and Biography generation datasets. The highest BAcc on both datasets is achieved by Fine-Tuning (76.1% on LongFact, 78.2% on Biography), followed by Probing (74.4%, 77.0%), while Prompt_Prob yields the lowest scores (53.4%, 56.3%).

Table LaTeX:

```latex
\begin{tabular}{llllll}
\hline
Dataset & Method & Method & Method & Method & Method \\
\hline
Dataset & \$ Prompt\_\{TF\} \$ & \$ Prompt\_\{Prob\} \$ & SelfCheckGPT & Probing & Fine-Tuning \\
LongFact & 69.9 & 53.4 & 69.1 & 74.4 & 76.1 \\
Biography & 72.3 & 56.3 & 71.9 & 77.0 & 78.2 \\
\hline
\end{tabular}
```

shown in Table 1, indicate that fine-tuning LLMs is the most effective among all existing methods (see Appendix A.4 for an analysis of fine-tuning effectiveness in Out-of-Distribution (OOD) scenarios). While both Prompt $ _{TF} $ and SelfCheckGPT achieve decent performance, Probing yields notable improvements by incorporating additional training with labels obtained from external search. Fine-Tuning further enhances performance by updating the internal features of LLMs, enabling more effective learning. In contrast, Prompt $ _{Prob} $ performs significantly worse, likely due to LLMs' tendency to output high probabilities for hallucinated claims, leading to overconfidence. Additionally, we extend the experiments to biography generation (Min et al., 2023). The results presented in Table 1 demonstrate that the observations and conclusions can be generalized to different datasets.

Building on these findings, a natural question arises: can Fine-Tuning be further improved to develop more effective hallucination detectors? We answer this question by incorporating rationales and an auxiliary task into the training process.

## 5 Rationale and Auxiliary Task Enhanced Fine-Tuning (RATE-FT)

While hallucination detection is not regarded as a reasoning task in the conventional sense, incorporating Chain-of-Thought (CoT) (Wei et al., 2022) explaining the judgment can still be beneficial for distinguishing factual content from hallucinated information as it enables LLMs to better evaluate the correctness of claims by systematically analyzing underlying components. To examine the impact of rationales, we prompt the model to generate a reasoning path before making a judgment (i.e., 'True' or 'False'), referred to as Prompt $ _{CoT-TF} $. This approach improves performance from 69.9 (using Prompt $ _{TF} $) to 74.9, highlighting the effectiveness of incorporating CoT reasoning.

Augmenting Fine-Tuning with Rationales Building on the above observation, we augment the fine-tuning dataset with rationales generated by the model during data construction, explaining

[Table 2 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 2: BAcc (%) of RATE-FT and baseline methods.

Summary: Table 2 reports balanced accuracy (BAcc %) for RATE-FT and four baseline methods (Prompt_TF, Prompt_CoT-TF, Probing, Fine-Tuning) on two datasets (LongFact and Biography). RATE-FT achieves the highest BAcc on both datasets: 79.6 on LongFact and 80.9 on Biography.

Table LaTeX:

```latex
\begin{tabular}{llllll}
\hline
Dataset & Method & Method & Method & Method & Method \\
\hline
Dataset & \$ Prompt\_\{TF\} \$ & \$ Prompt\_\{CoT-TF\} \$ & Probing & Fine-Tuning & RATE-FT \\
LongFact & 69.9 & 74.9 & 74.4 & 76.1 & 79.6 \\
Biography & 72.3 & 74.8 & 77.0 & 78.2 & 80.9 \\
\hline
\end{tabular}
```

whether the search results support the claims. Notably, we adopt the ‘label-rationale’ format to maintain the same inference cost as the baseline Fine-Tuning. This allows us to directly derive $ P_{factual} $ from the first output token without requiring the generation of the complete reasoning path.

Consolidating knowledge through repetition in diverse contexts is a fundamental principle of effective human learning (Ausubel, 2012). For example, medical students deepen their understanding of anatomy by studying diagrams, practicing in simulations, and engaging in hands-on dissections, each offering a unique perspective on the same foundational knowledge. Drawing inspiration from this paradigm, we introduce an auxiliary question answering (QA) task into the fine-tuning process to further strengthen the model's understanding and enhance its generalization capabilities. This auxiliary QA task serves as a complementary component to the primary hallucination detection task, offering the model an alternative but closely related perspective on the problem (see Appendix A.5 for more analysis on the auxiliary task).

Augmenting Fine-Tuning with QA Task Specifically, for each claim, we first prompt the model to generate a question about the key information within it. If the claim is factual, we ask the model to extract the correct answer directly from the claim and provide an explanation, forming a QA example. For hallucinated claims, we leverage the augmented rationale to guide the model in generating an appropriate correct answer along with an explanation. After constructing these QA examples, they are combined with the original data for fine-tuning.

By integrating these two strategies, we propose Rationale and Auxiliary Task Enhanced Fine-Tuning (RATE-FT) (Figure 1). RATE-FT requires the model to systematically analyze and explain its judgments and allows the model to benefit from complementary learning perspectives, reinforcing its understanding of claims through diverse yet interconnected tasks. Following the experimental setup described in Section 4, we show the comparison between RATE-FT and baseline approaches.

[Table 3 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 3: Results of different ablations.

Summary: This table presents the results of an ablation study comparing three methods (Fine-Tuning, w.o. aux, and RATE-FT) on two datasets (LongFact and Biography). The numbers are likely performance scores (e.g., accuracy or F1), with higher values indicating better performance. On LongFact, Fine-Tuning scores 76.1, w.o. aux scores 77.5, and RATE-FT scores 79.6. On Biography, Fine-Tuning scores 78.2, w.o. aux scores 79.4, and RATE-FT scores 80.9. RATE-FT achieves the highest scores on both datasets.

Table LaTeX:

```latex
\begin{tabular}{llll}
\hline
Dataset & Method & Method & Method \\
\hline
Dataset & Fine-Tuning & w.o. aux & RATE-FT \\
LongFact & 76.1 & 77.5 & 79.6 \\
Biography & 78.2 & 79.4 & 80.9 \\
\hline
\end{tabular}
```

in Table 2, which demonstrates the superiority of RATE-FT across different datasets (see Appendix A.6 for an analysis of the effect of additional data augmentation compared to the auxiliary QA task).

### 5.1 Further Analysis

Ablation Study We analyze the contribution of different components of RATE-FT by investigating the variant of RATE-FT without the auxiliary task (w.o. aux). Table 3 presents the performance of different methods, highlighting that each component plays an important role in achieving the overall performance.

Generalization to Different Models Our experiments and analysis so far use Llama-3-8B-Instruct as the backbone model. To verify whether the performance gain of RATE-FT is consistent across different backbone models, we extend the experiments to Llama-3.1-70B-Instruct (Dubey et al., 2024), Mistral-7B-Instruct (Jiang et al., 2023), and Qwen2.5-7B-Instruct (Yang et al., 2024) on LongFact (see Appendix A.7 for details on data collection). From the results shown in Table 4, we can observe that RATE-FT consistently outperforms baseline approaches across all models, demonstrating its robustness and generalizability to diverse model architectures and scales.

In addition, we provide results of incorporating uncertainty for hallucination detection, all prompts used in our experiments, and implementation details in Appendix A.8 ~ A.12, respectively.

## 6 Conclusion

In this work, we systematically investigate reference-free hallucination detection in opendomain long-form generation. Our study begins with an analysis of the model's internal states, demonstrating that these states alone cannot reliably detect hallucinations. We then evaluate several existing approaches, including prompting, probing, and fine-tuning, with fine-tuning emerging as the most effective method. Building on these findings, we introduce Rationale and Auxiliary Task Enhanced Fine-Tuning (RATE-FT), a novel approach

[Table 4 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 4: Results using different models.

Summary: Table 4 reports accuracy scores comparing three models (Llama-3.1-70B-Instruct, Mistral-7B-Instruct, Qwen2.5-7B-Instruct) across five methods: Prompt_{TF}, Prompt_{CoT-TF}, Probing, Fine-Tuning, and RATE-FT. The table shows that RATE-FT achieves the highest accuracy for all models, and Llama-3.1-70B-Instruct generally outperforms the others.

Table LaTeX:

```latex
\begin{tabular}{llllll}
\hline
Model & Method & Method & Method & Method & Method \\
\hline
Model & Prompt\_\{TF\} & Prompt\_\{CoT-TF\} & Probing & Fine-Tuning & RATE-FT \\
Llama-3.1-70B-Instruct & 73.2 & 76.8 & 79.4 & 80.6 & 83.8 \\
Mistral-7B-Instruct & 61.8 & 64.1 & 68.4 & 70.8 & 73.4 \\
Qwen2.5-7B-Instruct & 72.8 & 75.5 & 77.0 & 78.4 & 81.1 \\
\hline
\end{tabular}
```

that leverages rationales and an auxiliary task to achieve significant improvements in detection performance across two datasets and various LLMs.

## Limitations

One limitation of our work is its focus solely on improving the performance of the hallucination detector. A potential improvement could be to explore leveraging the detector's feedback as a reward signal to guide LLMs to generate more factual responses. Additionally, developing a more comprehensive benchmark for hallucination detection in open-domain long-form generation that covers a broader range of domains would further enhance its applicability.