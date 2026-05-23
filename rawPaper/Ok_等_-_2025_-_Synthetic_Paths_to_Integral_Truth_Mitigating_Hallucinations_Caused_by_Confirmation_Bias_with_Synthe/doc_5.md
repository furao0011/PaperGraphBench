nance and medicine. Although HaluBench primarily targets hallucination detection, we adapt it to prompt models for answer generation. The model-generated answers are then evaluated for hallucination.

We create three derived datasets/settings from NQ-Open to isolate and analyze the effects of context length, passage count, and the presence of multiple relevant passages on model hallucination.

#### 4.1.2 Derived Settings from NQ-Open

Lost in the Middle Setting In the original Liu et al. (2024) dataset, both the number of passages and the gold passage's position were varied extensively, resulting in a large number of data points. We adapt their original setting, observing how increasing the context size (number of passages) and altering the gold passage position may cause the model to overlook the correct passage, potentially leading to hallucinated answers.

• We increase the context size up to 40 passages.

Scaling with the Number of Passages Setting To investigate the effect of increasing the number of passages on hallucination, we modify the original dataset as follows:

• We randomize the gold passage position to focus solely on the effect of passage count rather than fixed ordering.

• To examine confirmation bias, we tag each passage using Mixtral-8x7B-Instruct-v0.1 to identify passages (other than the gold one) that could provide the answer.

This yields a dataset containing not only the gold passage but also multiple passages relevant to the query. Thus, we can analyze how scaling the total number of passages influences hallucination patterns.

Scaling with the Number of Relevant Passages

Setting Using the 40-passage dataset described above, we further refine the data to investigate how multiple relevant passages affect confirmation bias:

• Single: Only one passage is relevant.

• Multiple: Two or more passages are relevant.

For each category, we sample 500 examples. Unlike the previous setting, the model here only receives the relevant passages at inference time. This approach allows us to understand:

• In the Single scenario, whether the model distorts information even with a single relevant passage.



• In the Multiple scenario, whether the model selectively uses only some of the relevant evidence, demonstrating confirmation bias.

#### 4.1.3 Model and Training Details

We use the Llama-3-8B-Instruct $ ^{\ddagger} $, trained with a context length of 8,192 tokens, as our baseline. We apply direct preference optimization (DPO) fine-tuning on this model using our constructed datasets (details are provided in the Appendix).

#### 4.1.4 Evaluation Metrics

We employ the following metrics to assess model performance, hallucinations, and confirmation bias:

Accuracy Following Liu et al. (2024), accuracy measures whether the generated answer includes any correct solution.

Knowledge F1 (KF1) Based on Shuster et al. (2021), KF1 measures unigram overlap between the generated answer and the gold knowledge segment.

LLM as Judge Following Zheng et al. (2023), we use an LLM (Mixtral-8x7B-Instruct-v0.1) to assess whether the generated response is grounded in the provided context. Instead of simple lexical overlap, this method attempts to mimic human evaluation. For fairness, the LLM sees only the query-related context segments.

Lynx Score For the HaluBench dataset, we utilize the Lynx model (Ravi et al., 2024), trained to detect hallucinations by verifying the answer's faithfulness to the given document and question. We refer to this metric as the Lynx score, using an 8B model. This score complements the LLM as Judge approach and provides a specialized, model-based hallucination detection measure for the HaluBench domain.

### 4.2 Results in Question Answering

Table 1 presents a comparison between our model and the baseline, Llama-3-8B-Instruct. Our model outperforms Llama-3-8B-Instruct across all passage counts in the Scaling with the Number of Passages. With a single passage, our model achieves 92.32 accuracy, significantly surpassing the baseline's 84.56. This advantage persists even with 40