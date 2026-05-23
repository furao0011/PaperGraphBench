
<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Dataset</td><td colspan="3">Partition</td><td style='text-align: center; word-wrap: break-word;'>Total</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>CMHE-HD</td><td style='text-align: center; word-wrap: break-word;'>Generated 387</td><td style='text-align: center; word-wrap: break-word;'>Tampered 613</td><td style='text-align: center; word-wrap: break-word;'>Correct 1,000</td><td style='text-align: center; word-wrap: break-word;'>2,000</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>CMHE-DD</td><td style='text-align: center; word-wrap: break-word;'>Chat 327</td><td style='text-align: center; word-wrap: break-word;'>Exam 1,295</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>1,622</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>CMHE-CE</td><td style='text-align: center; word-wrap: break-word;'>Medicine 1,753  $ \times $ 4</td><td style='text-align: center; word-wrap: break-word;'>Disease 7,096  $ \times $ 4</td><td style='text-align: center; word-wrap: break-word;'>Checkup 1,060  $ \times $ 2</td><td style='text-align: center; word-wrap: break-word;'>38,576</td></tr></table>

<div style="text-align: center;">Table 1: The statistics of each sub-category in the CMHE dataset.</div>


### 4. Experiments

#### 4.1. Datasets

Our proposed CMHE benchmark comprises three individual datasets: CMHE-HD for hallucination detection, CMHE-DD for disease diagnosis, and CMHE-CE for concept explanation. We have performed calculations to determine the number of samples within each fine-grained subset of these datasets, and the resulting statistics are presented in Table 1.

#### 4.2. Baselines

Three well-known LLMs are examined to assess their performance in detecting medical hallucinations in Chinese text. All three models are capable of processing input in Chinese.

(1)ChatGPT $ ^{4} $ is a large generative language model created by OpenAI, which can generate human-like texts based on past conversations. We exploit GPT-3.5 as the backbone of ChatGPT in our experiments. (2) Baichuan(Yang et al., 2023) in the second version is a series of large-scale multilingual language models containing 7 billion and 13 billion parameters trained from scratch. The Baichuan2-13B chat is utilized in our evaluations. (3) Qwen (Bai et al., 2023) is a collection of language models that includes different models with different numbers of parameters. In our evaluation of the baseline models, we rely on Qwen-14B Chat as the foundation.

For all of LLMs used in our experiments, the hyperparameters are set as follows. The temperature is set to 0.5, the Top-P is 0.7, the Top-K is 200, and the repetition penalty is 1.1.

### 5. Results and Analysis

In this section, the experimental results of three baseline systems in Chinese medical consultation are evaluated from three aspects, i.e., disease diagnosis, concept understanding, and error identification with the CMHE-HD, CMHE-DD and CMHE-CE datasets, respectively.


<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Model</td><td style='text-align: center; word-wrap: break-word;'>Generated</td><td style='text-align: center; word-wrap: break-word;'>Tampered</td><td style='text-align: center; word-wrap: break-word;'>Correct</td><td style='text-align: center; word-wrap: break-word;'>All</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>ChatGPT</td><td style='text-align: center; word-wrap: break-word;'>29.2</td><td style='text-align: center; word-wrap: break-word;'>49.5</td><td style='text-align: center; word-wrap: break-word;'>59.8</td><td style='text-align: center; word-wrap: break-word;'>50.8</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Baichuan</td><td style='text-align: center; word-wrap: break-word;'>8.0</td><td style='text-align: center; word-wrap: break-word;'>42.7</td><td style='text-align: center; word-wrap: break-word;'>80.7</td><td style='text-align: center; word-wrap: break-word;'>55.0</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Qwen</td><td style='text-align: center; word-wrap: break-word;'>2.0</td><td style='text-align: center; word-wrap: break-word;'>25.3</td><td style='text-align: center; word-wrap: break-word;'>97.4</td><td style='text-align: center; word-wrap: break-word;'>56.9</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Random</td><td style='text-align: center; word-wrap: break-word;'>50.0</td><td style='text-align: center; word-wrap: break-word;'>50.0</td><td style='text-align: center; word-wrap: break-word;'>50.0</td><td style='text-align: center; word-wrap: break-word;'>50.0</td></tr></table>

<div style="text-align: center;">Table 2: Performances of the mainstream medical large language models on CMHE-HD dataset. Note that 'Generated', 'Modified', and 'Correct' denote that data partition with various generation mode. 'All' denotes the whole dataset. 'Random' denotes that the results are generated randomly and accuracy is used as the metrics.</div>


#### 5.1. Performance on Hallucination Detection

We used the CMHE-HD dataset to evaluate how well different models can identify different types of hallucinations in three distinct datasets: "Generated", "Tampered", and "Correct". The "Generated" dataset includes content that is either nonexistent or irrational, whereas the "Tampered" dataset features examples of contextual inconsistencies. In contrast, the "Correct" dataset acts as a control group with no hallucinations. The results of our experiments are presented in Table 2.

In terms of both "Generated" and "Tampered" data, ChatGPT exhibits superior performance compared to the other two models, indicating its proficiency in detecting various types of hallucinations. Particularly in the Generated data, ChatGPT outperforms the other models by a significant margin, demonstrating its unmatched ability to identify knowledgeable hallucinations. Among the three models, Qwen achieves the highest performance on the "Correct" data, followed by Baichuan in second place, and ChatGPT ranks third. All three models outperform the random model by a considerable margin. By comparing the performances of the "Generated", "Tampered", and "Correct" data, we can speculate that the Qwen consistently rejects hallucinations, while ChatGPT tends to try and identify them even when they don't exist.

Additionally, all models perform worse on "Generated" data compared to "Tampered" data. This indicates that detecting knowledgeable hallucinations is more difficult than identifying context inconsistencies. LLMs often miss hallucinations caused by inconsistent contexts.

#### 5.2. Performance on Disease Diagnosis

As shown in Table 3, the labels "Diagnose-chat" and "Diagnose-exam" indicate the origin of the data from different sources. The "chat" data consists of dialogues from real-life scenarios, which may contain excessive information that could potentially distract models. On the other hand, the "Exam"