A solution to better understand long passages has emerged that utilizes LLM to augment the data based on long passages and supervised fine-tuning. Data augmentation methods have been proposed to augment long passages by generating queries and extracting responses based on tasks that can be performed in long passages (Bai et al., 2024), dividing long passages into segments and generating instruction-response pairs, and generating multihop QA pairs based on multiple segments (An et al., 2024). However, many data augmentations, such as 10K and 14K, are required for supervised fine-tuning. Therefore, we try preference learning to achieve effective alignment with less data.

### 2.3 Preference Learning with LLM

Once LLM could understand and perform a variety of difficult instructions requested by humans, attention turned to aligning it with human preferences to provide more useful, less harmful, and preferred responses. Proximal Policy Optimization algorithms (PPO) (Schulman et al., 2017), which are reinforcement learning, have been used for this purpose and have shown successful performance (Bai et al., 2022; Ouyang et al., 2022). Using reinforcement learning, complex and useful behaviors can be elicited, such as the ability to discriminate useful knowledge from a long input and answer it with a pre-trained weight (Zha et al., 2023).

PPO explicitly specifies the reward model and uses it to train the model. However, labeling reward data is difficult. There are three main types of data for reward models: point-wise, pair-wise, and ranking. DPO (Rafailov et al., 2024) is a method that allows for simple learning by computing the pairwise logit between selected and rejected pairs as a reward. In our paper, we use the basic DPO methodology.

### 2.4 Synthetic Dataset for Preference Learning with LLM

There is a many of research that utilizes LLM to generate synthetic preference data. The basic way to construct a preference dataset is to give LLM a generative sampling option to extract multiple responses, and then utilize a trained reward model to select the best/worst responses based on their scores to form selected and rejected pairs. There is a way to construct a chosen-rejected pair without a reward model by requesting a reward score in LLM-as-judge (Yuan et al., 2024). Instead of requesting a reward score, you can also build a pairwise dataset by presenting the LLM with a specific criterion, such as truthfulness, and asking it to choose the better of two answers (Tian et al., 2023). Another approach, similar to our paper, is to construct completions based on contrasting positive and negative prompts (Yang et al., 2023).



## 3 Proposed Method

Our hypothesis posits that an LLM generates higher-quality responses when it reflects and integrates all relevant segments of knowledge from the given context when answering questions. In contrast, responses that exhibit confirmation bias lead to diminished quality.

In this section, we propose a method for creating a dataset that captures this hypothesis without relying on human annotation for question answering, which is then followed by DPO (Rafailov et al., 2024). See Figure 2 for an overview of our proposed method.

In §3.1, we describe the key properties of the dataset, ensuring they align with the overall hypothesis of this research. Following this, in §3.2, we introduce the dataset creation pipeline, which automatically generates datasets from a provided corpus. This section includes the processes of Chosen Response Generation, Rejected Response Generation, and application to a real world question answering dataset. Finally, in §3.3, we present the DPO method, demonstrating how the constructed dataset is used for model fine-tuning and performance optimization.

### 3.1 Dataset for Mitigating Confirmation Bias

Confirmation bias is the tendency to favor evidence that supports existing beliefs or expectations (Nickerson, 1998). We argue that confirmation bias arises in two specific forms within our task:

• Partial Evidence-Based Responses These occur when the model generates responses using only a subset of the knowledge segments from the provided context.

• Distorted Evidence-Based Responses These occur when the model produces responses that contradict or misinterpret the provided context.

Both types can lead to hallucinations, where the generated content is nonsensical or unfaithful to the original source material (Ji et al., 2023).