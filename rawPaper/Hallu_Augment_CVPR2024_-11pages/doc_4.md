 $$ \begin{array}{r l}&{\mathcal{L}_{C L}^{v}=}\\ &{-\displaystyle\sum_{i=1:B+1}\frac{1}{B+1}l o g\left[\frac{f\left(\hat{e}_{v}^{i},\hat{e}_{t}^{i}\right)}{f\left(\hat{e}_{v}^{i},\hat{e}_{t}^{i}\right)+f\left(\hat{e}_{v}^{i},\hat{e}_{t}^{i}\right)+\displaystyle\sum_{k\neq i}f\left(\hat{e}_{v}^{i},\hat{e}_{t}^{k}\right)}\right]}\end{array} $$ 

For the text-to-image contrastive learning, we have not made changes and have maintained consistency with the content presented in subsection 3.1.

#### 3.3. Training Paradigm

As shown in Figure 2 (b) which demonstrates how HACL is introduced during the training process of MLLMs. Typically, we incorporate HACL into the first-stage pretraining of the model to optimize the interface  $ F_{\alpha} $ better. Therefore, suppose the loss function of text generation task is denoted as  $ L_{G} $ and the optimization object of the first stage can be defined as follow:

 $$ \mathcal{O}_{\alpha}=\arg\min_{\alpha}\mathcal{L}_{G}+\left(\mathcal{L}_{C L}^{v}+\mathcal{L}_{C L}^{t}\right)/2 $$ 

In the second stage, we follow the same approach as other methods and fine-tune the model using only instructional data.

### 4. Experiments

#### 4.1. Implementation

We validated the effectiveness of our method by applying it to four different models: miniGPT-4 [55], LLaVA [33] and LLaVA-1.5 [32].

Data sets For MiniGPT-4, the pre-training phase utilized significantly large datasets such as LAION[42] (115 million), Conceptual Captions [6] (CC3M/CC12M), and others. However, generating hallucinative captions for such enormous datasets is very costly. As a result, for MiniGPT-4, we randomly sampled about 10 million data, representing 10% of the total, and didn't use hallucinative captions for contrastive learning for the remaining data during training. Moreover, we discovered that regardless of not using hallucinative captions for enhancement, our model still significantly enhances models such as MiniGPT-4 [55]. On the other hand, for the LLaVA [33] and LLaVA1.5 [32], which used subsets of LAION/CC/SBU datasets with roughly 558K data, we generated hallucinative captions for every training datum.

Training Settings We followed the original approach for MiniGPT-4 [55] and retrained it using the complete pretraining dataset, about 10M data included hallucinative captions. For LLaVA [33] and LLaVA 1.5 [32], we used the complete pre-training dataset introduced HACL during the first stage of pre-training. We keep the same hyperparameter settings for all above models. Our experiments were conducted using 16 NVIDIA A100 GPUs with 80G of memory. Due to the increased memory usage during MLLMs training (which includes model and gradient data), the batch size during contrastive learning was affected. To address this, we used a queue of size 16,384, similar to the approaches used for ALBEF [25] and MOCO [8], to store more negative samples. We used Deepspeed [?] for LLaVA and LLaVA 1.5, with a batch size of 64 and 32 on a single GPU, respectively. For MiniGPT-4, the batch size was 8.



#### 4.2. Effectiveness of HACL on Mitigating Hallucination

To verify the efficacy of our proposed method in addressing hallucination issues, we leveraged two widely used benchmark evaluation datasets that evaluate the presence of hallucinations in models. These datasets included MMHal-Bench [44] and POPE [28]. MMHal-Bench offers a comprehensive evaluation of models that encompasses multiple perspectives, such as attributes, relations, and counting. On the other hand, POPE particularly focuses on hallucinations related to objects. We employed both datasets to measure the effectiveness of our method in addressing hallucination across various scenarios.

Evaluation on MMHal-Bench For the MMHal-Bench [44]. We apply our method to iniGPT-4 [55], LLaVA [33], LLaVA1.5 [32] and compare the results with other recent vision-language models, including MKosmos-2 [40], IDEFICS [22], InstructBLIP [10], and anther LLaVA-RLHF [44]. Following [44], we use GPT-4 to evaluate the overall score and hallucination rate of different MLLMs. Table 1 demonstrates a significant improvement in the overall performance of MMHal-Bench after applying our method to LLaVA [33], MiniGPT-4 [55], and LLaVA1.5 [32]. Notably, MiniGPT-4-HACL exhibited considerable performance gain over MiniGPT-4 [55]. Moreover, compared with LLaVA-RLHF [44], a recently proposed method that uses human feedback and reinforcement learning to address hallucinations, LLaVA-HACL showed an even more significant improvement.

Evaluation on POPE In addition, we obtained consistent results using MMHal-Bench [44] in the POPE evaluation benchmark [28]. Table 2 shows that miniGPT-4-HACL and LLaVA-HACL both demonstrated significant improvements compared to the original model. Of particular note, the average F1 score of LLaVA-HACL increased by 17.8% compared to LLaVA [33], while the Yes ratio decreased from 99.55 to 48.25. Furthermore, by applying our method to LLaVA1.5 [32], LLaVA1.5-HACL easily achieved SOTA on this benchmark. Noted that LLaVA1.5 [32] is a high-performing model with a low likelihood of generating hallucination, surpassing MiniGPT-4 [55] and LLaVA [33]. This model's impressive benchmark scores make it a valuable