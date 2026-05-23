<!-- page 1: doc_0.md -->

# Hallucination Augmented Contrastive Learning for Multimodal Large Language Model
## Abstract

### 1. Introduction

Large Language Models (LLMs) like GPT-3 [4], LLaMA [45, 46], and GPT-4 [39] have received significant attention for their remarkable text understanding and generation

 

 

 

 

capabilities. Recently, GPT-4V $ ^{1} $ [38] has demonstrated impressive multi-modal abilities in tasks such as image caption and visual question answering, shedding light on the vision-language domain and attracting widespread research interests. Consequently, a new category of models, known as Multi-modal Large Language Models (MLLMs) [2, 10, 27, 33, 49–51, 55], has emerged, aiming to enhance

<!-- page 2: doc_1.md -->

LLMs with the capacity to comprehend and handle visual information. To integrate natural language with other modalities, MLLMs incorporate a learnable interface between pre-trained visual encoders and LLMs. Such interface includes learnable query tokens [10, 27, 51, 55] or a projection-based linear model [32, 33] that extracts and integrates information from visual modalities. MLLMs learn this interface to generate answers for multimodal instructions, resulting in remarkable performance in many multimodal tasks.

However, a fundamental limitation of MLLMs is their tendency to produce erroneous or fabricated information that doesn't match the provided visual input, known as hallucination [28, 31, 44, 47]. In this paper, we aim to tackle the issue from the perspective of representation learning. We first check the distribution of textual and visual tokens within the representation space of LLMs (Vicuna [54] in our experiments), in which visual representations are projected by the learned interface. As shown in Figure 1, we have two primary findings:

• A significant modality gap remains between the textual and visual tokens despite visual projection;

• Representations of texts that contain and do not contain hallucinations are entangled, making it challenging to differentiate them.

These preliminary observations indicate that the current learned interfaces are not effective enough to map visual representations into the textual representation space of LLMs. As a result, it is difficult for MLLMs to discriminate between texts containing minor errors at the level of objects or attributes and those manifesting typical hallucinative expressions. This issue potentially heightens the tendency for MLLMs to generate more hallucinations. Therefore, exploring more effective approaches to align visual representations with LLMs' textual representation space to address hallucinations is crucial.

Inspired by the findings above, we propose hallucination-augmented cross-modal contrastive learning (HACL), which enhances the alignment between visual and textual representations to alleviate hallucinations. Texts with hallucination are used as hard negative examples for image anchors, naturally pulling closer representations of non-hallucinating text and visual samples while pushing way representations of non-hallucinating and hallucinating text. Specifically, we separately feed the visual and textual token sequences into LLMs to obtain global representations for each modality, which is used for contrastive learning. We generate hallucinative image captions with GPT-4 [39]. These hallucinative texts contain partial object attribute errors or introduce additional non-existent information compared to the original image captions. As shown in 

• We underline a significant cross-modal semantic gap between visual and textual representations and an unexpected representation tangling among text containing and not containing hallucinations in MLLMs. These findings expose the inadequacies of current methodologies in efficiently bridging the gap between visual and textual representations.

• Based on these insights, we propose a simple yet effective method named Hallucination Augmented Cross-Modal Contrastive Learning (HACL). Introducing contrastive learning into MLLMs and using hallucinative text as hard negative samples yields a better cross-modal and more hallucinations-distinguishable representation space.

• Our experiments show that equipping MLLMs with HACL not only minigates hallucinations but also effectively improve the performance across multiple benchmark evaluations.

### 2. Related Work

Multi-Modal Large Language Foundation Models. The successful application of Large Language Models (LLMs) has paved the way for developing several approaches aiming to augment the perceptual capacities of LLMs with additional modalities, all within a unified model. There are three primary methods for constructing multi-modal large language foundational models, each showing promise for robust zero-shot generalization capabilities in the vision-language domain. For instance, Flamingo [1] is a forerunner in this area, using a frozen vision encoder and a large language model equipped with gated cross-attention for cross-modality alignment. In contrast, PaLM-E [11] integrates extracted visual features directly through linear layers into the pre-trained PaLM [9] model, which boasts 520 billion parameters, thereby leading to robust performance across numerous real-world applications. This approach has been broadly adopted by models such as LLaVA [33], Shikra [7], etc. One significant limitation of this method, however, is the creation of lengthy visual sequences. To address this, BLIP-2 [27], drawing inspiration from DETR [5], developed a Q-former to reduce the sequence length of visual features efficiently. This design has been mirrored by Kosmos-1 [17], mPLUG-Owl [51], and MiniGPT-4 [55].

<!-- page 3: doc_10.md -->

[40] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. ArXiv, abs/2306.14824, 2023. 5, 6

[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3

[42] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 5

[43] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 6

[44] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liangyan Gui, Yu-Xiong Wang, Yiming Yang, Kurt Keutzer, and Trevor Darrell. Aligning large multimodal models with factually augmented rlhf. ArXiv, abs/2309.14525, 2023. 1, 2, 3, 5, 6, 7

[45] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. ArXiv, abs/2302.13971, 2023. 1, 6

[46] Hugo Touvron, Louis Martin, Kevin R. Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Daniel M. Bikel, Lukas Blecher, Cristian Cantón Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony S. Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel M. Kloumann, A. V. Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, R. Subramanian, Xia Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zhengxu Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. ArXiv, abs/2307.09288, 2023.

[47] Bin Wang, Fan Wu, Xiao Han, Jiahui Peng, Huaping Zhong, Pan Zhang, Xiao wen Dong, Weijia Li, Wei Li, Jiaqi Wang, and Conghui He. Vigc: Visual instruction generation and correction. ArXiv, abs/2308.12714, 2023. 2, 3

[48] Jinyu Yang, Jiali Duan, Son Tran, Yi Xu, Sampath Chanda, Liqun Chen, Belinda Zeng, Trishul Chilimbi, and Junzhou

Huang. Vision-language pre-training with triple contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15671–15680, 2022. 4

[49] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-docowl: Modularized multimodal large language model for document understanding. CoRR, abs/2307.02499, 2023. 1

[50] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

[51] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023. 1, 2, 6, 7

[52] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 6, 7

[53] Yan Zeng, Xinsong Zhang, and Hang Li. Multi-grained vision language pre-training: Aligning texts with visual concepts. arXiv preprint arXiv:2111.08276, 2021. 4

[54] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. 2

[55] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. ArXiv, abs/2304.10592, 2023. 1, 2, 5, 6, 7, 8

<!-- page 4: doc_2.md -->

Minigating Hallucination for MLLMs. To address the issue of hallucination in MLLMs, researchers have developed various methods, which can be broadly categorized into two lines. The first line [30, 47] involve limiting the length of instruction data, which typically leads to a reduction in hallucination. For instance, LRV-Instruction[30] takes an intuitive approach by constraining the text length of instructions and constructing counterfactual instructions. However, this may result in less detailed descriptions from the fine-tuned model. The second line utilizes additional artificial data or tools to modify hallucinations in the model's output. For example, LLaVA-RLHF [44] employs manually annotated data as reward signals to guide the model in generating less hallucinative responses. Although effective, this approach requires extra manual annotation data. In this paper, we propose a method from the perspective of representation learning. We introduce hallucinative captions as hard negative samples in contrastive learning, aiming to narrow the gap between visual representations and correct textual representations, while pushing away from hallucinative textual representations. This approach effectively addresses the issue of hallucination and also enhances the model's visual understanding capability.

### 3. Method

The learnable interface of MLLMs plays a vital role in bridging diverse modalities and mapping visual representations to the representation space of LLMs. Our goal is to refine this interface to facilitate better matching of visual representations with the ground truth text in the representation space, while also increasing the distance between them and hallucinative text. To accomplish this, we propose a new approach called Hallucination Augmented Cross-modal Contrastive Learning (HACL). This approach is inspired by contrastive learning, which is a well-established technique in the fields of representation learning [37] and self-supervised learning [8, 16, 21, 41]. In the following subsection, we first introduce how to incorporate cross-modal contrastive learning during training. Next, we describe how to boost contrastive learning through additional generated hallucinative captions. Finally, we introduce the hallucination augmented contrastive learning training paradigm.

#### 3.1. Cross-modal Contrastive Learning

As shown in

<!-- page 5: doc_3.md -->

as $ \mathbf{F}_\alpha $, and a decoder-only based Large Language Model denoted $ \mathbf{L}_\beta $ where $ \theta $, $ \alpha $, $ \beta $ represent the parameters of each module. Additionally, we also have an unsupervised pretraining dataset, containing N image-text pairs, denoted as $ D = \{I_i, T_i\} $, $ i \in [1, 2, \ldots, N] $.

Assuming an image $I_i$ is processed by the vision encoder $\mathbf{V}_\theta$ and the learnable interface $\mathbf{F}_\alpha$, it is transformed into a visual token sequence of length $m$. Since most LLMs are decoder-only models, in order to obtain the representations that can capture global semantic information. We pass a $<EOS> token through an embedding layer $\mathbf{L}_\beta$ to obtain the vector representation $e \in \mathbf{R}^D$ and append it to this sequence. Thus, the new visual token sequence becomes $S_v^i = [v_1^i, v_2^i, \ldots, v_m^i, e_v^i]$, where $v_k^i \in \mathbb{R}^D$, $k \in [1, 2, /dots, m]$. Similarly, for the caption paired with this image, we also append an $<EOS> token to the text token sequence and pass it through the embedding layer of the LLM to obtain the text embedding sequence $S_t^i = [t_1^i, t_2^i, \ldots, t_n^i, e_t^i]$, where $t_k^i \in \mathbb{R}^D$, $k \in [1, 2, /dots, n]$. Subsequently, the visual embedding sequence $S_v$ and the text embedding sequence $S_v$ are individually passed through the LLM $\mathbf{L}_\beta$ to obtain the final output from the last layer of $\mathbf{L}_\beta$ as following:

 $$ H_{t}^{i}=\mathbf{L}_{\beta}\left(S_{t}^{i}\right) $$ 

 $$ H_{v}^{i}=\mathbf{L}_{\beta}\left(S_{v}^{i}\right) $$ 

where $ H_v^i = \begin{bmatrix} \hat{v}_1^i, \hat{v}_2^i, \ldots, \hat{v}_m^i, \hat{e}_v^i \end{bmatrix} $ and $ H_t^i = \begin{bmatrix} \hat{t}_1^i, \hat{t}_2^i, \ldots, \hat{t}_n^i, \hat{e}_t^i \end{bmatrix} $. Afterwards, we obtain the global representation $ \hat{e}_v^i $ that captures the overall semantic information of the image $ I_i $, as well as the global representation $ \hat{e}_t^i $ that captures the overall semantic information of the ground truth caption $ T_i $.

Afterwards, similar to many existing methods in the field of vision-language pretraining [3, 18–20, 25, 26, 48, 53], we introduce the following contrastive learning strategy. Assuming a batch size of B during the training process, we compute the text-to-image contrastive learning loss as follows:

 $$ \mathcal{L}_{C L}^{t}=-\sum_{i=1:B}\frac{1}{B}l o g\left\lfloor\frac{f\left(\hat{e}_{t}^{i},\hat{e}_{v}^{i}\right)}{f\left(\hat{e}_{t}^{i},\hat{e}_{v}^{i}\right)+\sum_{k\neq i}f\left(\hat{e}_{t}^{i},\hat{e}_{v}^{k}\right)}\right\rfloor $$ 

where $f\left(\hat{e}_{t}^{i},\hat{e}_{v}^{i}\right)$ measures the distance between $\hat{e}_{t}^{i}$ and $\hat{e}_{v}^{i}$ in a semantic space. Similar, the image-to-text contrastive learning loss as follows:

 $$ \mathcal{L}_{C L}^{v}=-\sum_{i=1:B}\frac{1}{B}l o g\left[\frac{f\left(\hat{e}_{v}^{i},\hat{e}_{t}^{i}\right)}{f\left(\hat{e}_{v}^{i},\hat{e}_{t}^{i}\right)+\sum_{k\neq i}f\left(\hat{e}_{v}^{i},\hat{e}_{t}^{k}\right)}\right] $$ 

#### 3.2. Improving Contrastive Learning with Hallucinative Captions

We propose to improve the effectiveness of contrastive learning by introducing hard negative samples which mimic the hallucinative text generated by MLLMs.

 

 

Generation of Hallucinative Captions In order to do this, we utilize GPT-4 [39] to incorporate some elements into the ground truth captions that are either inconsistent with the image content or completely absent from it. As shown in Figure 3, these hallucinations can be coarse-grained, focusing on the presence of objects, or fine-grained, focusing on specific attributes such as quantity, properties, or locations. Here is our prompt to GPT-4:

Hallucination in Large-scale Visual Language Models (LVLMs) refers to cases where these models generate descriptions introducing elements that are inconsistent with the content or completely absent from a provided image. These hallucinations can be coarse-grained, focusing on the mere existence of objects, or fine-grained, focusing on more specific attributes or characteristics such as quantity, properties, and locations. Your task is to revise a given caption to create a mirrored version that closely aligns with the original's content and length but incorporates elements of hallucination. The first step involves identifying the objects involved and their associated attributes within the given caption. Subsequently, combine this insight with the details concerning hallucinations provided above to complete your task.

To improve the generation of more appropriate hallucinative captions, we also provide some contextual examples for GPT-4. Please check our appendix for more details.

Hallucination Augmented Contrastive Learning Assuming that we have generated an hallucinative caption $ \hat{T}_i $ based on the original caption $ T_i $ for the image $ I_i $, and obtained the global representation $ \hat{e}_t^i $ of the hallucinative caption using the approach described in subsection 3.1, we can treat it as a negative sample in the image-text contrastive learning. Therefore, the new formula for the image-to-text contrastive learning becomes:

<!-- page 6: doc_4.md -->

$$ \begin{array}{r l}&{\mathcal{L}_{C L}^{v}=}\\ &{-\displaystyle\sum_{i=1:B+1}\frac{1}{B+1}l o g\left[\frac{f\left(\hat{e}_{v}^{i},\hat{e}_{t}^{i}\right)}{f\left(\hat{e}_{v}^{i},\hat{e}_{t}^{i}\right)+f\left(\hat{e}_{v}^{i},\hat{e}_{t}^{i}\right)+\displaystyle\sum_{k\neq i}f\left(\hat{e}_{v}^{i},\hat{e}_{t}^{k}\right)}\right]}\end{array} $$ 

For the text-to-image contrastive learning, we have not made changes and have maintained consistency with the content presented in subsection 3.1.

#### 3.3. Training Paradigm

As shown in 

 $$ \mathcal{O}_{\alpha}=\arg\min_{\alpha}\mathcal{L}_{G}+\left(\mathcal{L}_{C L}^{v}+\mathcal{L}_{C L}^{t}\right)/2 $$ 

In the second stage, we follow the same approach as other methods and fine-tune the model using only instructional data.

### 4. Experiments

#### 4.1. Implementation

We validated the effectiveness of our method by applying it to four different models: miniGPT-4 [55], LLaVA [33] and LLaVA-1.5 [32].

Data sets For MiniGPT-4, the pre-training phase utilized significantly large datasets such as LAION[42] (115 million), Conceptual Captions [6] (CC3M/CC12M), and others. However, generating hallucinative captions for such enormous datasets is very costly. As a result, for MiniGPT-4, we randomly sampled about 10 million data, representing 10% of the total, and didn't use hallucinative captions for contrastive learning for the remaining data during training. Moreover, we discovered that regardless of not using hallucinative captions for enhancement, our model still significantly enhances models such as MiniGPT-4 [55]. On the other hand, for the LLaVA [33] and LLaVA1.5 [32], which used subsets of LAION/CC/SBU datasets with roughly 558K data, we generated hallucinative captions for every training datum.

Training Settings We followed the original approach for MiniGPT-4 [55] and retrained it using the complete pretraining dataset, about 10M data included hallucinative captions. For LLaVA [33] and LLaVA 1.5 [32], we used the complete pre-training dataset introduced HACL during the first stage of pre-training. We keep the same hyperparameter settings for all above models. Our experiments were conducted using 16 NVIDIA A100 GPUs with 80G of memory. Due to the increased memory usage during MLLMs training (which includes model and gradient data), the batch size during contrastive learning was affected. To address this, we used a queue of size 16,384, similar to the approaches used for ALBEF [25] and MOCO [8], to store more negative samples. We used Deepspeed [?] for LLaVA and LLaVA 1.5, with a batch size of 64 and 32 on a single GPU, respectively. For MiniGPT-4, the batch size was 8.

#### 4.2. Effectiveness of HACL on Mitigating Hallucination

To verify the efficacy of our proposed method in addressing hallucination issues, we leveraged two widely used benchmark evaluation datasets that evaluate the presence of hallucinations in models. These datasets included MMHal-Bench [44] and POPE [28]. MMHal-Bench offers a comprehensive evaluation of models that encompasses multiple perspectives, such as attributes, relations, and counting. On the other hand, POPE particularly focuses on hallucinations related to objects. We employed both datasets to measure the effectiveness of our method in addressing hallucination across various scenarios.

Evaluation on MMHal-Bench For the MMHal-Bench [44]. We apply our method to iniGPT-4 [55], LLaVA [33], LLaVA1.5 [32] and compare the results with other recent vision-language models, including MKosmos-2 [40], IDEFICS [22], InstructBLIP [10], and anther LLaVA-RLHF [44]. Following [44], we use GPT-4 to evaluate the overall score and hallucination rate of different MLLMs. 

Evaluation on POPE In addition, we obtained consistent results using MMHal-Bench [44] in the POPE evaluation benchmark [28].

<!-- page 7: doc_5.md -->

foundation to build upon.

#### 4.3. Effectiveness of HACL on Visual Comprehension

HACL has shown effectiveness in solving the issue of hallucination. Nevertheless, we intend to explore the influence of HACL on the model's abilities of visual comprehension and generation. To achieve this objective, we carried out assessments on common benchmarks, such as Visual Question Answering (VQA) [15, 36, 43] after incorporating HACL into the MLLMs. Furthermore, as MLLMs possess robust zero-shot capabilities, traditional evaluation metrics often fail to provide a detailed assessment of their abilities. Additionally, their inability to match the given answer correctly exacerbates significant robustness issues. To mitigate these challenges, the research community introduced a series of benchmarks. These benchmarks aim to systematically structure and evaluate complex multi-modal tasks from various perspectives. Therefore, we also evaluated the model's performance on recently designed MLLM-focused Multi-modal Benchmarks including MME [12], MMBench [34], MM-Vet [52], SEED-Bench [23].

Results on Benchmark Tasks Our evaluation includes six popular benchmarks, as summarized in 

MLLM-oriented Multi-modal Benchmarks. We applied HACL to MiniGPT-4 [55], LLaVA [33], LLaVA1.5 [32] and evaluate them on five recently popular multi-modal benchmarks in a zero-shot manner. For a fair comparison, we select models with similar language model sizes, particularly those from the LLaMA [45] family, and detail their differences in the vision encoder. The results of our evaluation are listed in

<!-- page 8: doc_6.md -->

#### 4.4. Ablation Study

 

 

 

 

Impact of Hallucinative Captions To validate the effectiveness of using hallucinative captions as hard negative samples in contrastive learning for resolving hallucinations, we conducted the following experiments: In the Stage 1 pre-training phase, we did not introduce any additional hal

<!-- page 9: doc_7.md -->

lucinative captions, and the contrastive learning loss was calculated solely based on the equations 3 and 9 discussed in subsection 3.1 of our paper. We conducted experiments on MLLMs including LLaVA [33], MiniGPT-4 [55], and LLaVA1.5 [32], and reported the results on benchmarks such as POPE and MMHal-Bench. Additionally, we also reported results on the MME and VQA benchmarks. As illustrated in the Table 5, absent the facilitation from hallucinative captions, the models displayed moderate improvements on hallucination benchmarks such as MMhal-Bench, yet these improvements were somewhat constrained. However, the subsequent inclusion of hallucinative captions resulted in a marked enhancement on the same hallucination benchmark, thus affirming the potency of the hallucinative captions. Furthermore, we observed analogous improvements in the model's performance on both MME and VQA. Our hypothesis asserts that hallucinative captions aid MLLMs in diverting the visual representation from hallucinations and other textual inaccuracies. This action helps avoid instances of hallucination. Furthermore, contrastive learning supports the model by aligning the semantics of image-text, which ultimately enhances the model's effectiveness.

Discussion on Training Paradigm We have observed that certain Multimodal Language-and-Vision Models (MLLMs) may not freeze the activity of either the visual encoder or the Language-and-Vision Models (LLMs) during the initial stage of pretraining. To assess the impact of our methodology under such distinct training paradigms, we independently tested models where either the Visual Encoder or the LLMs were active during the first pretraining phase. These tests were conducted on two platforms: LLaVA and LLaVA1.5 and subsequently evaluated against multiple benchmark standards. As illustrated in Table 6, the models experienced a significant performance decline when LLMs are activated. We hypothesize that this downturn could be linked to low-quality data in the first pretraining stage and the introduction of additional contrast learning tasks, both of which affect the LLMs' representation distribution. This culminates in the catastrophic forgetting of the LLMs. Conversely, initiating the Visual Encoder led to a modest performance boost. This might be attributed to the fact that the target parameters our model can optimize extend beyond the learnable interface and incorporate the visual encoder as well. This expanded scope paves the way for a more successful alignment of visual and text representations within the MLLMs.

#### 4.5. Visualization

The objective of our research is the introduction of HACL, to further enhance the visual representation output of our interface. The aim is to closely align the output to the correct textual representation within the representation space of Language Models (LLMs) and, at the same time, distance it from hallucinative and other incorrect textual representations. To substantiate our objective, we randomly selected 200 image-text pairs from the COCO [29] val2017 dataset. Using GPT-4, we generated hallucination samples and subsequently reduced these samples using the hidden state representation of the last token through LLMs for visualization purposes. The data distribution under three conditions: without employing HACL, instigating cross-modal contrastive learning but without the use of hallucination-enhanced samples, and usage of hallucination-enhanced sample contrast learning was visualized respectively. The MLLM utilized in our study was LLaVA. As illustrated in

<!-- page 10: doc_8.md -->

learning, although the modal gap decreased, a differentiation in the distribution of hallucination samples and ground truth samples was unattainable. In 

### 5. Conclusion

This paper addresses the issue of hallucinations in Multi-modal Large Language Models (MLLMs) and proposes a method called Hallucination Augmented Contrastive Learning (HACL) to improve the alignment between visual and textual representations. By using contrastive learning on projected text and visual token sequences, and incorporating hallucinative captions as hard negative samples, HACL effectively reduces the occurrence of hallucinations. Experimental results demonstrate that incorporating HACL enhances the performance of MLLMs and significantly reduces the occurrence of hallucinations in benchmark evaluations.

<!-- page 11: doc_9.md -->

Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790, 2023. 6

[15] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 6

[16] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020. 3

[17] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, and Furu Wei. Language is not all you need: Aligning perception with language models. ArXiv, abs/2302.14045, 2023. 2

[18] Chaoya Jiang, Haiyang Xu, Chenliang Li, Ming Yan, Wei Ye, Shikun Zhang, Bin Bi, and Songfang Huang. TRIPS: Efficient vision-and-language pre-training with text-relevant image patch selection. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 4084–4096, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. 4

[19] Chaoya Jiang, Haiyang Xu, Wei Ye, Qinghao Ye, Chenliang Li, Ming Yan, Bin Bi, Shikun Zhang, Fei Huang, and Songfang Huang. Bus: Efficient and effective vision-language pretraining with bottom-up patch summarization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2900–2910, 2023.

[20] Chaoya Jiang, Haiyang Xu, Wei Ye, Qinghao Ye, Chenliang Li, Ming Yan, Bin Bi, Shikun Zhang, Fei Huang, and Ji Zhang. Copa: Efficient vision-language pre-training through collaborative object-and patch-text alignment. In Proceedings of the 31st ACM International Conference on Multimedia, pages 4480–4491, 2023. 4

[21] Chaoya Jiang, Wei Ye, Haiyang Xu, Miang yan, Shikun Zhang, Jie Zhang, and Fei Huang. Vision language pretraining by contrastive learning with cross-modal similarity regulation. In Annual Meeting of the Association for Computational Linguistics, 2023. 3

[22] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. Obelics: An open web-scale filtered dataset of interleaved image-text documents, 2023. 5, 6

[23] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 6, 7

[24] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. ArXiv, abs/2305.03726, 2023. 7

[25] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:9694–9705, 2021. 4, 5

[26] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022. 4

[27] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. ArXiv, abs/2301.12597, 2023. 1, 2, 6, 7

[28] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji rong Wen. Evaluating object hallucination in large vision-language models. ArXiv, abs/2305.10355, 2023. 2, 5, 6, 7

[29] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 8

[30] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal model with robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023. 3

[31] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Mitigating hallucination in large multi-modal models via robust instruction tuning, 2023. 2

[32] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. ArXiv, abs/2310.03744, 2023. 2, 5, 6, 7, 8

[33] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. ArXiv, abs/2304.08485, 2023. 1, 2, 5, 6, 7, 8

[34] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 6, 7

[35] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. ArXiv, abs/2206.08916, 2022. 7

[36] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pages 947–952. IEEE, 2019. 6

[37] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 3

[38] OpenAI. Gpt-4v(ision) system card. 2023. 1

[39] OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023. 1, 2, 3, 4