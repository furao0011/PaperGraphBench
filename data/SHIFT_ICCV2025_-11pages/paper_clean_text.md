<!-- page 1: doc_0.md -->

# SHIFT: Smoothing Hallucinations by Information Flow Tuning for Multimodal Large Language Models
## Abstract

Despite the remarkable progress of Multimodal Large Language Models (MLLMs) in recent years, the persistent challenge of “hallucination” has surfaced as a major barrier, sharply constraining their practical applicability and reliability in real-world systems. In this paper, we provide a novel perspective for the causes and mitigations for hallucinations by tracking the information flow within MLLMs. We find that information in MLLMs does not flow in a strictly continuous manner, instead, they may mutate abruptly in deep layers. The mutated information does not originate from shallow layers, on the contrary, it is directly injected into the model, which may cause the model's outputs to deviate from the input, leading to hallucinations. Inspired by this observation, we propose a hallucination mitigation method that directly operates on the mutated information, named Smoothing Hallucinations by Information Flow Tuning (SHIFT). In this method, the differences of feature encodings between adjacent layers are monitored, and once the mutated information is detected, the knowledge from shallow layers is used to tune it. This process filters out hallucinated knowledge, aligning features more faithfully with the input and effectively reducing hallucinations. Extensive experiments on multiple benchmarks have demonstrated the superior performance in terms of accuracy and efficiency of SHIFT on mitigating hallucinations compared with baselines.

### 1. Introduction

Recently, Multimodal Large Language Models (MLLMs) have advanced significantly in understanding and interpreting natural images, driving breakthroughs across various vision-language tasks [2, 4, 5, 9, 12, 17, 26, 34, 35, 47, 50–52, 55]. Despite the remarkable success in processing multimodal information, MLLMs still struggle with the "hallucination" challenge [16, 22, 29, 33, 36, 43]. Specifically, MLLMs may attach incorrect attributes (e.g. color, quantity) to objects in the input image, and might even fabricate non-existent objects, resulting in plausible-sounding but ridiculous responses. This phenomenon has raised concerns about the safety and accuracy of MLLMs, limiting their applications in commercial scenarios.

Existing mitigating MLLM hallucination methods mainly fall into two categories: training-based and training-free. The former attributes hallucinations to cross-modal misalignment, and thus finetunes the model using hallucination-targeted datasets or Reinforcement Learning with Human Feedback (RLHF) [3, 16, 23, 32, 41, 54]. While these approaches have shown effectiveness in reducing hallucinations, they rely heavily on manual data annotation and knowledge base integration, and the training process also incurs additional computational costs. By contrast, training-free methods address hallucinations during the inference stage, without requiring additional training costs. Currently, these methods operate at the token level, either performing contrastive decoding on token probabilities [25, 46] or penalizing over-trust tokens [19].

 

 

Current methods do not consider how hallucinated information appears during the model’s processing, which

<!-- page 2: doc_1.md -->

leads to reliance on multiple sampling or rollback strategies to correct token probabilities, affecting both accuracy and computational efficiency. Building on this insight, this paper investigates the transmission of information in MLLMs, identifying and filtering hallucinated knowledge based on changes in the information flow, thereby mitigating hallucinations from an information-level perspective. We design an effective strategy to track the differences in information between adjacent layers within the model, and find that for hallucinated responses, the model often undergoes significant information mutations in deep layers. We suspect that these inconsistent knowledge does not originate from the input image, and instead is injected directly into the model, which influences the probabilities of hallucinated tokens. To further validate this hypothesis, we analyze the probability changes of hallucinated tokens across different layers. The results reveal that these tokens' probabilities often surge in layers with drastic information changes, becoming dominant and ultimately appearing in the final response, thereby causing hallucinations.

Inspired by the above observation, we propose a novel Smoothing Hallucinations by Information Flow Tuning (SHIFT) method, which identifies knowledge “suddenly” injected into the model according to the changes in information between layers, and smooths it using the continuous knowledge transmitted from earlier layers. This operation eliminates hallucinated information from the model, making the responses more aligned with the input image. By directly removing hallucinations from the information flow, our method offers superior hallucination mitigation capabilities compared to token-based ones. Moreover, SHIFT requires calculating the response only once, without the need for contrastive decoding or rollback, making it faster and less disruptive to the model’s normal inference process. Our main contributions can be summarized as follows:

• We deeply analyze the transmission of information in MLLMs, and first reveal that hallucinated information does not originate from the shallow layers but is suddenly injected into the deeper layers by the models themselves.

- A novel hallucination mitigation method named SHIFT is proposed, which identifies potential hallucinated information through the information discrepancy in intermediate layers and tunes it using continuous knowledge, effectively eliminating hallucinations.

• Extensive experiments are conducted on multiple benchmarks, showing that SHIFT achieves state-of-the-art performance on various datasets and decoding methods.

### 2. Related Work

#### 2.1. Multimodal Large Language Models

The rapid advancement of Large Language Models (LLMs) [6, 10, 13, 15, 42, 44] has propelled the emergence of Multimodal Large Language Models (MLLMs), which aligns the features from input images with text prompts. Consequently, MLLMs are enabled to understand and generate multimodal information with impressive accuracy and versatility across diverse applications [1, 7, 8, 13, 18]. An MLLM typically contains three main modules: a modality encoder, an LLM, and a modality interface connecting the encoder and LLM. The modality encoder [27, 37] extracts and aligns multimodal features from the input data, such as images and texts. The LLM processes the multimodal features and generates the final response. The modality interface is responsible for linking the vision-text encoder output with the LLM by projecting the multimodal information into the space that LLM can understand efficiently.

#### 2.2. Hallucinations in MLLMs

The hallucination phenomenon has been widely observed in the field of language models [21, 24, 30, 40, 53], where researchers find that these models often produce responses with factual inaccuracies or not aligned with the input prompt, referred as factual and faithful hallucinations, respectively. When it comes to MLLMs, they may generate answers contradicting the input images [29, 38], indicating that they are prone to faithful hallucinations. To mitigate hallucinations in MLLMs, training-based methods optimize model architectures or training strategies, and finetune them using hallucination-specific datasets [3, 23, 29, 38, 49]. Some approaches also employ annotation enrichment techniques to guide the models in aligning with human instructions [16, 28, 32, 41]. While these methods show effectiveness in reducing hallucinations, they introduce extra costs for requiring additional annotations and training, limiting their practicality. In contrast, training-free methods directly address hallucinations in the inference stage. For example, [25, 46] introduce different contrastive decoding strategies that contrast the original and distorted inputs or instructions, then the statistical bias can be calibrated through the decoding process. OPERA [19] introduces a penalty term on the model logits for over-trust tokens during decoding, along with a rollback strategy that retrospects the presence of summary tokens in the previously generated tokens. The above methods directly manipulate the probability of individual tokens, requiring multiple requests to obtain the final results. In this paper, we offer a novel perspective by analyzing the information flow in MLLMs, allowing the model to directly locate and eliminate hallucinations during inference, achieving superior performance and efficiency.

### 3. Method

In this section, we first analyze the transmission procedure of information in MLLMs to distinguish between hallucinated and correct knowledge, then provide a detailed introduction to the proposed method.

<!-- page 3: doc_10.md -->

gren Zhou. mplug-owi2: Revolutionizing multi-modal large language model with modality collaboration. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13040–13051, 2023. 1, 5, 6, 7, 8

[48] Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, and Enhong Chen. Woodpecker: Hallucination correction for multimodal large language models. arXiv preprint arXiv:2310.16045, 2023. 5

[49] Yan Zeng, Xinsong Zhang, and Hang Li. Multi-grained vision language pre-training: Aligning texts with visual concepts. In International Conference on Machine Learning (ICML), 2021. 2

[50] Pan Zhang, Xiaoyi Dong Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Hang Yan, et al. Internlxcomposer: A vision-language large model for advanced text-image comprehension and composition. arXiv preprint arXiv:2309.15112, 2023. 1

[51] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601, 2023.

[52] Yichi Zhang, Ziqiao Ma, Xiaofeng Gao, Suhaila Shakiah, Qiaozi Gao, and Joyce Chai. Groundhog grounding large language models to holistic segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14227–14238, 2024. 1

[53] Chunting Zhou, Graham Neubig, Jiatao Gu, Mona T. Diab, Francisco Guzmán, Luke Zettlemoyer, and Marjan Ghazvininejad. Detecting hallucinated content in conditional neural sequence generation. In Findings of the Association for Computational Linguistics (ACL), pages 1393–1404, 2021. 2

[54] Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, and Huaxiu Yao. Analyzing and mitigating object hallucination in large vision-language models. In International Conference on Learning Representations (ICLR), 2024. 1

[55] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In International Conference on Learning Representations (ICLR), 2024. 1

<!-- page 4: doc_2.md -->

#### 3.1. Information Flow across Layers

Early exiting [14, 39] has been proven to be effective in observing the hidden representations in language models [11]. [45] employs this tool to devise a series of methods for tracking information flow, leading to the observation of several intriguing phenomena. Inspired by that, we begin by investigating the origin of hallucinations. In MLLMs, the aligned features from multimodal encoders are first embedded into a sequence of vectors $ V_0 = \{v_1^0, ..., v_{n-1}^0\} $, we apply the affine function $ \phi(\cdot) $ on each hidden transformer block. Denoting the output of the $ i $-th layer as $ V_i $, then the next token predicted with it can be calculated as

 $$ p_{i}(x_{t}|x_{0,\ldots,t-1})=softmax(\phi(V_{i}))_{x_{t}}, $$ 

where $ i \in \{0, ..., N-1\} $. The probability distribution of all candidate tokens $ p_i(\cdot | x_0, ..., t-1) $ in the vocabulary set $ X $ represents the information contained in the $ i $-th layer. We calculate the Jensen-Shannon Divergence (JSD) between the token probability distributions of two adjacent layers, which provides an intuitive display for the information difference between layers. It can be formulated as:

 $$ \begin{align*}d(p_{i}(\cdot|x_{0,\ldots,t-1}),p_{j}(\cdot|x_{0,\ldots,t-1}))\\=JSD(p_{i}(\cdot|x_{0,\ldots,t-1})||p_{j}(\cdot|x_{0,\ldots,t-1})),\end{align*} $$ 

where i and j are two adjacent layers. JSDs for the hallucinated example in 

We refer to the layers exhibiting significant JSD changes as mutation layers. To analyze the nature of injected information, we trace the probability evolution of affected tokens

 

 

 

 

 

across layers. For hallucinated predictions, such as “man” are shown in Figure 3(a), the token initially has low probability, with “woman” as the top prediction. After 27-th layer, the probability of “man” rises sharply and surpasses “woman”, indicating that contradictory information is injected at this stage, ultimately leading to hallucination. In contrast, for correct predictions (Figure 3(b)), probability fluctuations remain within semantically related tokens (e.g., “boy” and “child”), suggesting that the injected information is correlated with the original context and does not impair prediction accuracy. Among 1206 manually identified hallucinated tokens, 1062 exhibit mutations in deep layers, indicating that hierarchical mutation is a widespread phenomenon in MLLMs. Our analysis reveals a consistent pattern: when the injected knowledge contradicts prior context, hallucinations emerge; when it is semantically aligned, the model maintains faithful outputs by preserving contextual consistency.

 

 

Furthermore, as shown in Figure 5, we examine attention maps at the mutation layers and observe substantial shifts in attention distribution. Compared to earlier layers, attention in deeper layers drifts from the original semantics, influenced by the injected concepts. This redistribution suggests that mutation layers not only cause token-level deviations but also reshape the model's internal visual grounding. These findings support our hypothesis that mutation layers serve as integration points for new information, with their impact depending on the semantic alignment between the injected knowledge and the model's prior context.

New dominant tokens may emerge in mutation layers, falling into four categories: both original and new tokens are correct (Type 1), correct-to-hallucinated (Type

<!-- page 5: doc_3.md -->

2), hallucinated-to-correct (Type 3), and both hallucinated (Type 4). We randomly sample 100 images from the MSCOCO dataset [31] and tally these cases. As shown in Table 1, among the changed tokens (Types 2 and 3), correct tokens are more often replaced by hallucinations than vice versa, indicating that mutation layers tend to introduce hallucinated content. While not all mutations lead to hallucinations, most hallucinated tokens stem from such mutations. This suggests that smoothing mutation layers with information from preceding layers can help retain correct tokens and suppress hallucinations.

 

 

#### 3.2. Smoothing Hallucinations by Information Flow Tuning (SHIFT)

Inspired by the analyses above, we propose to tune the information in the mutation layers with the continuous information from earlier layers. As illustrated in Figure 7, this approach ensures that the visual information extracted by the shallow layers can be effectively transmitted to the deep layers, thereby reducing hallucinated information.

Assuming we are currently predicting the t-th token, the predicted distributions of all layers for the output token are

 

 

calculated with the affine layer, denoted as

 $$ P_{l}=p_{i}(\cdot|x_{0},...,x_{t-1}),l\in[0,N-1], $$ 

where $ P_{l} $ is the probability distribution of the l-th layer. After that, the JSD between the probability distributions of any two adjacent layers is calculated, then the mutation layer $ L_{mutation} $ with the maximum JSD is selected by

 $$ L_{mutation}=\underset{l^{*}<l<N-1}{\arg\max}JSD(P_{l-1}||P_{l}), $$ 

where the $ l^{*} $-th layer is the boundary of the hierarchical phenomenon in 

 $$ \begin{aligned}L_{mutation}^{*}=\{l\mid\exists\epsilon,\delta>0such that JSD(P_{l-1}||P_{l})>\epsilon\\ and\left|\frac{JSD(P_{l}||P_{l+1})-JSD(P_{l-1}||P_{l})}{JSD(P_{l-1}||P_{l})}\right|>\delta\right\},\end{aligned} $$ 

where $ \epsilon $ and $ \delta $ are two controlling parameters.

For any retained mutation layer, adjustments need to be made using the continuous information transmitted from the shallow layers to reduce the potential hallucinations caused by the injected knowledge. Since the JSD values typically converge before the mutation layer, we choose the layer preceding the mutation layer to smooth it. It is important to note that this operation is performed on the encoding vectors before applying the affine layer, rather than on the token probabilities. The feature vectors are computed as

 $$ V_{l}^{*}=\begin{cases}V_{l}&if l<L_{mutation}^{*}\\ \alpha\cdot V_{l-1}+(1-\alpha)\cdot V_{l}&if l=L_{mutation}^{*}\\ V_{l-1}&if l>L_{mutation}^{*}\end{cases}, $$

<!-- page 6: doc_4.md -->

where $ V_l $ is the feature vector of the $ l $-th layer, $ \alpha $ is an adjustment parameter, and $ f(\cdot) $ is the transformer block. With the greedy decoding, the prediction of the next token $ x_t^* $ is

 $$ x_{t}^{*}=\underset{x_{t}\in X}{\arg\max}softmax(\phi(F(V_{l^{*}}^{*}_{\mathrm{m u t a t i o n}})))_{x_{t}}, $$ 

where $ F(\cdot) $ is the collection of the transformer blocks after the $ L_{mutation}^{*} $-th layer.

### 4. Experiment

#### 4.1. Setup

##### 4.1.1. Models & Baselines

SHIFT is evaluated on three MLLMs, including LLaVA-1.5 [34], mPLUG-Owl2 [47], and InstructBlip [12]. For more comprehensive comparisons, we choose three decoding methods, including greedy, beam search, and nucleus sampling. Greedy decoding selects tokens with the highest probability in logits step by step. Beam search maintains a set of beams and selects the best token from them. Nucleus sampling focuses on the most significant probability mass at each time step by keeping a limited subset of the vocabulary. Besides the basic decoding methods, we also consider OPERA [19], which is implemented on beam search, and the contrastive decoding methods VCD [25] and ICD [46], which are improvements on nucleus sampling.

##### 4.1.2. Benchmark & Evaluation Metrics

CHAIR [38] The Caption Hallucination Assessment with Image Relevance (CHAIR) assesses hallucinations on the image captioning task. CHAIR works by generating ground-truth object labels for each image, and any object mentioned absent from the label set is deemed a hallucinated object. It comprises two metrics: sentence-level $ (CHAIR_{S}) $ and image-level $ (CHAIR_{I}) $, calculated as:

 $$ \mathrm{CHAIR}_{S}=\frac{\left|\left\{\mathrm{hallucinated\ objects}\right\}\right|}{\mathrm{all\ mentioned\ objects}}, $$ 

 $$ \mathrm{CHAIR}_{I}=\frac{\left|\left\{\text{captions with hallucinated objects}\right\}\right|}{\text{all captions}}. $$ 

We conduct CHAIR evaluation on the MSCOCO dataset [31], with the prompt “Please describe this image in detail”. Following the setup in [19], we randomly sample 500 images from the validate set for test.

POPE [29] The Polling-based Object Probing Evaluation (POPE) is a widely used benchmark for identifying object-level hallucinations. It utilizes a question-answering format such as "Is there a {} in the image", prompting MLLMs to assess whether a specified object is present in the image. We adopt two subsets for evaluation, including MSCOCO and GQA [20]. There encompasses three sampling settings for each subset: random, popular, and adversarial. MMHal-Bench [32] MMHal-Bench is another VQA-based evaluation on object hallucinations. Different from POPE, its questions contain logical considerations, which is more challenging for MLLMs. This benchmark includes 8 types of high-difficulty questions about object attributes: adversarial objects, comparisons, counting, spatial relations, environment, holistic descriptions, and others.

GPT-4v Assisted Evaluation CHAIR can only evaluate hallucinations based on the presence of objects in descriptions, struggling to assess physical attributes, positions, and other aspects. To provide a more comprehensive evaluation of hallucinations, following [19, 48], we use a GPT-4v-based assessment strategy. Specifically, we randomly sample 500 images from the MSCOCO dataset, and ask MLLMs to describe them. For each sample, the image together with responses from the vanilla decoding method and SHIFT are fed into GPT-4v, which is prompted to give judgements from 0-10 to assess the responses' qualities.

#### 4.2. Quantitive Results

##### 4.2.1. Results on CHAIR

We test SHIFT's performance when generating long sentences on the CHAIR benchmark, and the results are present in 

##### 4.2.2. Results on POPE

The experimental results on the POPE benchmark are shown in

<!-- page 7: doc_5.md -->

operate with any decoding method and effectively reduce hallucinations. For instance, on the MSCOCO dataset, SHIFT achieves accuracy improvements of up to 5.59%, 5.43%, and 7.6% compared to the original LLaVA-1.5, mPLUG-Owl2, and InstructBlip models under different settings, with F1 scores also improving by 4.31%, 4.24%, and 5.87%. Additionally, as the difficulty increases from random to popular and then to adversarial settings, models experience noticeable performance drops. Nevertheless, our method continues to enhance the model's resistance to hallucinations, demonstrating its robustness in challenging scenarios. Compared to other hallucination mitigation methods, SHIFT still demonstrates superior performance across different datasets and settings. Taking the MSCOCO dataset as an example, SHIFT achieves maximum accuracy and F1 score improvements of up to 2.6%/1.97%, 4.1%/2.34%, and 0.94%/0.89% across the three models compared to OPERA, which is based on Beam Search. In comparison with the nucleus-based VCD and ICD methods, SHIFT shows performance advantages of 3.8%/2.3%, 4.08%/2.88%, and 11.53%/6.76%. Notably, as the setting difficulty increases, SHIFT's performance lead becomes even more pronounced, further highlighting its enhanced capability to handle complex tasks compared to baselines.

##### 4.2.3. Results on MMHal-Bench

The performance of SHIFT is tested on more challenging and comprehensive VQA scenarios with the MMHal-Bench benchmark. In this experiment, we first answer questions about the input image, and responses are scored by GPT-4 based on ground-truth answers. The average scores of the vanilla MLLMs and our proposed methods are shown in 

 

 

tasks, and incorporating SHIFT can improve the hallucination mitigation capability for the vanilla MLLMs.

##### 4.2.4. Results on GPT-4v Assisted

We further use GPT-4v to evaluate the quality of descriptions generated by SHIFT. Two metrics are considered, including correctness (C) and detailedness (D). As shown in Table 4, the proposed method outperforms the vanilla models up to 0.84% and 0.24% in terms of correctness and detailedness, respectively, indicating that SHIFT can provide accurate and detailed descriptions even on more challenging hallucination evaluation scenarios. Given GPT-4V's perception and reasoning abilities, which closely resemble those of humans, the evaluation results based on it provide a meaningful indication of SHIFT's effectiveness in reducing hallucinations from a human-centered perspective.

##### 4.2.5. Efficiency Comparisons

In addition to accuracy, time complexity is also an essential metric for evaluating hallucination mitigation methods. The lower the complexity, the smaller its impact on the normal inference. We compare the inference time and mem

<!-- page 8: doc_6.md -->



<!-- page 9: doc_7.md -->

ory usage of various methods on LLaVA-1.5, with the results shown in 

 

 

#### 4.3. Visualization of Token Probabilities

To more intuitively demonstrate SHIFT's hallucination mitigation capabilities, we visualize the probability changes of hallucinated tokens before and after applying our method in 

### 5. Conclusion

This paper deeply analyzes information flow within MLLMs, revealing information mutations in deep layers by examining differences of representation between adjacent layers. The mutations may contain hallucinated knowledge, potentially leading the model to produce outputs not faithful to the input. Therefore, we propose SHIFT, which aims to mitigate hallucinations by tuning the information flow. This method identifies mutation layers by tracking the differences in information across layers, and smooths out the injected knowledge using continuous information from the previous layers. Experimental results demonstrate SHIFT's superior hallucination mitigation performance across various benchmarks and metrics.

<!-- page 10: doc_8.md -->

## References

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 2

[2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 1

[3] Ali Furkan Biten, Lluís Gómez, and Dimosthenis Karatzas. Let there be a clock on the beach: Reducing object hallucination in image captioning. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1381–1390, 2022. 1, 2

[4] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations (ICLR), 2024. 1

[5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18392–18402, 2023. 1

[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems (NeurIPS), 2020. 2

[7] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 2

[8] Xi Chen, Xiao Wang, Soravit Changpinyo, A. J. Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish V. Thapliyal, James Bradbury, and Weicheng Kuo. Pali: A jointly-scaled multilingual language-image model. In International Conference on Learning Representations (ICLR), 2023. 2

[9] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Intern v1: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24185–24198, 2023. 1

[10] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90% $ ^{*} $ chatgpt quality, 2023. 2

[11] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R. Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models. In International Conference on Learning Representations (ICLR), 2024. 3

[12] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 1, 5, 6, 7, 8

[13] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. Palm-e: An embodied multimodal language model. In International Conference on Machine Learning (ICML), pages 8469–8488, 2023. 2

[14] Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael Auli. Depth-adaptive transformer. ArXiv, 2019. 3

[15] Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. Chat-gpt outperforms crowd workers for text-annotation tasks. National Academy of Sciences of the United States of America, 2023. 2

[16] Anisha Gunjal, Jihan Yin, and Erhan Bas. Detecting and preventing hallucinations in large vision language models. In AAAI Conference on Artificial Intelligence (AAAI), pages 18135–18143, 2024. 1, 2

[17] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13504–13514, 2024. 1

[18] Qidong Huang, Xiaoyi Dong, Dongdong Chen, Weiming Zhang, Feifei Wang, Gang Hua, and Nenghai Yu. Diversity-aware meta visual prompting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10878–10887, 2023. 2

[19] Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. OPERA: alleviating hallucination in multimodal large language models via over-trust penalty and retrospection-allocation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13418–13427, 2024. 1, 2, 5, 6, 7, 8

[20] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 6700–6709, 2019. 5

[21] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023. 2

[22] Prannay Kaul, Zhizhong Li, Hao Yang, Yonatan Dukler, Ashwin Swaminathan, CJ Taylor, and Stefano Soatto. Throne: An object-based hallucination benchmark for the free-form generations of large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27218–27228, 2024. 1

<!-- page 11: doc_9.md -->

[23] Jae-Myung Kim, A. Sophia Koepke, Cordelia Schmid, and Zeynep Akata. Exposing and mitigating spurious correlations for cross-modal retrieval. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)- Workshops, pages 2585–2595, 2023. 1, 2

[24] Katherine Lee, Orhan Firat, Ashish Agarwal, Clara Fannjiang, and David Sussillo. Hallucinations in neural machine translation. OpenReview, 2018. 2

[25] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13872–13882, 2024. 1, 2, 5, 6, 7, 8

[26] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 1

[27] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning (ICML), pages 12888–12900, 2022. 2

[28] Lei Li, Yuwei Yin, Shicheng Li, Liang Chen, Peiyi Wang, Shuhuai Ren, Mukai Li, Yazheng Yang, Jingjing Xu, Xu Sun, et al. A large-scale dataset towards multi-modal multilingual instruction tuning. arXiv preprint arXiv:2306.04387, 2023. 2

[29] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 292–305, 2023. 1, 2, 5

[30] Stephanie C. Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In Annual Meeting of the Association for Computational Linguistics (ACL), 2021. 2

[31] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision (ECCV), 2014. 4, 5

[32] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal model with robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023. 1, 2, 5

[33] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Mitigating hallucination in large multi-modal models via robust instruction tuning. In International Conference on Learning Representations (ICLR), 2024. 1

[34] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 1, 5, 6, 7, 8

[35] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning.

In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26286–26296, 2024. 1

[36] Holy Lovenia, Wenliang Dai, Samuel Cahyawijaya, Ziwei Ji, and Pascale Fung. Negative object presence evaluation (nope) to measure object hallucination in vision-language models. arXiv preprint arXiv:2310.05338, 2023. 1

[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. 2

[38] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018. 2, 5

[39] Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Tran, Yi Tay, and Donald Metzler. Confident adaptive language modeling. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 3

[40] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih. Replug: Retrieval-augmented black-box language models. In North American Chapter of the Association for Computational Linguistics (NAACL), 2023. 2

[41] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liangyan Gui, Yu-Xiong Wang, Yiming Yang, Kurt Keutzer, and Trevor Darrell. Aligning large multimodal models with factually augmented RLHF. In Findings of the Association for Computational Linguistics (ACL), pages 13088–13110, 2024. 1, 2

[43] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9568–9578, 2024. 1

[44] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2

[45] Sudong Wang, Yunjian Zhang, Yao Zhu, Jianing Li, Zizhe Wang, Yanwei Liu, and Xiangyang Ji. Towards understanding how knowledge evolves in large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 29858–29868, 2025. 3

[46] Xintong Wang, Jingheng Pan, Liang Ding, and Chris Biemann. Mitigating hallucinations in large vision-language models with instruction contrastive decoding. In Findings of the Association for Computational Linguistics (ACL), pages 15840–15853, 2024. 1, 2, 5, 6, 7, 8

[47] Qinghao Ye, Haiyang Xu, Jiabo Ye, Mingshi Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jin-