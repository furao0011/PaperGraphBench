# Multi-Granular Multimodal Clue Fusion for Meme Understanding
## Abstract

With the continuous emergence of various social media platforms frequently used in daily life, the multimodal meme understanding (MMU) task has been garnering increasing attention. MMU aims to explore and comprehend the meanings of memes from various perspectives by performing tasks such as metaphor recognition, sentiment analysis, intention detection, and offensiveness detection. Despite making progress, limitations persist due to the loss of fine-grained metaphorical visual clue and the neglect of multimodal text-image weak correlation. To overcome these limitations, we propose a multigranular multimodal clue fusion model (MGMCF) to advance MMU. Firstly, we design an object-level semantic mining module to extract object-level image feature clues, achieving fine-grained feature clue extraction and enhancing the model's ability to capture metaphorical details and semantics. Secondly, we propose a brand-new global-local cross-modal interaction model to address the weak correlation between text and images. This model facilitates effective interaction between global multimodal contextual clues and local unimodal feature clues, strengthening their representations through a bidirectional cross-modal attention mechanism. Finally, we devise a dual-semantic guided training strategy to enhance the model's understanding and alignment of multimodal representations in the semantic space. Experiments conducted on the widely-used MET-MEME bilingual dataset demonstrate significant improvements over state-of-the-art baselines. Specifically, there is an 8.14% increase in precision for offensiveness detection task, and respective accuracy enhancements of 3.53%, 3.89%, and 3.52% for metaphor recognition, sentiment analysis, and intention detection tasks. These results, underpinned by in-depth analyses, underscore the effectiveness and potential of our approach for advancing MMU.

## Introduction

Memes, as a popular form of online communication, express viewpoints, sentiments, and intentions in a concise and humorous manner. With the development of social networks, Multimodal Meme Understanding (MMU) (Wang et al. 2024; Xu et al. 2022), as an emerging research area in Natural Language Processing (NLP), plays a crucial role in many downstream applications, such as question answering (Zheng et al.

[Figure 1 was here. The original paper contained a figure at this position. Brief visual description: Figure 1 displays two examples of metaphorical memes. Panel (a) shows a man wearing a slice of pizza as a makeshift face mask, accompanied by the text 'HOW ITALIANS FIGHT CORONA VIRUS'. Panel (b) features a scene from the movie *Wonder Woman* showing a dance between two characters, overlaid with the quote 'Peace is only an armistice in an endless war' attributed to Thucydides, and the caption 'Wonder Woman Knows Thucydides. Do you?'.]
Caption: Figure 1: Examples of Metaphorical Memes.
Key visible elements:
- Panel (a): Left subfigure containing a food-based meme
- Man with pizza mask: Central visual subject in panel (a)
- Caption 'HOW ITALIANS FIGHT CORONA VIRUS': Top text providing context for panel (a)
- Panel (b): Right subfigure containing a pop-culture/philosophy meme
- Movie characters (Diana & Steve): Central visual subjects in panel (b)
- Quote text overlay: Text explaining the philosophical theme in panel (b)
- Yellow arrows: Visual indicators pointing to the characters in panel (b)

(2024b) and sentiment analysis (Zheng et al. 2023a,b). The definition of the MMU task involves predicting understanding from four dimensions: metaphor, sentiment, intention, and offensiveness. However, memes are nuanced, and accurately grasping the underlying meaning embedded within the combination of text and images poses a crucial challenge.

Several studies have made commendable efforts in MMU. Kiela et al. (2020); Kirk et al. (2021) introduced multimodal hate meme datasets specifically designed for hate detection. However, these studies overlooked the crucial aspect of metaphorical features in memes. Therefore, Xu et al. (2022) considered the richer metaphorical features in memes and constructed a baseline model and a bilingual dataset called MET-MEME for this purpose. Furthermore, Wang et al. (2024) proposed a metaphor-aware multimodal multitask framework on this dataset to capture the interactions between text and images. Despite achieving notable success, current researches in this field face two significant limitations: 1) the loss of fine-grained metaphorical visual clues and 2) the neglect of multimodal text-image weak correlation. These limitations hinder its further flourishing and widespread adoption.

On the one hand, existing works (Qu et al. 2023; Ji, Ren, and Naseem 2023) exhibit a lack of emphasis on images and simply encode broad visual representations at the image-level, ignoring metaphorical clues at the fine-grained object-level of images. This neglect leads to a critical absence of key visual metaphorical details, resulting in semantic ambiguity and omissions, ultimately failing to comprehensively capture

the complexity and diversity of memes. As shown in Figure 1 (a), encoding visual features solely at the image-level falls short in capturing the crucial metaphorical clues of a pizza being used as a mask. Detecting the presence of metaphors and accurately predicting the conveyed sentiments, intentions, and potential offensiveness becomes exceedingly challenging in such cases.

On the other hand, existing methods (Xu et al. 2022; Wang et al. 2024) primarily focus on directly integrating textual and visual information to comprehend memes, overlooking the issue of weak correlations between modalities and disregarding the intrinsic crucial metaphorical clues within each modality. This oversight leads to the loss of crucial details and clues, thereby limiting a comprehensive understanding of memes. For instance, in the illustrated example in Figure 1 (b), there is a weak correlation between the image and text, where the image conveys peace while the text reveals hatred towards war. Merely concatenating and fusing the image and text information directly could lead to a misinterpretation of the meme as peaceful.

In this paper, motivated by the aforementioned observations, we propose a Multi-Granular Multimodal Clue Fusion model (MGMCF) to improve multimodal meme understanding. First, we design an object-level semantic mining module to extract fine-grained object-level feature clues from images. We then integrate these object-level feature clues with the overall image-level feature clues to obtain a multi-granular representation. This enables our model to better capture the metaphorical details and semantics of images, offering a more comprehensive visual understanding. Second, considering the weak correlation between text and images, we not only focus on the interactions between different modalities but also emphasize the crucial metaphorical clues within each modality, integrating multi-granular clues to enhance the ultimate understanding of multimodal memes. Therefore, we propose a novel global-local cross-modal interaction model to enable effective interaction between the global multimodal contextual clues and local unimodal feature clues. Specifically, the global multimodal context enhances the local unimodal features through a symmetric cross-modal attention mechanism. This interaction process is bidirectional, allowing the global context to extract useful clues from the local unimodal features and update itself. Through multi-level stacking, the global multimodal context and local unimodal features mutually enhance each other and gradually improve. Moreover, to enhance semantic alignment, we devise a dual-semantic guided training strategy. By bringing related image-text pairs closer in the forward direction and pushing unrelated pairs apart in the reverse direction, we aim to foster a more robust understanding of complex multimodal semantic clues.

To verify the effectiveness of our model, we conduct experiments on the benchmark MET-MEME bilingual dataset (Xu et al. 2022), which contains both English and Chinese memes. The results demonstrate that our model significantly outperforms all state-of-the-art (SoTA) baselines across all evaluation metrics in the four tasks. On the English meme dataset, the precision in the offensiveness detection task increased by 8.14%, and the accuracy in metaphor recognition, sentiment analysis, and intention detection tasks improved by 3.53%, 3.89%, and 3.52%, respectively. Additionally, extensive experiments validate the effectiveness of our fine-grained visual information enhancement and global-local interactions.

Our main contributions are summarized as follows:

• We analyze and summarize two intrinsic challenges in the MMU task and propose a multi-granular multimodal clue fusion model, the first to consider fine-grained visual clues and integrate unimodal feature clues to enhance MMU.

• We design an object-level semantic mining module and a global-local cross-modal interaction model to facilitate effective interaction between global multimodal clues and local unimodal clues, achieving multi-granular understanding of meme metaphorical clues and semantics.

• Our extensive experimental results on MET-MEME dataset demonstrate that our scheme achieves state-of-the-art performance on the MMU task.

## Related Work

## Multimodal Meme Understanding

Recently, multimodal meme understanding (Lin et al. 2024; Hee, Chong, and Lee 2023; Qu et al. 2023; Fang et al. 2024b, 2023, 2024c, 2025; Zheng et al. 2024a) has attracted increasing attention. Unlike general multimodal learning tasks (Ji et al. 2021, 2022; Li et al. 2022b,a; Wu et al. 2023; Li et al. 2023a; Fei et al. 2024a; Luo et al. 2024), meme understanding relies more heavily on contextual and metaphorical information. Existing research has mainly focused on hateful memes. Yang et al. (2023) proposed a scalable invariant and specific modality representation learning framework based on graph neural networks for harmful meme detection. Ji, Ren, and Naseem (2023) introduced a prompt-based method to identify harmful memes. Beyond just focusing on hateful meme detection, Xu et al. (2022) introduced a fine-grained multimodal meme understanding dataset, which includes tasks such as sentiment analysis, intention detection, offensiveness detection, and metaphor recognition, to analyze memes at a finer granularity. Wang et al. (2024) created cross-modal and intra-modal attention mechanisms on this dataset to capture the interactions between text and images for multimodal meme understanding. However, existing works overlook the object-level fine-grained clues in images, potentially leading to the loss of crucial metaphorical visual details. Moreover, these methods do not address the issue of cross-modal weak correlations, neglecting the essential clues within unimodal, which result in semantic ambiguity and confusion, failing to fully capture the complexity and diversity of memes.

## Metaphorical Information Processing

In the field of NLP, there has been growing interest in developing models for metaphor detection (He et al. 2024; Fang et al. 2024a; Elzohbi and Zhao 2024; Zhang and Liu 2023). Understanding the essence of memes relies critically on identifying the metaphorical information embedded within them. Existing researches (Qiao, Zhang, and Ma 2024; Elzohbi and Zhao 2024; Badathala et al. 2023) have primarily focused on unimodal metaphor detection. Zhang and Liu (2023) proposed a novel multi-task learning framework based on a metaphor

[Figure 2 was here. The original paper contained a figure at this position. Brief visual description: Figure 2 illustrates the overall architecture of a multimodal model designed for meme understanding. The system processes text inputs (source/target domains and literal text) and image inputs (full images and detected objects) through separate encoders. These features are fused and processed through a 'global-local interaction' module involving cross-modal attention blocks ($CAP_{G-T}$ and $CAP_{G-I}$) across multiple layers. Finally, the model aggregates these features to predict four downstream tasks: metaphor recognition (mr), sentiment analysis (sa), intention detection (id), and offensiveness detection (od).]
Caption: Figure 2: The overall architecture of our model. “mr” means metaphor recognition, “sa” means sentiment analysis, “id” means intention detection, “od” means offensiveness detection.
Key visible elements:
- Source/Target Domain & Text Inputs: Provides semantic context for metaphor and literal meaning.
- Text Encoder: Encodes textual information into feature vectors.
- Image & Detected Object Inputs: Provides visual context including global scene and local details.
- Coarse/Fine-grained Encoders: Extracts visual features from full images and cropped objects respectively.
- Global-Local Interaction Module: Core processing unit combining text and image features via CAP blocks.
- CAP Blocks ($CAP_{G-T}$, $CAP_{G-I}$): Facilitates cross-modal attention and feature fusion.
- Task Outputs ($y_{mr}, y_{sa}, y_{id}, y_{od}$): Final predictions for metaphor, sentiment, intention, and offensiveness.

recognition program, a set of linguistic rules. Li et al. (2023b) performed metaphor detection by explicitly modeling the basic meanings of concepts. Tian et al. (2023) designed a domain contrastive learning strategy to capture the semantic inconsistencies. While these unimodal metaphor detection methods have achieved promising results, there has been relatively less exploration in the area of multimodal metaphor detection. Alnajjar, Hämäläinen, and Zhang (2022) introduced a multimodal metaphor annotated corpus and designed a video-text content-based method for metaphor detection. He et al. (2024) developed a multi-interactive cross-modal residual network for multimodal metaphor recognition.

## Methodology

## Task Definition

This paper addresses the task of multimodal meme understanding, encompassing metaphor recognition, sentiment analysis, intention detection, and offensiveness detection. Specifically, given an example consisting of an image I, its corresponding text T, a source domain $ T_s $, and a target domain $ T_a $, the objective of multimodal meme analysis is to predict the categories of metaphor $ y_{mr} $, sentiment $ y_{sa} $, intention $ y_{id} $, and offensiveness $ y_{od} $. As shown in Figure 2, the source domain serves as the basis for the metaphor, while the target domain embodies the concept or idea metaphorically conveyed, typically in textual form.

## Feature Extraction

Text Encoder. In accordance with the approach described in Wang et al. (2024), we utilize Multilingual BERT (Wang et al. 2019) to extract textual features from the corresponding text $ x^{t} $, source domain $ x^{s} $, and target domain $ x^{a} $. The encoding process can be formulated briefly as:

$$ \begin{aligned}\{\boldsymbol{h}_{1}^{t},...,\boldsymbol{h}_{n}^{t}\}&=M-B E R T(\{\boldsymbol{x}_{1}^{t},...,\boldsymbol{x}_{n}^{t}\})\\\{\boldsymbol{h}_{1}^{s},...,\boldsymbol{h}_{p}^{s}\}&=M-B E R T(\{\boldsymbol{x}_{1}^{s},...,\boldsymbol{x}_{p}^{s}\})\\\{\boldsymbol{h}_{1}^{a},...,\boldsymbol{h}_{q}^{a}\}&=M-B E R T(\{\boldsymbol{x}_{1}^{a},...,\boldsymbol{x}_{q}^{a}\})\end{aligned} $$

where n, p, and q represent the word counts of the corresponding text, source domain, and target domain, respectively.

## Image Encoder

Multimodal meme images contain rich metaphorical details that are crucial clues for understanding memes. Therefore, when extracting visual features from memes, we cannot solely focus on image-level visual semantic clues as with other multimodal tasks. It is imperative to capture object-level fine-grained clue features that encompass these metaphorical details. To achieve this goal, we devise a visual information enhancement strategy for extracting feature clues of different granularities.

For image I, following (Wang et al. 2024), we first employ a pretrained convolutional neural network classifier, VGG16 (Simonyan and Zisserman 2014), to extract image-level features $ \boldsymbol{h}^{c} = \boldsymbol{V} \boldsymbol{G} \boldsymbol{G} \boldsymbol{1} \boldsymbol{G} \boldsymbol{1} \boldsymbol{G} $. Then, we transform the input image I into a series of embedded blocks to capture fine-grained image features. By integrating object detection, attribute recognition, and positional information, we enrich the representation of image features and enhance enhance image metaphor comprehension. Specifically, we design an object-level semantic mining module (Anderson et al. 2018) to identify and localize objects in an image. For each visual region $ I_{i} $ represented by a bounding box, we resize the region to a standard size of $ 224 \times 224 $ pixels. Following Xu, Zeng, and Mao (2020), we reshape the resized region $ I_{i} $ into a sequence $ I_{i} = \{r_{1}, \ldots, r_{m}\} $, where each region is represented by a

block. This reshaping divides the region into a grid of blocks, with m being the total number of blocks. Next, we flatten each block $ r_j $ and project it into a $ d^I $-dimensional vector. This projection is performed using a trainable linear projection matrix E, and the resulting embedded representation of block $ r_j $ is denoted as $ z_j = r_j E $. To incorporate contextual information and retain positional information, we prepend a [class] token embedding at the beginning of the patch sequence. Position embeddings are also appended to the patch embeddings, indicating their relative positions within the sequence. The input representation of each visual region $ I_i $ is expressed as:

$$ \boldsymbol{Z}_{i}=[\boldsymbol{z}_{[class]};\boldsymbol{z}_{1},...,\boldsymbol{z}_{m}]+\boldsymbol{E}_{pos} $$

where $ Z_i $ represents the input matrix of image patches, and $ E_{pos} $ denotes the position embedding matrix. Subsequently, we feed the input matrix $ Z_i $ into the VGG16 encoder to obtain the visual region $ I_i $ representation $ h_i^v = VGG16(Z_i) $. Finally, the representation of the image I is defined as:

$$ h_{I}=\{h^{c},h_{1}^{v},...,h_{m}^{v}\} $$

## Modal Fusion

The text and image of multimodal meme have the problem of weak correlation, and directly fusing the text and image may result in incorrect meme understanding. A good fusion solution should extract and integrate sufficient information from multimodal sequences while preserving the independence of each modality. Therefore, we propose a novel global-local cross-modal interaction model that not only considers interactions between modalities but also emphasizes the importance of each modality itself to enhance multimodal fusion at multiple granularities. Specifically, we devise an efficient mechanism called Cross-modal Attention Promotion (CAP) that leverages symmetric cross-modal attention to explore the inherent correlations between the two input feature sequences, promoting the exchange of beneficial information across the sequences. CAP utilizes self-attention to model the temporal dependencies within each feature sequence, enabling the integration of more information. The mechanism takes sequences $ \pmb{h}^{T} $ and $ \pmb{h}^{I} $ as inputs and generates their mutually reinforcing information $ \pmb{h}_{T\rightarrow I} $ and $ \pmb{h}_{I\rightarrow T} $. The computation of $ CAP_{T\leftrightarrow I}(\pmb{h}^{T},\pmb{h}^{I}) $ is as follows:

$$ \begin{aligned}\boldsymbol{h}^{\prime}_{T\rightarrow I}&=MCA(LN(\boldsymbol{h}_{T}),LN(\boldsymbol{h}_{I}))+\boldsymbol{h}_{T}\\\boldsymbol{h}^{\prime\prime}_{T\rightarrow I}&=MSA(LN(\boldsymbol{h}^{\prime}_{T\rightarrow I}))+\boldsymbol{h}^{\prime}_{T\rightarrow I}\\\boldsymbol{h}_{T\rightarrow I}&=FN(LN(\boldsymbol{h}^{\prime\prime}_{T\rightarrow I}))+\boldsymbol{h}^{\prime\prime}_{T\rightarrow I}\end{aligned} $$

where LN denotes layer normalization and FN is the feedforward neural network. $ MSA(\cdot) $ refers to the output of the multi-head self-attention mechanism computation. $ MCA(\cdot,\cdot) $ represents the result of the multiple cross-attention mechanism calculation. Similarly, we can obtain $ CAP_{I\leftrightarrow T}(h_I,h_T) $.

The traditional cross-attention interaction requires two updates during the modal interaction process to achieve modal enhancement, which is inefficient and introduces redundant features into the sequence. Based on the historical experience from large-scale pretraining, it has been observed that a single token can represent the entire sequence, further improving the efficiency of modal interaction. Motivated by this observation, we propose a global-local cross-modal interaction model with linear computational cost to enhance efficiency. The discourse-level representation of each modality replaces the standard information and interacts with local unimodal features within a global multimodal context. This means that the representation of each modality not only relies on local features but also takes into account the influence of global context. This global-local interaction model reduces the introduction of redundant features and improves modal interaction effectiveness while maintaining efficiency.

We establish the global multimodal context information denoted as $ \boldsymbol{g}^{i} = \text{concat}(\boldsymbol{h}_{T}^{i}, \boldsymbol{h}_{I}^{i}) $ by concatenating the representations of each modality at each layer of global-local interaction, where i represents the number of layers of global local interaction. By integrating the global context information and local modal information, and learning modal consistency and specificity, we ensure effective interaction and capture relevant information from both the global and local perspectives. The entire interaction process is as follows:

$$ \begin{aligned}\boldsymbol{h}_{T}^{(i+1)},\boldsymbol{g}_{T\rightarrow G}^{(i)}&=C A P_{T\leftrightarrow G}^{(i)}(\boldsymbol{h}_{T}^{(i)},\boldsymbol{g}^{(i)})\\\boldsymbol{h}_{I}^{(i+1)},\boldsymbol{g}_{I\rightarrow G}^{(i)}&=C A P_{I\leftrightarrow G}^{(i)}(\boldsymbol{h}_{I}^{(i)},\boldsymbol{g}^{(i)})\\\end{aligned} $$

By stacking multiple layers, the global multimodal context and local unimodal features can mutually reinforce and progressively refine each other. We hierarchically handle the entire learning process, with each layer capturing different features corresponding to the model's main stages. The model initially learns shallow interaction features, gradually progressing to acquire higher-order semantic features in later stages. This hierarchical learning method successfully integrates information from various modalities by ingeniously designed aggregation blocks, providing the model with a more comprehensive and enriched representation of multimodal features. Through the model's interactions, information from different modalities can be combined in a deeper and more effective manner, enabling the acquisition of more advanced feature representations in subsequent hierarchical learning. Subsequently, we aggregate the features from both unimodal and multimodal sources to facilitate subsequent task predictions.

$$ y_{m}=softmax(\boldsymbol{W}_{m}MSA([\boldsymbol{h}_{T}^{(L)},\boldsymbol{h}_{I}^{(L)},\boldsymbol{g}^{(L)}])+\boldsymbol{b}) $$

where $ y_{m} $ is the feature output distribution after multimodal fusion, $ W_{m} $ and b are trainable parameters.

Our approach focuses not only on the interactions between modalities but also on the individual feature representations of each modality. We separately use the unimodal features obtained from text and image encoders to predict subsequent tasks. This allows to capture the unique characteristics and information within each modality, thereby improving the accuracy and effectiveness of MMU.

$$ \begin{aligned}\boldsymbol{y}_{T}&=softmax(\boldsymbol{W}_{t}\boldsymbol{h}_{T}+\boldsymbol{b})\\\boldsymbol{y}_{I}&=softmax(\boldsymbol{W}_{i}\boldsymbol{h}_{I}+\boldsymbol{b})\end{aligned} $$

Given the $ y_{M}, y_{T} $, and $ y_{I} $, we obtain the final prediction y:

$$ \boldsymbol{y}=\boldsymbol{y}_{M}+\boldsymbol{y}_{T}+\boldsymbol{y}_{I} $$

where y can be considered as a comprehensive feature set encompassing multi-granular features, including text, image, and image-text modalities.

[Table 1 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 1: Experimental results on the MET-MEME dataset of four tasks.

Summary: This table reports experimental results on the MET-MEME dataset for four tasks: Sentiment Analysis, Intention Detection, Offensiveness Detection, and Metaphor Recognition, in both English and Chinese. It compares ten baseline methods (VGG16, DenseNet-161, ResNet-50, Multi-BERT_EfficientNet, Multi-BERT_ViT, Multi-BERT_PiT, MET_add, MET_cat, M3F_add, M3F_cat) against a proposed method (Ours). Metrics include Accuracy, Precision, and Recall. The Ours row shows improvements (in parentheses) over a baseline, and Ours consistently achieves the highest values across all task-language-metric combinations.

Table LaTeX:

```latex
\begin{tabular}{lllllllllllll}
\hline
Method & English & English & English & Chinese & Chinese & Chinese & English & English & English & Chinese & Chinese & Chinese \\
\hline
Method & Acc & Pre & Rec & Acc & Pre & Rec & Acc & Pre & Rec & Acc & Pre & Rec \\
Sentiment Analysis & Sentiment Analysis & Sentiment Analysis & Sentiment Analysis & Sentiment Analysis & Sentiment Analysis & Sentiment Analysis & Intention Detection & Intention Detection & Intention Detection & Intention Detection & Intention Detection & Intention Detection \\
VGG16 & 20.57 & 20.84 & 24.22 & 29.94 & 26.04 & 29.20 & 37.19 & 38.71 & 38.15 & 47.48 & 49.21 & 47.81 \\
DenseNet-161 & 21.88 & 21.71 & 25.65 & 29.45 & 27.50 & 29.36 & 38.10 & 39.31 & 37.89 & 47.23 & 39.24 & 47.06 \\
ResNet-50 & 21.74 & 18.63 & 21.35 & 29.36 & 27.50 & 29.28 & 39.19 & 37.12 & 40.10 & 47.15 & 39.23 & 47.06 \\
Multi-BERT\_EfficientNet & 28.52 & 24.52 & 29.04 & 33.50 & 35.29 & 33.42 & 43.10 & 41.54 & 42.19 & 51.03 & 43.06 & 51.03 \\
Multi-BERT\_ViT & 24.43 & 23.41 & 23.96 & 33.25 & 27.33 & 32.84 & 41.28 & 40.13 & 40.62 & 50.62 & 41.32 & 50.62 \\
Multi-BERT\_PiT & 25.00 & 27.82 & 28.12 & 33.66 & 33.58 & 33.09 & 42.23 & 41.09 & 41.02 & 50.21 & 50.00 & 50.04 \\
MET\_add & 24.65 & 24.52 & 25.26 & 32.50 & 32.62 & 33.50 & 40.32 & 40.39 & 41.28 & 52.93 & 52.68 & 54.01 \\
MET\_cat & 27.68 & 28.41 & 29.82 & 33.42 & 34.33 & 33.91 & 38.56 & 39.19 & 39.84 & 51.58 & 51.48 & 52.85 \\
M3F\_add & 30.47 & 33.45 & 30.34 & 39.95 & 41.80 & 39.87 & 44.40 & 41.89 & 44.32 & 55.25 & 54.57 & 55.00 \\
M3F\_cat & 29.82 & 34.18 & 30.73 & 37.22 & 39.55 & 37.97 & 44.10 & 44.56 & 43.53 & 53.52 & 54.72 & 54.52 \\
Ours & 34.36 (+3.89\%) & 37.77 (+3.52\%) & 34.38 (+3.65\%) & 42.11 (+2.16\%) & 47.59 (+5.79\%) & 42.02 (+2.15\%) & 47.92 (+3.52\%) & 47.53 (+2.97\%) & 47.06 (+2.74\%) & 58.56 (+3.31\%) & 57.99 (+3.27\%) & 58.32 (+3.32\%) \\
Offensiveness Detection & Offensiveness Detection & Offensiveness Detection & Offensiveness Detection & Offensiveness Detection & Offensiveness Detection & Offensiveness Detection & Metaphor Recognition & Metaphor Recognition & Metaphor Recognition & Metaphor Recognition & Metaphor Recognition & Metaphor Recognition \\
VGG16 & 67.10 & 63.42 & 72.53 & 70.07 & 64.11 & 72.07 & 78.39 & 79.73 & 79.95 & 67.00 & 67.24 & 67.82 \\
DenseNet-161 & 69.66 & 62.07 & 69.98 & 71.43 & 70.82 & 75.43 & 80.08 & 80.23 & 80.47 & 67.16 & 67.91 & 67.99 \\
ResNet-50 & 69.21 & 64.62 & 72.57 & 73.24 & 69.62 & 75.74 & 80.34 & 81.22 & 80.86 & 67.74 & 67.63 & 67.66 \\
Multi-BERT\_EfficientNet & 73.78 & 67.98 & 74.56 & 78.15 & 72.11 & 79.98 & 82.46 & 84.39 & 83.11 & 74.19 & 71.26 & 74.28 \\
Multi-BERT\_ViT & 71.22 & 62.96 & 72.66 & 76.92 & 67.31 & 78.74 & 81.90 & 82.01 & 82.46 & 73.28 & 72.55 & 73.70 \\
Multi-BERT\_PiT & 72.79 & 66.69 & 74.26 & 77.17 & 70.16 & 79.05 & 82.07 & 83.05 & 82.98 & 75.10 & 73.15 & 74.28 \\
MET\_add & 68.39 & 66.21 & 72.14 & 76.01 & 74.76 & 78.16 & 81.33 & 81.49 & 82.29 & 74.04 & 74.51 & 74.96 \\
MET\_cat & 67.25 & 66.15 & 74.48 & 73.19 & 71.59 & 79.49 & 82.39 & 82.69 & 83.33 & 72.90 & 72.80 & 75.67 \\
M3F\_add & 76.17 & 69.45 & 76.19 & 80.81 & 76.00 & 80.73 & 83.98 & 85.86 & 84.38 & 77.01 & 72.94 & 82.68 \\
M3F\_cat & 74.09 & 69.59 & 76.15 & 80.07 & 76.20 & 80.62 & 83.20 & 85.97 & 85.81 & 76.18 & 73.02 & 80.00 \\
Ours & 78.11 (+1.94\%) & 77.73 (+8.14\%) & 78.32 (+2.13\%) & 82.15 (+1.34\%) & 81.80 (+5.6\%) & 82.02 (+1.29\%) & 87.51 (+3.53\%) & 88.58 (+2.61\%) & 88.89 (+3.08\%) & 78.39 (+1.38\%) & 78.16 (+3.65\%) & 83.92 (+1.24\%) \\
\hline
\end{tabular}
```

## Training

For each task, we train the model using the standard gradient descent algorithm to minimize the cross-entropy loss.

$$ \min_{\theta}\mathcal{L}_{*}=-\sum_{i=1}^{N}y_{*}^{i}log\hat{y}_{*}^{i}+\lambda_{*}\left\|\theta_{*}\right\|^{2} $$

where $ * $ stands for the representation of different tasks. N is the training data size. $ y^{i} $ and $ \hat{y}^{i} $ respectively represent the ground-truth and estimated label distribution of instance i. $ \theta $ denotes all trainable parameters of the model, $ \lambda $ represents the coefficient of L2-regularization.

Dual-semantic Guided Loss. In addition to task-specific losses, we devise a dual-semantic guided loss to effectively leverage cross-modal information, enhancing the model's comprehension and alignment of multimodal representations in the semantic space. The context-aware multimodal representations ( $ hT_i $ and $ hI_I $) contain contextually relevant information associated with specific memes. By bringing related image-text pairs closer in the forward direction and pushing unrelated pairs apart in the reverse direction, these representations are aligned in the same semantic space, effectively utilizing cross-modal information. Specifically, we contrast the multimodal representation of a specific meme sample (i.e., $ h_{T_i} $), with another multimodal representation ( $ hI_i $) from within the same batch of sampled memes. By comparing the similarities and differences between these representations, the model learns how to better differentiate and capture the semantic information among different memes samples.

$$ \mathcal{L}_{d g}=-l o g\frac{e x p(s i m(\boldsymbol{h}_{T_{i}},\boldsymbol{h}_{I_{i}})/\tau)}{\sum_{k=1}^{2N}e x p(s i m(\boldsymbol{h}_{T_{k}},\boldsymbol{h}_{I_{k}})/\tau))} $$

where sim is the cosine-similarity, N is the batch size, and $ \tau $ is the temperature to scale the logits.

By minimizing task-specific losses for each individual task and incorporating a contrastive loss, our overall loss function is defined as follows:

$$ \mathcal{L}=\mathcal{L}_{m r}+\mathcal{L}_{s a}+\mathcal{L}_{i d}+\mathcal{L}_{o d}+\mathcal{L}_{d g} $$

## Experiments

## Experimental Setting

Datasets. We assess the efficacy of our model on the widely used MET-MEME bilingual dataset (Xu et al. 2022), which consists of both English and Chinese memes. The English meme dataset is sourced from MEMOTION and Google search, comprising 4,000 text-image pairs, comprising 4,000 text-image pairs. The Chinese meme dataset consists of 6,045 text-image pairs, covering six different categories, including animals, scenery, animations, films, dolls, and humans.

Evaluation Metrics. In terms of evaluation metrics, we align with Wang et al. (2024) and use three metrics, namely accuracy (Acc), weighted precision (Pre), and recall (Rec), to assess the performance. All our scores are the average over 5 runnings with random seeds.

## Baseline Systems

To validate the effectiveness of our model, we compare it against the following state-of-the-art baselines. (1) Unimodal models that solely utilize image modality information, such as VGG16 (Simonyan and Zisserman 2014), DenseNet-161 (Huang et al. 2017), and ResNet-50 (He et al. 2016). (2) Multimodal models that incorporate both text and image modalities. These include Multi-BERT-EfficientNet (Tan and Le 2019), Multi-BERT-ViT (Dosovitskiy et al. 2020), Multi-BERT-PiT (Heo et al. 2021), MET (Xu et al. 2022) employ a straightforward concatenation or element-wise addition to

[Table 2 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 2: Ablation results on the MET-MEME English dataset of four tasks. “OM” means object-level semantic mining, “UP” means unimodal prediction, “GL” means global-local interaction, “DG” means dual-semantic guided strategy.

Summary: This table presents ablation results on the MET-MEME English dataset comparing the full model (Ours) against four ablated variants: without object-level semantic mining (w/o OM), without unimodal prediction (w/o UP), without global-local interaction (w/o GL), and without dual-semantic guided strategy (w/o DG). Metrics reported are Accuracy, Precision, and Recall for four tasks: Sentiment Analysis, Intention Detection, Offensiveness Detection, and Metaphor Recognition.

Table LaTeX:

```latex
\begin{tabular}{lllllll}
\hline
Method & Sentiment Analysis & Sentiment Analysis & Sentiment Analysis & Intention Detection & Intention Detection & Intention Detection \\
\hline
Method & Acc & Pre & Rec & Acc & Pre & Rec \\
Ours & 34.36 & 37.77 & 34.38 & 47.92 & 47.53 & 47.06 \\
w/o OM & 32.47 & 36.16 & 32.39 & 46.27 & 46.17 & 45.89 \\
w/o UP & 32.85 & 36.53 & 32.76 & 46.73 & 46.59 & 46.11 \\
w/o GL & 31.76 & 35.49 & 31.83 & 45.82 & 45.41 & 45.47 \\
w/o DG & 33.59 & 36.94 & 33.52 & 47.05 & 46.94 & 46.63 \\
 & Offensiveness Detection & Offensiveness Detection & Offensiveness Detection & Metaphor Recognition & Metaphor Recognition & Metaphor Recognition \\
 & Acc & Pre & Rec & Acc & Pre & Rec \\
Ours & 78.11 & 77.73 & 78.32 & 87.51 & 88.58 & 88.89 \\
w/o OM & 77.28 & 73.41 & 77.22 & 85.42 & 87.45 & 87.46 \\
w/o UP & 77.53 & 74.66 & 77.68 & 85.81 & 87.64 & 87.74 \\
w/o GL & 76.89 & 72.57 & 76.94 & 84.98 & 86.77 & 86.79 \\
w/o DG & 77.74 & 75.39 & 77.93 & 86.43 & 87.93 & 88.02 \\
\hline
\end{tabular}
```

merge feature vectors for meme understanding, and M3F (Wang et al. 2024) devise attention mechanisms to capture the interaction between text and images.

## Main Results

We conduct a comprehensive comparison of our MGmCF with SoTA unimodal and multimodal models on the METMEME dataset. Table 1 presents the results for four tasks: sentiment analysis (SA), intention detection (ID), offensive-ness detection (OD), and metaphor recognition (MR). The results highlight that our approach outperforms SoTA baselines across all evaluation metrics for the four tasks, revealing several key findings. Firstly, compared to unimodal models, multimodal models demonstrate superior performance by leveraging additional visual-textual features. However, it is crucial to thoroughly exploit visual information and deeply fuse multimodal features. Otherwise, they only yield marginal improvements over unimodal models or even underperform them (e.g., in the OD task, the MET method performs worse than DenseNet-161 and ResNet-50). By creating cross-modal attention, M3F achieves the current SoTA results. Notably, our model significantly surpasses the SoTA techniques. On the English meme dataset, our precision improves by 8.14% in the OD task, and accuracy improves by 3.53%, 3.89%, and 3.52% in MR, SA, and ID, respectively. On the Chinese dataset, our precision improves by 3.65%, 5.79%, 3.27%, and 5.6% in MR, SA, ID, and OD, respectively. Furthermore, our approach exhibits significant enhancements compared to MET. On the English MEME dataset, improvements range from 6.18% to 13.25%, and on the Chinese MEME dataset, enhancements range from 3.86% to 14.97%. These findings indicate that by focusing on object-level fine-grained image details and the intrinsic unimodal clues, and integrating multi-granular clues, we achieve significant performance improvements in the MMU.

## Ablation Study

We perform ablation experiments to evaluate the contribution of each component in our model. As depicted in Table 2, no variant matches the full model's performance, highlighting the indispensability of each component. Specifically, when

[Figure 3 was here. The original paper contained a figure at this position. Brief visual description: The figure displays a comparative bar chart evaluating two interaction methods, 'global-local interaction' (blue bars) and 'local-local interaction' (pink bars), across four distinct tasks: Sentiment Analysis, Intention Detection, Offensiveness Detection, and Metaphor Recognition. For each task, performance metrics including Recall (Rec), Precision (Pre), and Accuracy (Acc) are plotted as percentages. Visually, the blue bars representing the global-local interaction consistently extend further to the right than the pink bars in every metric and task category, indicating superior performance.]
Caption: Figure 3: Comparative results between global-local and local-local interaction on the MET-MEME English dataset.
Key visible elements:
- global-local interaction: Method being evaluated, represented by blue bars
- local-local interaction: Baseline or comparative method, represented by pink bars
- Sentiment Analysis: Task panel title (top-left)
- Intention Detection: Task panel title (top-right)
- Offensiveness Detection: Task panel title (bottom-left)
- Metaphor Recognition: Task panel title (bottom-right)
- Rec: Metric label for Recall
- Pre: Metric label for Precision

the global-local interaction is not utilized, three evaluation metric scores for all four tasks suffer the most significant decline. In particular, the precision score for the OD task drops by 5.16%, and for the SA task drops by 2.28%. This indicates that the global-local interaction successfully integrates information from different modalities, providing a more comprehensive multimodal feature representation. To validate the necessity of object-level semantic mining module, we remove this module, and the decline in results signifies its indispensable impact on MMU. This finding suggests that mining fine-grained information from images can offer more detailed insights into image content. Furthermore, removing unimodal predictions leads to a performance drop, indicating that unimodal predictions contribute to the final predictions and effectively address the issue of weak correlation between image and text. Additionally, performance declines when dual-semantic guided strategy is removed, demonstrating its crucial role in enhancing semantic alignment and reducing modalities' inconsistencies.

## Deep Analyses on The Proposed Methods

To further investigate the effectiveness of our method, we conduct in-depth analyses to answer the following questions, aiming to mine the intuition and analyze implicit phenomena. Q1: What are the advantages of the global-local interaction? To further validate the effectiveness of our proposed global-local interaction model, we conduct a comparative analysis between our global-local interaction method and the local-local interaction method. The local-local interaction method refers to performing pairwise local interactions within each modality. As shown in Figure 3, the results consistently demonstrate the superiority of the global-local interaction across all evaluation metrics for the four tasks. This finding indicates that by incorporating a higher-level global context that encompasses the entire modality, the global-local interaction method achieves more comprehensive and effective interactions between modalities. In contrast, the local-

[Figure 4 was here. The original paper contained a figure at this position. Brief visual description: The figure presents a visualization of a meme image containing a person and two bears, overlaid with red bounding boxes to highlight specific regions. Text labels identify the person as 'Me', the smaller bear as 'Self doubt', and the larger bear as 'This one is just a real bear'. A legend on the right side lists these labels alongside a color scale ranging from 0.0 to 1.0, indicating a quantitative measure of relevance or attention for the identified elements.]
Caption: Figure 4: Visualization of a typical example.
Key visible elements:
- Meme image content: The base visual input showing a person and two bears in a forest setting.
- Red bounding boxes: Visual indicators highlighting specific regions of interest within the image.
- Text labels ('Me', 'Self doubt', 'This one is just a real bear'): Semantic identifiers for the highlighted regions.
- Legend/Scale panel: A vertical bar on the right displaying labels and a numerical score range (0.0-1.0) associated with the concepts.

[Figure 5 was here. The original paper contained a figure at this position. Brief visual description: The figure presents four bar charts evaluating the performance of a proposed multimodal method ('Our') against two unimodal variants ('Without Image' and 'Without Text') across four tasks: Sentiment Analysis, Intention Detection, Offensiveness Detection, and Metaphor Recognition. Each chart displays Accuracy (Acc), Precision (Pre), and Recall (Rec) as percentages.]
Caption: Figure 5: Influence of unimodal prediction on the MET-MEME English dataset.
Key visible elements:
- Sentiment Analysis Chart: Displays performance metrics for the sentiment analysis task
- Intention Detection Chart: Displays performance metrics for the intention detection task
- Offensiveness Detection Chart: Displays performance metrics for the offensiveness detection task
- Metaphor Recognition Chart: Displays performance metrics for the metaphor recognition task
- Blue Bars ('Our'): Represents the performance of the proposed full model
- Red Bars ('Without Image'): Represents the performance of the text-only variant
- Beige Bars ('Without Text'): Represents the performance of the image-only variant
- X-axis Categories (Acc, Pre, Rec): Indicates the evaluation metrics: Accuracy, Precision, Recall

local interaction method solely focuses on intra-modality local feature interactions and fails to fully leverage the holistic information across modalities. By leveraging the global context, we capture a broader range of semantic information, enabling a deeper understanding of the interdependence between modalities and effectively addressing inconsistencies and differences between modalities, thereby enhancing the performance of the MMU task.

Q2: How can fine-grained visual features help improve model performance? We conduct a visualization in Figure 4 to better illustrate the outstanding performance of the fine-grained visual enhancement module. By visualizing the attention distribution of the model, we observe that the module exhibits higher attention towards specific object parts in the image when the fine-grained visual enhancement module is utilized. Focusing on specific objects rather than the entire image allows for more accurate capture of crucial visual details. This confirms the effectiveness of the fine-grained visual enhancement module in improving the model's attention and understanding of key visual information.

Q3: What are the advantages of unimodal prediction in multimodal meme understanding? In order to validate the effectiveness of incorporating unimodal prediction in multimodal meme understanding, we conduct a comparative analysis of the performance of combining unimodal prediction with separately removing text modal prediction and image modal prediction. As illustrated in Figure 5, the results

[Figure 6 was here. The original paper contained a figure at this position. Brief visual description: Figure 6 presents a comparative analysis of the proposed method ('clip + our') against a baseline ('clip') using four radar charts. The charts evaluate performance across four tasks: Sentiment Analysis, Intention Detection, Offensiveness Detection, and Metaphor Recognition. Each chart plots three metrics: Precision (Pre%), Accuracy (Acc%), and Recall (Rec%). In all four tasks, the 'clip + our' model (represented by the larger red/pink polygon) demonstrates superior performance compared to the 'clip' baseline (represented by the smaller blue polygon).]
Caption: Figure 6: Influence of CLIP in MMU on the MET-MEME English dataset.
Key visible elements:
- Sentiment Analysis Chart: Radar chart showing performance on sentiment analysis task
- Intention Detection Chart: Radar chart showing performance on intention detection task
- Offensiveness Detection Chart: Radar chart showing performance on offensiveness detection task
- Metaphor Recognition Chart: Radar chart showing performance on metaphor recognition task
- clip + our (Red/Pink Polygon): Data series representing the proposed method's performance
- clip (Blue Polygon): Data series representing the baseline CLIP model's performance
- Axes (Pre%, Acc%, Rec%): Performance metrics plotted for each task

consistently show that the combined unimodal prediction outperform the scenarios where any unimodal prediction is removed across all evaluation metrics in the four tasks. This indicates that by focusing on the crucial information within each modality, we can achieve a more comprehensive and accurate understanding of the visual and textual elements within memes. Furthermore, we observe that removing the image modality prediction has a larger impact on the performance compared to removing the text modality prediction. This finding emphasizes the importance of enhancing visual information. By capturing fine-grained visual details, the model can better leverage the rich visual cues and context present in memes. These findings highlight the value of unimodal prediction and underscore the significance of considering the specific characteristics of each modality in MMU.

Q4: What impact does a large language model have on MMU? Considering the extensive usage of large language models (Wu et al. 2024a,b; Fei et al. 2024c,b), we investigate their influence on MMU. Figure 6 displays the results achieved by employing the standalone CLIP model and by integrating the CLIP model with our proposed method. Notably, employing the CLIP model alone yields impressive performance, attesting to its adeptness in comprehending multimodal memes. Moreover, the integration of the CLIP model with our method results in additional performance improvements, underscoring the efficacy of our approach.

## Conclusion

In this paper, we explore two major challenges in the MMU task: the loss of fine-grained metaphorical visual clues and the neglect of weak correlation between multimodal text and images, proposing a solution named MGMCF. MGMCF enhances the complexity and diversity of images by capturing object-level fine-grained visual clues, and resolves the weak correlation between text and images through a novel global-local cross-modal interaction for multi-granular clue fusion. Extensive experiments on the MET-MEME bilingual dataset demonstrate the effectiveness of all our proposed innovative methods and hypotheses, achieving SoTA performance.