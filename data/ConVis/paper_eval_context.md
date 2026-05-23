# ConVis: Contrastive Decoding with Hallucination Visualization for Mitigating Hallucinations in Multimodal Large Language Models
Yejì Park $ ^{*1} $, Deokyeong Lee $ ^{*1} $, Junsuk Choe $ ^{\dagger1} $, Buru Chang $ ^{\dagger2} $
## Abstract

Hallucinations in Multimodal Large Language Models (MLLMs) where generated responses fail to accurately reflect the given image pose a significant challenge to their reliability. To address this, we introduce ConVis, a novel training-free contrastive decoding method. ConVis leverages a text-to-image (T2I) generation model to semantically reconstruct the given image from hallucinated captions. By comparing the contrasting probability distributions produced by the original and reconstructed images, ConVis enables MLLMs to capture visual contrastive signals that penalize hallucination generation. Notably, this method operates purely within the decoding process, eliminating the need for additional data or model updates. Our extensive experiments on five popular benchmarks demonstrate that ConVis effectively reduces hallucinations across various MLLMs, highlighting its potential to enhance model reliability.

## Introduction

Multimodal Large Language Models (MLLMs) (Dai et al. 2023; Liu et al. 2024b) are advanced language models capable of understanding both images and text, such as image captioning and visual question answering (VQA). While MLLMs have achieved significant success that utilize both visual and textual information, the issue of hallucination, where the models generate responses that do not align with the given image, has greatly undermined their reliability (Liu et al. 2023a; Sun et al. 2024). This problem poses a significant obstacle to adopting MLLMs in critical fields where reliability is crucial. For instance, in medical applications, it could lead to incorrect diagnoses (Liu et al. 2023b), while in MLLM-based autonomous systems, it might result in erroneous interpretations (Shao et al. 2024).

Recent research has been actively conducted to address this. WoodPecker (Yin et al. 2023) and LURE (Zhou et al. 2024) reduce hallucinations by post-processing the generated responses. Datasets such as LRV-Instruction (Liu et al. 2023a) and RLHF-V (Yu et al. 2024) have been proposed to mitigate hallucinations through instruction tuning of MLLMs. However, these studies often rely on external

[Figure 1 was here. The original paper contained a figure at this position. Brief visual description: The figure illustrates a pipeline for analyzing image hallucinations in text-to-image models. It shows an 'Original Image' of a man on a beach being processed by an MLLM (Multimodal Large Language Model) with the prompt 'Describe this image in detail'. The MLLM generates a 'Hallucinated Caption' mentioning a 'book'. This caption is then fed into a 'T2I Model' (Text-to-Image), which produces a 'Generated Image' where the man is holding a large, distinct blue book. The figure highlights how the generated image emphasizes the 'book' mentioned in the caption.]
Caption: Figure 1: The text-to-image model visualizes hallucinations (e.g., book') in the semantically reconstructed images based on the hallucinated caption, exhibiting differences (e.g., missing clock') from the original image.
Key visible elements:
- Original Image: Input image showing a man on a beach holding an object
- Prompt: Instruction given to the MLLM ('Describe this image in detail')
- MLLM: Model block processing the image and prompt to generate text
- Hallucinated Caption: Text output describing the scene, specifically mentioning 'carrying a book'
- T2I Model: Model block generating an image from the text caption
- Generated Image: Output image showing the man holding a large blue book

APIs like GPT-3.5, require costly human feedback collection, and necessitate additional training of MLLMs.

In contrast, this paper focuses on decoding strategies that reduce hallucinations by intervening solely in the decoding process, without the need for additional data or model training. The following studies fall into this category: OPERA (Huang et al. 2024) imposes penalties on token generation that does not reference visual tokens. VCD (Leng et al. 2024) creates contrasting distributions using distorted images to reduce the model's reliance on statistical biases and priors that lead to hallucinations. HALC (Chen et al. 2024) corrects hallucinations by leveraging cues provided by visual information from various fields of view.

In this study, we propose a contrastive decoding method called ConVis (Contrastive Decoding with Hallucination Visualization), which can be applied to any existing MLLM without additional training. Inspired by the previous work (Kim et al. 2024), ConVis leverages text-to-image (T2I) generation models, specifically Hyper-SDXL (Ren et al. 2024), to capture visual contrast signals. The process begins with the MLLM generating a caption for the input image, after which the T2I model reconstructs an image based on this caption. As shown in Figure 1, if the generated caption contains hallucinations (e.g., a book'), there will be visual discrepancies between the original and reconstructed

[Figure 2 was here. The original paper contained a figure at this position. Brief visual description: The figure illustrates a qualitative comparison of how an original image versus a generated (reconstructed) image influences the logit distribution of a Multimodal Large Language Model (MLLM). It shows two parallel processing streams where both images are paired with the same text input sequence containing a masked token. The top stream uses the original image, resulting in a logit distribution where 'book' is predicted but shares prominence with other tokens like 'clock'. The bottom stream uses the generated image, which clearly depicts a book, resulting in a significantly amplified logit for the token 'book' compared to others.]
Caption: Figure 2: The original and reconstructed image generate the contrastive logit distribution for the hallucinated tokens (e.g., ‘book’). The reconstructed image tends to amplify the logits of tokens corresponding to the visualized hallucination.
Key visible elements:
- Original Image v: Input visual showing a man on a beach with an indistinct object
- Generated Image v': Input visual showing a man on a beach clearly holding a book
- Input Sequence x: Text prompt describing the scene with a masked token '[?]'
- MLLM: Model block processing the image and text inputs
- Logit Distribution Charts: Bar charts showing model confidence scores for tokens: bottle, clock, book, towel

images (e.g., a missing clock'). ConVis then uses the original and reconstructed images to compare the probability distributions (Figure 2), capturing visual contrast signals that highlight hallucinations. Based on these signals, ConVis penalizes the generation of hallucinations during the decoding process, reducing the hallucinations.

To validate the effectiveness of ConVis, we conduct experiments across five benchmarks: CHAIR (Rohrbach et al. 2018), HallusionBench (Guan et al. 2024), POPE (Li et al. 2023c), MME (Fu et al. 2023) and LLaVA-Bench (Liu et al. 2024b). The results consistently demonstrate that our decoding method reduces hallucinations while maintaining overall response generation performance across various MLLMs, including LLaVA-1.5 (Liu et al. 2024a), MiniGPT-4 (Zhu et al. 2024), and mPLUG-Owl2 (Ye et al. 2024).

Our contributions can be summarized as follows: (1) Propose ConVis, a novel contrastive decoding method that visualizes hallucinations using a T2I model. To the best of our knowledge, this is the first time a T2I model has been employed to mitigate hallucinations through a decoding strategy. (2) Conduct extensive experiments to validate the effectiveness of ConVis in reducing hallucinations. (3) Provide insights into how T2I models can serve as a valuable source of visual contrastive signals in decoding methods aimed at mitigating hallucinations.

## Related Work

## Multimodal Large Language Models

The emergence of LLMs has revolutionized the paradigm of Natural Language Processing (NLP). The significant success of LLMs in the NLP field has led to research on leveraging LLMs in the visual domain. Consequently, MLLMs that can simultaneously handle visual and textual data have recently been proposed. Specifically, to process visual information, LLaVA (Liu et al. 2024b) uses a CLIP vision encoder (Radford et al. 2021) and a linear layer to project images into the LLM's input embedding space. MiniGPT-4 (Zhu et al. 2024) employs a Q-Former (Li et al. 2023a) and a linear layer to project images into the LLM's input embedding space. Additionally, mPLUG-Owl2 (Lai et al. 2024) introduces a modality-adaptive module that preserves modality-specific features, allowing the model to excel in both multimodal and NLP tasks.

However, despite these efforts, misalignment between modalities can still occur for various reasons, leading to generated responses that do not correspond to the visual information. This phenomenon, known as hallucination, undermines the reliability of MLLMs and poses a significant challenge to their application in real-world scenarios.

## Hallucination Mitigation

To address the hallucination problem in MLLMs, several studies have been proposed recently. LURE (Zhou et al. 2024) and Woodpecker (Yin et al. 2023) employ post-processing methods to revise generated responses, either by training a revisor or using GPT-3.5-turbo (Brown et al. 2020). Fine-tuning approaches (Liu et al. 2023a; Yu et al. 2024) mitigate hallucinations through instruction tuning with additional data, but they require significant data collection and training resources. Given the large number of parameters in MLLMs, this is computationally inefficient.

Therefore, methods for improving the decoding process have recently received great attention due to the advantage that they do not require additional training. Specifically, OPERA (Huang et al. 2024) explores aggregation patterns that cause hallucinations. OPERA utilizes this insight to suppress the generation of tokens that exhibit these patterns. VCD (Leng et al. 2024) leverages the characteristic that the model tends to prioritize prior knowledge over visual information when responding to distorted images. As a result, the responses to the distorted image and the original image show significant differences in hallucinated tokens, and VCD contrasts these to mitigate the hallucinations. HALC (Chen et al. 2024) observes that when images with varying fields of view are input into the MLLM, the probability changes for ground truth tokens are much greater than for hallucinated tokens. This observation helps identify visual context candidates that clearly depict objects, and by contrasting these candidates, HALC reduces hallucinations.

Unlike existing techniques, we propose a new decoding method that utilizes a T2I model. Specifically, our approach visualizes hallucinations in the initially generated caption using a T2I model, then contrasts the responses generated from the reconstructed image with those from the original image. Through this process, we contrast distributions of the hallucinated tokens and effectively mitigate hallucinations.

## Methodology

## Preliminaries

Response Generation. The MLLM generates a response y corresponding to a given input image v and instruction text x. The input image is projected into visual tokens through an image encoder, and these tokens, along with the tokens corresponding to the instruction text, are fed into the LLM. The response is generated through autoregressive decoding according to the following equation:

$$ y_{t}\sim p_{\theta}(\cdot|v,x,y_{<t})\propto\exp\left(f_{\theta}(\cdot|v,x,y_{<t})\right), $$

[Figure 3 was here. The original paper contained a figure at this position. Brief visual description: The figure illustrates a pipeline for visualizing hallucinations in multimodal models. It shows an original image of a man on a beach being processed by an MLLM to generate a caption containing a hallucination ('carrying a book'). This caption is used by a Text-to-Image (T2I) model to create synthetic images depicting the hallucination. Finally, both the original and generated images are encoded and fed into a language model to compute logits, demonstrating that the generated image amplifies the probability of the hallucinated token compared to the original image.]
Caption: The original and generated image produce the contrastive distribution for the hallucinated tokens (e.g., ‘book’). The generated image tends to amplify the logits of tokens corresponding to the visualized hallucination.
Key visible elements:
- Original Image v: Source input showing a man on a beach without a book
- MLLM $ heta$: Multimodal Large Language Model generating the caption
- Caption c: Generated text containing a hallucination ('book')
- T2I Model: Text-to-Image model generating images from the caption
- Generated Images v': Synthetic images depicting the hallucinated object (man with a book)
- Image Encoder: Component encoding images for the LLM
- LLM: Language model computing token logits
- Contrastive Distribution: Comparison mechanism between original and generated image processing paths

where $ \theta $ denotes the parameters of the MLLM, $ y_t $ represents the $ t $-th token of response, and $ y_{<t} $ is the sequence of tokens generated up to time $ t $. $ f_\theta $ denotes the logit distribution generated by the MLLM. Hallucination refers to the phenomenon where the output $ y $ generated by the MLLM does not correspond to the input image $ v $. This study focuses on mitigating hallucinations while maintaining the overall performance of the MLLM as a language model.

Text-to-Image Generation. The core component of ConVis is the T2I model that generates images based on a given query. The goal of the T2I model is to create an image that accurately depicts the query. Among the recently proposed T2I models, we utilize Hyper-SDXL (Ren et al. 2024), an enhanced version of Stable Diffusion (Ho, Jain, and Abbeel 2020), which has demonstrated excellent T2I performance. The diffusion-based Hyper-SDXL model begins with a pure noise and progressively reconstructs it through an iterative reverse diffusion process which ultimately results in the generated image $ v_{0}^{\prime} $.

## Hallucination Visualization

We hypothesize that the T2I model can help mitigate hallucinations by providing visual contrast signals during the decoding process. If the T2I model receives a caption generated by the MLLM that contains hallucinations, it will faithfully visualize those hallucinations in the generated image. We refer to this process as hallucination visualization.

To implement this, ConVis first generates an initial caption c for the original image v using a simple instruction text that directs the MLLM to describe the image. This process is illustrated in Figure 2. The T2I model then takes the caption c as a query and generates an image $ v' $ based on it. If the caption contains hallucinations, these will be faithfully visualized in the generated image $ v' $. Conversely, if the initial caption is accurate and free of hallucinations, the generated image will be semantically similar to the original image.

Diversity of Generated Images. Given that the current T2I model may not generate images that fully align with the captions, we address this limitation by increasing the diversity of the generated images using the following approaches: (1) We first generate a diverse set of n captions using Nucleus Decoding (Holtzman et al. 2020) instead of Greedy Decoding. (2) Then, the T2I model uses these n captions to generate n corresponding images. This approach increases coverage of the various potential hallucinations that the MLLM might generate by diversifying the captions. Additionally, by using multiple images instead of a single one, we enhance the robustness of our method against the T2I model's potential misalignment between the caption and the generated image due to its imperfect performance.

We have found these approaches to be effective, with detailed results available in the experiment section.

## Contrastive Decoding

Hallucinations in captions cause visual differences between the original image v and the generated image v'. We mitigate these hallucinations by capturing the visual contrast signals from these differences. To achieve this, during the decoding process, we utilize both the original image v and the n generated images to produce the logit distribution for each image. The final contrastive logit distribution $ \hat{f}_{\theta} $ is derived by averaging the contrastive logit distributions between the original image and each generated image as follows:

$$ \hat{f}_{\theta}=\frac{1}{n}\sum_{i=1}^{n}\Big((1+\alpha)f_{\theta}(\cdot|v,x,y_{<t})-\alpha f_{\theta}(\cdot|v_{i}^{\prime},x,y_{<t})\Big), $$

where $ \alpha $ is a hyperparameter that controls the strength of the difference between the logit distributions from the original and generated images. The contrastive logit distribution $ \hat{f}_{\theta} $ is used to generate the response y. For tokens associated with hallucinations, the contrastive logit distribution is significantly amplified compared to other tokens, allowing us to penalize these tokens and reduce the hallucinations.

Note that, Equation 2 is similar to the contrastive decoding methods used in VCD (Leng et al. 2024) and HALC (Chen et al. 2024). However, our method is distinguished from existing approaches by directly capturing visual contrastive signals from the hallucinations visualized by the T2I generative model.

## Experiments

Benchmarks. To evaluate the performance of our method, we conduct experiments on three benchmarks to evaluate the

[Table 1 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 1: Evaluation results on the CHAIR benchmark using the MSCOCO dataset (val2014 split). We conduct experiments with three different sets of 500 images, each selected by random seeds. The reported value is the mean of the results from the three different seeds, with the $ \pm $ symbol representing the standard deviation.

Summary: This table reports CHAIR_S and CHAIR_I evaluation results for three multimodal models (LLaVA-1.5, mPLUG-Owl2, MiniGPT-4) under seven decoding methods (Greedy Search, Nucleus Sampling, Beam Search, VCD, OPERA, HALC, Ours) on the MSCOCO dataset (val2014 split). Each value is the mean ± standard deviation across three runs with different random seeds. Lower CHAIR_S and CHAIR_I scores indicate fewer hallucinations.

Table LaTeX:

```latex
\begin{tabular}{lllllll}
\hline
Method & LLaVA-1.5 & LLaVA-1.5 & mPLUG-Owl2 & mPLUG-Owl2 & MiniGPT-4 & MiniGPT-4 \\
\hline
Method & \$ \textbackslash{}text\{CHAIR\}\_\{S\}\textbackslash{}downarrow \$ & \$ \textbackslash{}text\{CHAIR\}\_\{I\}\textbackslash{}downarrow \$ & \$ \textbackslash{}text\{CHAIR\}\_\{S\}\textbackslash{}downarrow \$ & \$ \textbackslash{}text\{CHAIR\}\_\{I\}\textbackslash{}downarrow \$ & \$ \textbackslash{}text\{CHAIR\}\_\{S\}\textbackslash{}downarrow \$ & \$ \textbackslash{}text\{CHAIR\}\_\{I\}\textbackslash{}downarrow \$ \\
Greedy Search & 22.4 \$ \textbackslash{}pm \$ 1.11 & 7.4 \$ \textbackslash{}pm \$ 0.27 & 22.2 \$ \textbackslash{}pm \$ 1.10 & 7.3 \$ \textbackslash{}pm \$ 0.24 & 34.0 \$ \textbackslash{}pm \$ 1.11 & 13.8 \$ \textbackslash{}pm \$ 0.85 \\
Nucleus Sampling & 26.0 \$ \textbackslash{}pm \$ 1.93 & 9.5 \$ \textbackslash{}pm \$ 0.76 & 25.2 \$ \textbackslash{}pm \$ 1.59 & 9.3 \$ \textbackslash{}pm \$ 0.34 & 30.1 \$ \textbackslash{}pm \$ 1.45 & 14.2 \$ \textbackslash{}pm \$ 0.90 \\
Beam Search & 19.5 \$ \textbackslash{}pm \$ 1.42 & 6.4 \$ \textbackslash{}pm \$ 0.09 & 18.3 \$ \textbackslash{}pm \$ 0.42 & 6.0 \$ \textbackslash{}pm \$ 0.34 & 31.1 \$ \textbackslash{}pm \$ 1.03 & 12.4 \$ \textbackslash{}pm \$ 0.59 \\
VCD & 23.7 \$ \textbackslash{}pm \$ 1.90 & 8.2 \$ \textbackslash{}pm \$ 0.80 & 25.7 \$ \textbackslash{}pm \$ 1.30 & 9.0 \$ \textbackslash{}pm \$ 0.28 & 31.6 \$ \textbackslash{}pm \$ 1.83 & 13.8 \$ \textbackslash{}pm \$ 0.83 \\
OPERA & 18.5 \$ \textbackslash{}pm \$ 0.90 & 6.6 \$ \textbackslash{}pm \$ 0.23 & 18.2 \$ \textbackslash{}pm \$ 0.40 & 6.2 \$ \textbackslash{}pm \$ 0.18 & 30.6 \$ \textbackslash{}pm \$ 1.06 & 12.5 \$ \textbackslash{}pm \$ 0.91 \\
HALC & 23.7 \$ \textbackslash{}pm \$ 2.66 & 9.1 \$ \textbackslash{}pm \$ 0.41 & 24.3 \$ \textbackslash{}pm \$ 1.22 & 9.4 \$ \textbackslash{}pm \$ 0.19 & 24.2 \$ \textbackslash{}pm \$ 1.91 & 10.8 \$ \textbackslash{}pm \$ 0.53 \\
Ours & 18.4 \$ \textbackslash{}pm \$ 0.53 & 6.4 \$ \textbackslash{}pm \$ 0.37 & 17.6 \$ \textbackslash{}pm \$ 3.54 & 6.0 \$ \textbackslash{}pm \$ 0.89 & 23.5 \$ \textbackslash{}pm \$ 0.31 & 10.0 \$ \textbackslash{}pm \$ 0.69 \\
\hline
\end{tabular}
```

mitigation of hallucinations and two general-purpose benchmarks to assess the general performance of the MLLM:

• Hallucination: CHAIR (Rohrbach et al. 2018), HallusionBench (Guan et al. 2024), and Polling-based Object Probing Evaluation (POPE) (Li et al. 2023c)

• General-purpose: MLLM Evaluation (MME) (Fu et al. 2023) and LLaVA-Bench (Liu et al. 2024b)

Detailed information on these benchmarks can be found in the Appendix.

Backbones. To evaluate our method, we utilize three well-known MLLMs with publicly available checkpoint weights: LLaVA-1.5 (Liu et al. 2024a), mPLUG-Owl2 (Ye et al. 2024), and MiniGPT-4 (Zhu et al. 2024).

Compared Methods. Our method is designed to replace existing decoding methods used in the LLM component, and therefore, we compare it against baselines such as Greedy Search, Nucleus Sampling (Holtzman et al. 2020), and Beam Search (beam=5). We also evaluate our method's effectiveness against other decoding methods in hallucination mitigation, including OPERA (Huang et al. 2024), VCD (Leng et al. 2024), and HALC (Chen et al. 2024). We use the same hyperparameters borrowed from the original papers of the compared methods to ensure a fair comparison.

Implementation Details. We utilize the Hyper-SDXL (Ren et al. 2024) T2I model for image generation. Specifically, in all experiments, unless otherwise noted, we use the Step 1 generation results of Hyper-SDXL model. The maximum length of text queries that the T2I model could accept is 77 tokens, which is too short to process the captions generated by MLLM. To address this, we leverage Compel (Stewart 2023), which allows for processing more than 77 tokens. We set the maximum token count for the caption generation to 256 and use Nucleus sampling with a temperature of 0.7 and a top-p of 0.9 to generate the images. The query used in this process is “Please describe this image in detail.” We set the number of generated images, n, to 4, producing four images based on distinct captions generated using different random seeds. For contrastive decoding, we follow (Li et al. 2023b) using adaptive plausibility constraint to contrast only meaningful tokens. The plausibility constraint hyperparameter $ \lambda $ is set to 0.1. We also set $ \alpha $, which controls the degree of contrastive emphasis, to 1 for captioning-based metrics such as

[Table 2 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 2: Evaluation results on HallusionBench. We report Figure Acc and All Acc using LLaVA-1.5.

Summary: This table reports evaluation results on HallusionBench using LLaVA-1.5, comparing eight methods (Greedy Search, Nucleus Sampling, Beam Search, VCD, OPERA, HALC, and Ours) on two metrics: Figure Acc (fAcc) and All Acc (aAcc). The method 'Ours' achieves the highest scores on both metrics: 23.5 fAcc and 50.8 aAcc.

Table LaTeX:

```latex
\begin{tabular}{lll}
\hline
Method & Figure Acc (fAcc) & All Acc (aAcc) \\
\hline
Greedy Search & 22.2 & 50.1 \\
Nucleus Sampling & 17.8 & 46.2 \\
Beam Search & 19.1 & 48.4 \\
VCD & 21.7 & 47.5 \\
OPERA & 20.9 & 49.9 \\
HALC & 21.7 & 50.6 \\
Ours & 23.5 & 50.8 \\
\hline
\end{tabular}
```

CHAIR and LLaVA-Bench, and to 0.1 for VQA metrics, including POPE, HallusionBench, and MME. To generate responses, we use a greedy decoding approach for all methods. For CHAIR, we sample three different sets of images using different random seeds and assess the performance using the mean and standard deviation of these results.

## Experimental Results

Results on CHAIR. We report our evaluation results on the CHAIR (Rohrbach et al. 2018) benchmark in Table 1. Our assessment includes basic decoding strategies—Greedy search, Nucleus sampling, and Beam search—along with three state-of-the-art approaches—VCD (Leng et al. 2024), OPERA (Huang et al. 2024), and HALC (Chen et al. 2024). Our method achieves the best performance on the CHAIR $ _{S} $ metric across all three backbone models (LLaVA-1.5, mPLUG-Owl2, and MiniGPT-4). Remarkably, it significantly improves the CHAIR $ _{S} $ score compared to both the basic decoding strategies and the state-of-the-art methods, highlighting its superior ability to mitigate hallucinations. In terms of the CHAIR $ _{I} $ metric, our method consistently ranks either first or second across all backbone models. These results demonstrate that our method both excels in reducing the total number of hallucinations throughout entire sentences and minimizes the number of hallucinated objects across all evaluated image sets.

Results on HallusionBench. In Table 2, we present the evaluation results for the visual dependent category of the HallusionBench (Guan et al. 2024) benchmark. Hallusion-

[Table 3 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 3: Evaluation results on the POPE benchmark using the MSCOCO dataset (val2014 split).

Summary: Table reports evaluation results (accuracy) on the POPE benchmark (MSCOCO val2014) for four methods (VCD, OPERA, HALC, Ours) across three models (LLaVA-1.5, mPLUG-Owl2, MiniGPT-4) and their average.

Table LaTeX:

```latex
\begin{tabular}{lllll}
\hline
Method & LLaVA-1.5 & mPLUG-Owl2 & MiniGPT-4 & Average \\
\hline
VCD & 82.8 & 81.6 & 59.8 & 74.7 \\
OPERA & 83.0 & 83.3 & 66.1 & 77.4 \\
HALC & 50.6 & 83.4 & 69.7 & 67.9 \\
Ours & 83.0 & 83.0 & 69.9 & 78.6 \\
\hline
\end{tabular}
```

[Table 4 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 4: Evaluation results on the MME using LLaVA-1.5.

Summary: Table 4 presents evaluation results on the MME benchmark using LLaVA-1.5, comparing seven decoding methods: Greedy Search, Nucleus Sampling, Beam Search, VCD, OPERA, HALC, and Ours. Scores are reported for Perception, Cognition, and Total metrics.

Table LaTeX:

```latex
\begin{tabular}{llll}
\hline
Method & Category & Category & Total \\
\hline
Method & Perception & Cognition & Total \\
Greedy Search & 1472.5 & 303.9 & 1776.4 \\
Nucleus Sampling & 1203.4 & 311.1 & 1514.5 \\
Beam Search & 1478.0 & 287.5 & 1765.5 \\
VCD & 1326.7 & 374.6 & 1701.3 \\
OPERA & 1456.9 & 306.4 & 1763.3 \\
HALC & 887.7 & 269.6 & 1157.3 \\
Ours & 1487.6 & 306.1 & 1793.7 \\
\hline
\end{tabular}
```

Bench is evaluated with the assistance of GPT-4V, which incurs significant costs; therefore, we conduct experiments using only the LLaVA-1.5 (Liu et al. 2024a) backbone. Our method demonstrates superior performance in Figure Accuracy (fAcc), outperforming all baseline decoding strategies (Greedy Search, Nucleus Sampling, Beam Search) as well as state-of-the-art techniques (VCD, OPERA, HALC). This indicates that our model effectively interprets the visual details of images when responding to visually dependent questions, indicating its ability to mitigate hallucinations by providing responses that closely align with the given visual content. Furthermore, our method achieves the highest performance on the All Accuracy (aAcc) metric, which measures overall accuracy across all questions within the visual dependent category, demonstrating its effectiveness in handling a wide range of visually dependent queries.

Results on POPE. Table 3 reports the evaluation results on the POPE (Li et al. 2023c) benchmark using the MSCOCO (Lin et al. 2014) dataset (val2014 split). We present the average F1-scores across the three POPE question splits—Random, Popular, and Adversarial—for three different backbone models. Detailed performances on each POPE question split are in the Appendix.

Our method achieves a new SOTA performance on MiniGPT-4, and demonstrates performance comparable to existing techniques on LLaVA-1.5 and mPLUG-Owl2. In terms of average performance across all backbones, our method outperforms previous techniques. This indicates that our approach consistently delivers strong performance across various backbones.

While we achieves overall strong performance on this benchmark, the performance improvements across different backbone models are relatively modest. This might be because the POPE question split does not fully align with the types of hallucinations that T2I models generate. POPE

[Table 5 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 5: Evaluation results on LLaVA-Bench using LLaVA-1.5.

Summary: This table reports evaluation results on LLaVA-Bench using the LLaVA-1.5 model, comparing seven decoding methods (Greedy Search, Nucleus Sampling, Beam Search, VCD, OPERA, HALC, and Ours) across four metrics: Complex, Conv, Detail, and All. The scores are numeric values likely representing quality or accuracy percentages.

Table LaTeX:

```latex
\begin{tabular}{lllll}
\hline
Method & Complex & Conv & Detail & All \\
\hline
Greedy Search & 82.0 & 47.3 & 64.1 & 67.0 \\
Nucleus Sampling & 76.2 & 41.2 & 52.6 & 59.9 \\
Beam Search & 83.9 & 58.7 & 58.8 & 70.0 \\
VCD & 79.9 & 53.5 & 56.3 & 66.2 \\
OPERA & 78.7 & 53.0 & 58.3 & 66.0 \\
HALC & 55.8 & 31.1 & 50.4 & 47.1 \\
Ours & 84.2 & 63.5 & 64.8 & 73.3 \\
\hline
\end{tabular}
```

questions, which ask, “Is this [object] in this image?” sample objects randomly, popularly, or adversarially. Meanwhile, our method visualizes hallucinations in captions generated by prompts like “Please describe this image in detail.” As a result, T2I model may visualize the objects unrelated to the actual POPE questions which limits our method’s effectiveness. This limitation will be explored further through a qualitative analysis of POPE samples later in this section.

Results on MME. In Table 4, we present the evaluation results on the MME benchmark using the LLaVA-1.5 backbone. Due to space limitations, we focus on the performance in the two main categories of the MME benchmark: Perception and Cognition. Scores for the subcategories are provided in the Appendix. Our method outperforms all others in the Perception category, demonstrating its effectiveness in accurately interpreting and processing visual information across various tasks. This strong performance indicates that our model is particularly well-suited for visual tasks, making it highly effective for applications that require precise visual understanding. In the Cognition category, our method demonstrates competitive performance, comparable to OPERA and superior to HALC, further underscoring the versatility and robustness of our approach. While VCD excels in cognitive tasks, our method achieves stronger overall performance when both the Perception and Cognition categories are considered together. This suggests that our model provides a more comprehensive and effective solution across diverse tasks. Its balanced and reliable performance in both visual and cognitive challenges makes it an adaptable solution for a wide range of applications.

Results on LLaVA-Bench. Table 5 shows the experimental results on the LLaVA-Bench, which verify whether the language model capabilities are preserved. For this evaluation, we use the LLaVA-1.5 backbone. Our method outperforms existing techniques across all categories: complex reasoning, conversation, and detailed description. These results demonstrate that our method effectively mitigates hallucinations while also enhancing the performance of the MLLM.

## Analysis and Discussion

Diversity of Generated Captions and Images. Although T2I models have made significant advancements, they still struggle to generate images that perfectly align with the given captions (Ruiz et al. 2023). To address these limitations, we increase the coverage of hallucination visualizations.

[Table 6 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 6: Our performance when differentiating the T2I models for visualizing hallucinations. We generate captions with nucleus sampling and set max new token for 64 and generate the image with those captions. Inference step for diffusion set to be all 1.

Summary: This table compares the performance of three text-to-image (T2I) models (Hyper-SD1.5, SDXL-Turbo, Hyper-SDXL) in terms of CLIPScore, CHAIRs, and CHAIRI when evaluated with three different LLMs (LLaVA-1.5, mPLUG-Owl2, MiniGPT-4). The T2I models generate images from captions produced by the LLMs, and all diffusion inference steps are set to 1.

Table LaTeX:

```latex
\begin{tabular}{llllllll}
\hline
T2I Model & CLIPScore \$ \textbackslash{}uparrow \$ & LLaVA-1.5 & LLaVA-1.5 & mPLUG-Owl2 & mPLUG-Owl2 & MiniGPT-4 & MiniGPT-4 \\
\hline
T2I Model & CLIPScore \$ \textbackslash{}uparrow \$ & CHAIRs \$ \textbackslash{}downarrow \$ & CHAIRI \$ \textbackslash{}downarrow \$ & CHAIRs \$ \textbackslash{}downarrow \$ & CHAIRI \$ \textbackslash{}downarrow \$ & CHAIRs \$ \textbackslash{}downarrow \$ & CHAIRI \$ \textbackslash{}downarrow \$ \\
Hyper-SD1.5 & 30.87 & 20.2 & 6.6 & 19.4 & 6.4 & 28.2 & 11.8 \\
SDXL-Turbo & 32.33 & 18.8 & 6.6 & 20.2 & 6.68 & 25.2 & 9.9 \\
Hyper-SDXL & 32.85 & 17 & 5.6 & 17 & 5.3 & 24.4 & 10.0 \\
\hline
\end{tabular}
```

[Table 7 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 7: Comparison of performance using a single image $ (n = 1) $ generated by two different decoding strategies, Greedy search and Nucleus sampling.

Summary: Table 7 reports CHAIR_S and CHAIR_I scores for three vision-language models (LLaVA-1.5, mPLUG-Owl2, MiniGPT-4) using two decoding strategies (Greedy Search and Nucleus Sampling) on a single image (n=1). Lower scores indicate less hallucination.

Table LaTeX:

```latex
\begin{tabular}{llll}
\hline
Captioning by & LLaVA-1.5 & mPLUG-Owl2 & MiniGPT-4 \\
\hline
CHAIR \$ \_\{S\} \$ \$ \textbackslash{}downarrow \$ & CHAIR \$ \_\{S\} \$ \$ \textbackslash{}downarrow \$ & CHAIR \$ \_\{S\} \$ \$ \textbackslash{}downarrow \$ & CHAIR \$ \_\{S\} \$ \$ \textbackslash{}downarrow \$ \\
Greedy Search & 19.4 & 19.4 & 27.2 \\
Nucleus Sampling & 18.8 & 15.2 & 24.4 \\
CHAIR \$ \_\{I\} \$ \$ \textbackslash{}downarrow \$ & CHAIR \$ \_\{I\} \$ \$ \textbackslash{}downarrow \$ & CHAIR \$ \_\{I\} \$ \$ \textbackslash{}downarrow \$ & CHAIR \$ \_\{I\} \$ \$ \textbackslash{}downarrow \$ \\
Greedy Search & 6.6 & 6.4 & 11.6 \\
Nucleus Sampling & 6.7 & 5.1 & 10.3 \\
\hline
\end{tabular}
```

[Figure 4 was here. The original paper contained a figure at this position. Brief visual description: Figure 4 consists of two line charts (Panel 1 and Panel 2) analyzing the effect of the number of images (n) on model performance. It compares three models: MiniGPT-4, LLaVA-1.5, and mPLUG-Owl2. Both panels feature a broken y-axis to separate MiniGPT-4's higher scores from the other models. Panel 1 displays data for MiniGPT-4 and mPLUG-Owl2, while Panel 2 displays data for all three models.]
Caption: Figure 4: Effect of the number of images with different captions.
Key visible elements:
- MiniGPT-4: Model represented by the blue line, consistently showing the highest scores in both panels.
- mPLUG-Owl2: Model represented by the green line, showing lower scores than MiniGPT-4 in both panels.
- LLaVA-1.5: Model represented by the orange line, visible in Panel 2 but absent in Panel 1.
- n: X-axis variable representing the number of images (values 1, 2, 3, 4).
- Broken Y-axis: Visual separator indicating a discontinuity in scale between MiniGPT-4 and the other models.

tion by generating diverse images. Specifically, we use Nucleus sampling, which is known for producing more varied responses than Greedy search, to generate multiple captions. These captions are then utilized to generate images.

To evaluate the effectiveness of this strategy, we analyze how caption diversity impacts hallucination reduction. First, we compare the CHAIR scores of the final responses when using Greedy search and Nucleus sampling during the image generation stage. In this experiment, we limit the number of generated images to one and compare which decoding strategy performs better. As shown in Table 7, Nucleus sampling outperforms Greedy search, demonstrating its potential to generate more diverse captions. Furthermore, in Figure 4, we investigate how the number of generated images from different captions using Nucleus sampling affects CHAIR scores. We observe that the number of images n increases, both CHAIRs and CHAIR scores improve, confirming that using multiple reconstructed images, rather than

[Figure 5 was here. The original paper contained a figure at this position. Brief visual description: The figure is a bar chart displaying the Kullback-Leibler (KL) divergence values for a sequence of tokens: 'ing', 'over', 'a', 'car', 'in', 'a', and 'par'. The chart visually demonstrates that the token 'car' has a significantly higher KL divergence compared to all other tokens shown, with its bar extending beyond the value of 6 on the y-axis.]
Caption: Figure 5: KL divergence between output distributions across each decoding step when the MLLM is provided with the images and caption from Figure 6 (a). The KL divergence is significantly elevated for the hallucinated token “car”.
Key visible elements:
- KL Divergence: Y-axis label indicating the metric measured
- car: X-axis label corresponding to the tallest bar
- par: X-axis label corresponding to the second tallest bar
- ing, over, a, in: X-axis labels corresponding to shorter bars

a single image, is more effective for improving performance. These findings validate our design choice of utilizing Nucleus sampling and multiple captions for image generation.

Impacts of Image Generation Quality. To investigate the impact of generated image quality on hallucination mitigation, we evaluate the performance of our method using various text-to-image (T2I) models. Table 6 presents the generation quality (CLIPScore) of the T2I models alongside their corresponding CHAIR scores. We compare three T2I models: Hyper-SD1.5 (Ren et al. 2024), SDXL-Turbo (Sauer et al. 2023), and Hyper-SDXL (Ren et al. 2024), with the inference step fixed at 1.

The results indicate a clear trend: as the CLIPScore improves, so does the CHAIR score. Notably, SDXL-Turbo consistently outperforms Hyper-SD1.5 across all backbones, except for mPLUG-Owl2. Moreover, Hyper-SDXL significantly outperforms Hyper-SD1.5 in all cases. These findings suggest that using higher-quality T2I models, which are better aligned with the original captions, can more effectively mitigate hallucination issues. Consequently, we believe that as more advanced T2I models are developed, the performance of our method will continue to improve.

Qualitative Analysis. Figure 5 shows the KL divergence between output distributions at each decoding step when the images and caption from Figure 6 (a) are provided to the MLLM. We observe that the KL divergence is high for the hallucinated token car, while non-hallucinated tokens exhibit lower KL divergence. This indicates that the generated image can produce visual contrastive signals for hallucinated tokens when compared to the original image. This supports our argument that the differences between the original

[None None was here. The original paper contained a figure at this position. Brief visual description: The figure displays two comparative examples involving image generation and understanding. The left panel contrasts an 'Original Image' of a black and white dog jumping a pool with a 'Generated Image' of a brown dog jumping a car, accompanied by the caption used for the generated image and a description ('Ours Decoding') of the original image. The right panel shows a fruit stand scene (labeled both Original and Generated), its generation caption, and a Visual Question Answering (VQA) task where the model ('Ours Decoding') correctly answers 'No' to the presence of a bed.]
Key visible elements:
- Original Image (Left): A real photograph of a black and white dog jumping over a swimming pool.
- Generated Image (Left): An AI-generated image of a brown dog jumping over a car in a parking lot.
- Caption used for image generation (Left): Text describing the content of the Generated Image (dog jumping car).
- Ours Decoding (Left): Text describing the content of the Original Image (dog jumping pool).
- Original Image (Right): A photograph of a fruit stand filled with oranges.
- Generated Image (Right): A generated image of a fruit stand, appearing very similar to the original.
- Caption used for image generation (Right): Text describing the fruit stand scene.
- Question: A VQA prompt asking if there is a bed in the image.

## Caption used for image generation

The image captures a dog skillfully jumping over a car in a parking lot. The dog is in mid-air, showcasing its agility and athleticism. The car is positioned in the middle of the scene, with the dog jumping over it.

## Ours Decoding

The image captures a black and white dog skillfully jumping over a pool in an outdoor swimming pool competition. The dog is in mid-air, showcasing its agility and athleticism. There are several people watching the dog's performance from various positions around the pool. Some ...

## Caption used for image generation

The image features a group of people gathered around a fruit stand at a market, particularly focusing on oranges. The stand is filled with a variety of oranges, with some placed on a table and others spread out across the scene.

Question

There a bed in the image? Please answer yes or no.

Greedy Decoding

No, there is no bed in the image.

Ours Decoding

No, there is no bed in the image.

Figure 6: Qualitative samples using LLaVA-1.5 for backbone model. (a) shows an example that T2I model faithfully generate the images that depicts the hallucination in the caption. (e.g., jumping over a car) (b) is an example of our limitation in VQA tasks, which there can be a misalignment between visualized hallucination and actual main subject of question.

inal and generated images are primarily influenced by the hallucinated tokens.

To more clearly demonstrate how our method mitigates multimodal hallucinations, we present an example in Figure 6 (a), illustrating the process from the initial hallucinated caption to the generated image, followed by the contrastive decoding result. Specifically, for an image of a dog jumping into a pool, the MLLM incorrectly describes the scene as “a dog jumping over a car in a parking lot.” Using this caption, the T2I model generates a reconstructed image that faithfully visualized the hallucinated content. By contrasting the distributions of the reconstructed and original images during decoding, our method effectively reduces hallucinations.

Limitations. One of the key limitations of our approach is its strong dependence on T2I generation models. This reliance may hinder effectiveness in tasks like VQA, where the generated captions can sometimes contain hallucinations that deviate significantly from the specific question. This limitation is particularly evident in our experiments with the POPE benchmark, where the performance gain is not as significant as expected. Regarding questions about the presence of specific objects, if the object in question is not related to the hallucinations generated by the caption, visualizing with a T2I model may not sufficiently reflect the information needed for the VQA task. In Figure 6 (b), a question about the presence of a bed in an original image where people are looking at fruits might not be well served by the reconstructed image. This indicates the effectiveness of our method may decrease for certain type of questions.

## Conclusion

Currently, our technique employs a fixed prompt for image captioning. However, we believe that adapting the prompt to respond more specifically to the given question could mitigate this issue. We plan to explore this adaptive approach in future work.

In this paper, we presented ConVis, a novel contrastive decoding method designed to mitigate hallucinations in MLLMs. By utilizing a T2I generation model, our approach effectively visualizes hallucinations and contrasts probability distributions between the original and reconstructed images. This process allows for the penalization of hallucinated content during the decoding phase, all without the need for additional data or model retraining.

Our extensive experiments across five benchmarks, including CHAIR, HallusionBench, and LLaVA-Bench, demonstrated that ConVis consistently reduces hallucinations while preserving the core language model capabilities of MLLMs. The method achieves competitive or superior performance compared to existing techniques in various categories, validating its effectiveness in enhancing the reliability of MLLM outputs.