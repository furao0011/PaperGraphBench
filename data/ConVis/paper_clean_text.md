<!-- page 1: doc_0.md -->

# ConVis: Contrastive Decoding with Hallucination Visualization for Mitigating Hallucinations in Multimodal Large Language Models
Yejì Park $ ^{*1} $, Deokyeong Lee $ ^{*1} $, Junsuk Choe $ ^{\dagger1} $, Buru Chang $ ^{\dagger2} $
## Abstract

Hallucinations in Multimodal Large Language Models (MLLMs) where generated responses fail to accurately reflect the given image pose a significant challenge to their reliability. To address this, we introduce ConVis, a novel training-free contrastive decoding method. ConVis leverages a text-to-image (T2I) generation model to semantically reconstruct the given image from hallucinated captions. By comparing the contrasting probability distributions produced by the original and reconstructed images, ConVis enables MLLMs to capture visual contrastive signals that penalize hallucination generation. Notably, this method operates purely within the decoding process, eliminating the need for additional data or model updates. Our extensive experiments on five popular benchmarks demonstrate that ConVis effectively reduces hallucinations across various MLLMs, highlighting its potential to enhance model reliability.

## Introduction

Multimodal Large Language Models (MLLMs) (Dai et al. 2023; Liu et al. 2024b) are advanced language models capable of understanding both images and text, such as image captioning and visual question answering (VQA). While MLLMs have achieved significant success that utilize both visual and textual information, the issue of hallucination, where the models generate responses that do not align with the given image, has greatly undermined their reliability (Liu et al. 2023a; Sun et al. 2024). This problem poses a significant obstacle to adopting MLLMs in critical fields where reliability is crucial. For instance, in medical applications, it could lead to incorrect diagnoses (Liu et al. 2023b), while in MLLM-based autonomous systems, it might result in erroneous interpretations (Shao et al. 2024).

Recent research has been actively conducted to address this. WoodPecker (Yin et al. 2023) and LURE (Zhou et al. 2024) reduce hallucinations by post-processing the generated responses. Datasets such as LRV-Instruction (Liu et al. 2023a) and RLHF-V (Yu et al. 2024) have been proposed to mitigate hallucinations through instruction tuning of MLLMs. However, these studies often rely on external

 

 

APIs like GPT-3.5, require costly human feedback collection, and necessitate additional training of MLLMs.

In contrast, this paper focuses on decoding strategies that reduce hallucinations by intervening solely in the decoding process, without the need for additional data or model training. The following studies fall into this category: OPERA (Huang et al. 2024) imposes penalties on token generation that does not reference visual tokens. VCD (Leng et al. 2024) creates contrasting distributions using distorted images to reduce the model's reliance on statistical biases and priors that lead to hallucinations. HALC (Chen et al. 2024) corrects hallucinations by leveraging cues provided by visual information from various fields of view.

In this study, we propose a contrastive decoding method called ConVis (Contrastive Decoding with Hallucination Visualization), which can be applied to any existing MLLM without additional training. Inspired by the previous work (Kim et al. 2024), ConVis leverages text-to-image (T2I) generation models, specifically Hyper-SDXL (Ren et al. 2024), to capture visual contrast signals. The process begins with the MLLM generating a caption for the input image, after which the T2I model reconstructs an image based on this caption. As shown in Figure 1, if the generated caption contains hallucinations (e.g., a book'), there will be visual discrepancies between the original and reconstructed

<!-- page 2: doc_1.md -->

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

<!-- page 3: doc_2.md -->

where $ \theta $ denotes the parameters of the MLLM, $ y_t $ represents the $ t $-th token of response, and $ y_{<t} $ is the sequence of tokens generated up to time $ t $. $ f_\theta $ denotes the logit distribution generated by the MLLM. Hallucination refers to the phenomenon where the output $ y $ generated by the MLLM does not correspond to the input image $ v $. This study focuses on mitigating hallucinations while maintaining the overall performance of the MLLM as a language model.

Text-to-Image Generation. The core component of ConVis is the T2I model that generates images based on a given query. The goal of the T2I model is to create an image that accurately depicts the query. Among the recently proposed T2I models, we utilize Hyper-SDXL (Ren et al. 2024), an enhanced version of Stable Diffusion (Ho, Jain, and Abbeel 2020), which has demonstrated excellent T2I performance. The diffusion-based Hyper-SDXL model begins with a pure noise and progressively reconstructs it through an iterative reverse diffusion process which ultimately results in the generated image $ v_{0}^{\prime} $.

## Hallucination Visualization

We hypothesize that the T2I model can help mitigate hallucinations by providing visual contrast signals during the decoding process. If the T2I model receives a caption generated by the MLLM that contains hallucinations, it will faithfully visualize those hallucinations in the generated image. We refer to this process as hallucination visualization.

To implement this, ConVis first generates an initial caption c for the original image v using a simple instruction text that directs the MLLM to describe the image. This process is illustrated in 

Diversity of Generated Images. Given that the current T2I model may not generate images that fully align with the captions, we address this limitation by increasing the diversity of the generated images using the following approaches: (1) We first generate a diverse set of n captions using Nucleus Decoding (Holtzman et al. 2020) instead of Greedy Decoding. (2) Then, the T2I model uses these n captions to generate n corresponding images. This approach increases coverage of the various potential hallucinations that the MLLM might generate by diversifying the captions. Additionally, by using multiple images instead of a single one, we enhance the robustness of our method against the T2I model's potential misalignment between the caption and the generated image due to its imperfect performance.

We have found these approaches to be effective, with detailed results available in the experiment section.

## Contrastive Decoding

Hallucinations in captions cause visual differences between the original image v and the generated image v'. We mitigate these hallucinations by capturing the visual contrast signals from these differences. To achieve this, during the decoding process, we utilize both the original image v and the n generated images to produce the logit distribution for each image. The final contrastive logit distribution $ \hat{f}_{\theta} $ is derived by averaging the contrastive logit distributions between the original image and each generated image as follows:

 $$ \hat{f}_{\theta}=\frac{1}{n}\sum_{i=1}^{n}\Big((1+\alpha)f_{\theta}(\cdot|v,x,y_{<t})-\alpha f_{\theta}(\cdot|v_{i}^{\prime},x,y_{<t})\Big), $$ 

where $ \alpha $ is a hyperparameter that controls the strength of the difference between the logit distributions from the original and generated images. The contrastive logit distribution $ \hat{f}_{\theta} $ is used to generate the response y. For tokens associated with hallucinations, the contrastive logit distribution is significantly amplified compared to other tokens, allowing us to penalize these tokens and reduce the hallucinations.

Note that, Equation 2 is similar to the contrastive decoding methods used in VCD (Leng et al. 2024) and HALC (Chen et al. 2024). However, our method is distinguished from existing approaches by directly capturing visual contrastive signals from the hallucinations visualized by the T2I generative model.

## Experiments

Benchmarks. To evaluate the performance of our method, we conduct experiments on three benchmarks to evaluate the

<!-- page 4: doc_3.md -->

mitigation of hallucinations and two general-purpose benchmarks to assess the general performance of the MLLM:

• Hallucination: CHAIR (Rohrbach et al. 2018), HallusionBench (Guan et al. 2024), and Polling-based Object Probing Evaluation (POPE) (Li et al. 2023c)

• General-purpose: MLLM Evaluation (MME) (Fu et al. 2023) and LLaVA-Bench (Liu et al. 2024b)

Detailed information on these benchmarks can be found in the Appendix.

Backbones. To evaluate our method, we utilize three well-known MLLMs with publicly available checkpoint weights: LLaVA-1.5 (Liu et al. 2024a), mPLUG-Owl2 (Ye et al. 2024), and MiniGPT-4 (Zhu et al. 2024).

Compared Methods. Our method is designed to replace existing decoding methods used in the LLM component, and therefore, we compare it against baselines such as Greedy Search, Nucleus Sampling (Holtzman et al. 2020), and Beam Search (beam=5). We also evaluate our method's effectiveness against other decoding methods in hallucination mitigation, including OPERA (Huang et al. 2024), VCD (Leng et al. 2024), and HALC (Chen et al. 2024). We use the same hyperparameters borrowed from the original papers of the compared methods to ensure a fair comparison.

Implementation Details. We utilize the Hyper-SDXL (Ren et al. 2024) T2I model for image generation. Specifically, in all experiments, unless otherwise noted, we use the Step 1 generation results of Hyper-SDXL model. The maximum length of text queries that the T2I model could accept is 77 tokens, which is too short to process the captions generated by MLLM. To address this, we leverage Compel (Stewart 2023), which allows for processing more than 77 tokens. We set the maximum token count for the caption generation to 256 and use Nucleus sampling with a temperature of 0.7 and a top-p of 0.9 to generate the images. The query used in this process is “Please describe this image in detail.” We set the number of generated images, n, to 4, producing four images based on distinct captions generated using different random seeds. For contrastive decoding, we follow (Li et al. 2023b) using adaptive plausibility constraint to contrast only meaningful tokens. The plausibility constraint hyperparameter $ \lambda $ is set to 0.1. We also set $ \alpha $, which controls the degree of contrastive emphasis, to 1 for captioning-based metrics such as

 

 

CHAIR and LLaVA-Bench, and to 0.1 for VQA metrics, including POPE, HallusionBench, and MME. To generate responses, we use a greedy decoding approach for all methods. For CHAIR, we sample three different sets of images using different random seeds and assess the performance using the mean and standard deviation of these results.

## Experimental Results

Results on CHAIR. We report our evaluation results on the CHAIR (Rohrbach et al. 2018) benchmark in 

Results on HallusionBench. In Table 2, we present the evaluation results for the visual dependent category of the HallusionBench (Guan et al. 2024) benchmark. Hallusion-

<!-- page 5: doc_4.md -->

Bench is evaluated with the assistance of GPT-4V, which incurs significant costs; therefore, we conduct experiments using only the LLaVA-1.5 (Liu et al. 2024a) backbone. Our method demonstrates superior performance in Figure Accuracy (fAcc), outperforming all baseline decoding strategies (Greedy Search, Nucleus Sampling, Beam Search) as well as state-of-the-art techniques (VCD, OPERA, HALC). This indicates that our model effectively interprets the visual details of images when responding to visually dependent questions, indicating its ability to mitigate hallucinations by providing responses that closely align with the given visual content. Furthermore, our method achieves the highest performance on the All Accuracy (aAcc) metric, which measures overall accuracy across all questions within the visual dependent category, demonstrating its effectiveness in handling a wide range of visually dependent queries.

Results on POPE. 

Our method achieves a new SOTA performance on MiniGPT-4, and demonstrates performance comparable to existing techniques on LLaVA-1.5 and mPLUG-Owl2. In terms of average performance across all backbones, our method outperforms previous techniques. This indicates that our approach consistently delivers strong performance across various backbones.

While we achieves overall strong performance on this benchmark, the performance improvements across different backbone models are relatively modest. This might be because the POPE question split does not fully align with the types of hallucinations that T2I models generate. POPE

 

 

questions, which ask, “Is this [object] in this image?” sample objects randomly, popularly, or adversarially. Meanwhile, our method visualizes hallucinations in captions generated by prompts like “Please describe this image in detail.” As a result, T2I model may visualize the objects unrelated to the actual POPE questions which limits our method’s effectiveness. This limitation will be explored further through a qualitative analysis of POPE samples later in this section.

Results on MME. In Table 4, we present the evaluation results on the MME benchmark using the LLaVA-1.5 backbone. Due to space limitations, we focus on the performance in the two main categories of the MME benchmark: Perception and Cognition. Scores for the subcategories are provided in the Appendix. Our method outperforms all others in the Perception category, demonstrating its effectiveness in accurately interpreting and processing visual information across various tasks. This strong performance indicates that our model is particularly well-suited for visual tasks, making it highly effective for applications that require precise visual understanding. In the Cognition category, our method demonstrates competitive performance, comparable to OPERA and superior to HALC, further underscoring the versatility and robustness of our approach. While VCD excels in cognitive tasks, our method achieves stronger overall performance when both the Perception and Cognition categories are considered together. This suggests that our model provides a more comprehensive and effective solution across diverse tasks. Its balanced and reliable performance in both visual and cognitive challenges makes it an adaptable solution for a wide range of applications.

Results on LLaVA-Bench. 

## Analysis and Discussion

Diversity of Generated Captions and Images. Although T2I models have made significant advancements, they still struggle to generate images that perfectly align with the given captions (Ruiz et al. 2023). To address these limitations, we increase the coverage of hallucination visualizations.

<!-- page 6: doc_5.md -->

tion by generating diverse images. Specifically, we use Nucleus sampling, which is known for producing more varied responses than Greedy search, to generate multiple captions. These captions are then utilized to generate images.

To evaluate the effectiveness of this strategy, we analyze how caption diversity impacts hallucination reduction. First, we compare the CHAIR scores of the final responses when using Greedy search and Nucleus sampling during the image generation stage. In this experiment, we limit the number of generated images to one and compare which decoding strategy performs better. As shown in Table 7, Nucleus sampling outperforms Greedy search, demonstrating its potential to generate more diverse captions. Furthermore, in Figure 4, we investigate how the number of generated images from different captions using Nucleus sampling affects CHAIR scores. We observe that the number of images n increases, both CHAIRs and CHAIR scores improve, confirming that using multiple reconstructed images, rather than

 

 

a single image, is more effective for improving performance. These findings validate our design choice of utilizing Nucleus sampling and multiple captions for image generation.

Impacts of Image Generation Quality. To investigate the impact of generated image quality on hallucination mitigation, we evaluate the performance of our method using various text-to-image (T2I) models. 

The results indicate a clear trend: as the CLIPScore improves, so does the CHAIR score. Notably, SDXL-Turbo consistently outperforms Hyper-SD1.5 across all backbones, except for mPLUG-Owl2. Moreover, Hyper-SDXL significantly outperforms Hyper-SD1.5 in all cases. These findings suggest that using higher-quality T2I models, which are better aligned with the original captions, can more effectively mitigate hallucination issues. Consequently, we believe that as more advanced T2I models are developed, the performance of our method will continue to improve.

Qualitative Analysis.

<!-- page 7: doc_6.md -->

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

 

 

inal and generated images are primarily influenced by the hallucinated tokens.

To more clearly demonstrate how our method mitigates multimodal hallucinations, we present an example in 

Limitations. One of the key limitations of our approach is its strong dependence on T2I generation models. This reliance may hinder effectiveness in tasks like VQA, where the generated captions can sometimes contain hallucinations that deviate significantly from the specific question. This limitation is particularly evident in our experiments with the POPE benchmark, where the performance gain is not as significant as expected. Regarding questions about the presence of specific objects, if the object in question is not related to the hallucinations generated by the caption, visualizing with a T2I model may not sufficiently reflect the information needed for the VQA task. In 

## Conclusion

Currently, our technique employs a fixed prompt for image captioning. However, we believe that adapting the prompt to respond more specifically to the given question could mitigate this issue. We plan to explore this adaptive approach in future work.

In this paper, we presented ConVis, a novel contrastive decoding method designed to mitigate hallucinations in MLLMs. By utilizing a T2I generation model, our approach effectively visualizes hallucinations and contrasts probability distributions between the original and reconstructed images. This process allows for the penalization of hallucinated content during the decoding phase, all without the need for additional data or model retraining.

Our extensive experiments across five benchmarks, including CHAIR, HallusionBench, and LLaVA-Bench, demonstrated that ConVis consistently reduces hallucinations while preserving the core language model capabilities of MLLMs. The method achieves competitive or superior performance compared to existing techniques in various categories, validating its effectiveness in enhancing the reliability of MLLM outputs.

<!-- page 8: doc_7.md -->

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2024-00350430) and Hyundai Motor Chung Mong-Koo Foundation. The authors sincerely thank the members of the Visual Representation Lab for their assistance throughout this project. You can find the full version of the paper including the Appendix on https://arxiv.org/abs/2408.13906.

<!-- page 9: doc_8.md -->

Yin, S.; Fu, C.; Zhao, S.; Xu, T.; Wang, H.; Sui, D.; Shen, Y.; Li, K.; Sun, X.; and Chen, E. 2023. Woodpecker: Hallucination correction for multimodal large language models. arXiv preprint arXiv:2310.16045.

Yu, T.; Yao, Y.; Zhang, H.; He, T.; Han, Y.; Cui, G.; Hu, J.; Liu, Z.; Zheng, H.-T.; Sun, M.; et al. 2024. Rlhf-v: Towards trustworthy mlrms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13807–13816.

Zhou, Y.; Cui, C.; Yoon, J.; Zhang, L.; Deng, Z.; Finn, C.; Bansal, M.; and Yao, H. 2024. Analyzing and Mitigating Object Hallucination in Large Vision-Language Models. In International Conference on Learning Representations.

Zhu, D.; Chen, J.; Shen, X.; Li, X.; and Elhoseiny, M. 2024. MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. In International Conference on Learning Representations.