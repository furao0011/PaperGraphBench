# Pioneering Explainable Video Fact-Checking with a New Dataset and Multi-role Multimodal Model Approach
## Abstract

Existing video fact-checking datasets often lack detailed evidence and explanations, compromising the reliability and interpretability of fact-checking methods. To address these gaps, we developed a novel dataset featuring comprehensive annotations for each news item, including veracity labels, the rationales behind these labels, and supporting evidence. This dataset significantly enhances models' ability to accurately identify and explain video content. We also present an explainable automatic framework 3MFact, utilizing Multi-role Multimodal Models for video Fact-checking. Our framework iteratively gathers and synthesizes online evidence to progressively determine the veracity label, generating three key outputs: veracity label, rationale, and supported evidence. We aim for this work to be a pioneering effort, providing robust support for the field of video fact-checking.

## 1 Introduction

Misinformation has been a persistent issue since the rise of digital media, with large models exacerbating the problem through their advanced text generation capabilities, enabling the creation of highly persuasive, multimodal misinformation (Kasneci et al. 2023; Xu, Fan, and Kankanhalli 2023). Despite numerous detection techniques and various fact-checking tools, these measures often fall short. The impressive AI-generated content (AIGC) can make simplistic classification results seem trivial, and studies show that merely labeling content as misinformation has limited persuasive power on affected users (Sanderson, Farrell, and Ecker 2022). In contrast, providing correct answers alongside errors can significantly enhance the corrective impact (Mera, Rodríguez, and Marin-García 2022; Mullet and Marsh 2016). To increase persuasiveness and retention, it is crucial to offer convincing rationale and robust evidence.

Early feature-based supervised models often struggle to fully capture the context of specific claims, rendering them less effective against unseen or complex misinformation. While fact-checking websites like Snopes and PolitiFact can

[Figure 1 was here. The original paper contained a figure at this position. Brief visual description: Figure 1 illustrates a sample entry from the TRUE Dataset designed for explainable video fact-checking. The figure is structured into multiple sections displaying a multimodal input: a textual claim, a ground truth label (marked FALSE), video content frames, and video metadata (headline, date, platform). It also details the reasoning process through 'Original Rationale', 'Summary Rationale' (with detailed reasons), and specific 'Evidences'.]
Caption: Figure 1: A sample in the proposed TRUE Dataset. It includes the claim, video, and video background information. Besides, three types of annotations are provided: 1) label, 2) evidences, and 3) original and summary rationales.
Key visible elements:
- Claim: The statement being verified (e.g., F-18 breaking sound barrier)
- Label: The ground truth verification result (e.g., FALSE)
- Video Content: Visual frames extracted from the video evidence
- Video Information: Metadata about the video source including headline, date, and platform
- Original Rationale: Initial human-provided reasoning steps
- Summary Rationale: Synthesized conclusion and detailed supporting reasons
- Evidences: Specific factual points supporting the rationale

verify suspicious claims using external evidence, they heavily depend on human labor, making them impractical for addressing the vast volume of AI-generated misinformation. Although zero-shot methods using large language models (LLMs) have been applied to fact-checking, they often focus on isolated text (Pan et al. 2023; Zhang and Gao 2023), limiting their effectiveness in multimodal scenarios. While recent multimodal models address text-image tasks (Tahmasebi, Müller-Budack, and Ewerth 2024; Liu et al. 2024a), fact-checking for video-based content remains largely unexplored. Moreover, many early approaches lack comprehensive methodological rigor and robust experimental validation, leading to unstable and uncertain performance.

Furthermore, existing fact-checking datasets that include evidence or rationales are limited and suffer from several drawbacks: 1) Most explainable fact-checking datasets focus primarily on text modality, with limited attention to the

[Table 1 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 1: Comparison of the proposed TRUE with other datasets. ECR indicates the Expert-Crafted Rationale, LSR stands for the specific LLM-Summary-Rationale. 1;2 See Figure 2(a) for the sub_labels and their sizes. Abbreviations: T: text, A: audio, V: video, I: image. ⚖: Not Enough Information, ✓: True (Real), ∅: False (Fake/Misleading), ✰: Debunk

Summary: This table compares the proposed TRUE dataset with five existing datasets (CHECKED, Shang et al. 2021, Mocheg, FakeSV, VMH) across modalities (T, I, A, V), label types (Categories, ECR, LSR, Evidence), time span, and dataset size (True/False counts). It demonstrates that TRUE is the only dataset that includes ECR, LSR, and Evidence labels, and covers text, audio, and video but not image. Other datasets vary in coverage and included explanation types.

Table LaTeX:

```latex
\begin{tabular}{lllllllllll}
\hline
Dataset & Modality & Modality & Label & Label & Explanation Related & Explanation Related & Time & Time & Size & Size \\
\hline
Dataset & T & I & A & V & Categories & ECR & LSR & Evidence & 2,000+ & True / False \\
CHECKED & ✓ & ✓ & ✗ & ✓ & ☑, ☑ & ✗ & ✗ & ✗ & 19-20 & 1760 ✓ / 344 ☑ \\
(Shang et al. 2021) & ✓ & ✗ & ✓ & ✓ & ☑, ☑ & ✗ & ✗ & ✗ & 21-21 & 665 ✓ / 226 ☑ \\
Mocheg & ✓ & ✓ & ✗ & ✗ & ☑, ☑, ☑ & ✗ & ✗ & ✓ & - & 5,144 ✓ / 5,855 ✓ / 4,602 ☑ \\
FakeSV & ✓ & ✗ & ✓ & ✓ & ☑, ☑, ☑ & ✗ & ✗ & ✓ & 17-22 & 1827 ✓ / 1,827 ✓ / 1,884 ☑ \\
VMH & ✓ & ✗ & ✓ & ✓ & ☑, ☑ & ✓ & ✗ & ✗ & 14-16 & 341 ✓ / 1,906 ☑ \\
TRUE (Ours) & ✓ & ✗ & ✓ & ✓ & ☑, ☑ + 8 sub 1 & ✓ & ✓ & ✓ & 16-24 & 1,097 ✓ / 1,828 ☑ 2 \\
\hline
\end{tabular}
```

multimodal nature of online media, particularly video content (Yao et al. 2023). 2) Some datasets include machine-generated evidence and ratings, raising concerns about their reliability (Mishra et al. 2022; Abdelnabi, Hasan, and Fritz 2022). 3) Rationales and evidence are seldom well-organized within a single dataset, which undermines the robustness of the labels.

To address these challenges, we propose a zero-shot video fact-checking framework 3MFact (Multi-role Multimodal Models Fact-checking), that leverages both video and text information through Large Multimodal Models (LMMs) and LLMs. This framework analyzes both internal and external information sourced from online search, to automatically fact-check unseen and complex multimodal content without requiring additional human intervention. To support this effort and benefit the video fact-checking community, we have also developed a comprehensive dataset, TRUE (Truthfulness and Rationale with Underlying Evidence). As illustrated in Figure 1, this dataset includes rating labels, detailed reasons, evidence for each claim, featuring both original human rationales and LLM-summarized rationales sourced from reliable fact-checking websites. Extensive experiments using both traditional and novel reasoning-related metrics demonstrate that our framework produces more accurate results, supported by well-founded reasoning and robust evidence. Our contributions focus on three key areas:

• Exploratory Video Fact-Checking Dataset: We present the first explainable video fact-checking dataset designed to address general video-related claims. This dataset emphasizes the importance of explainable analysis supported by robust evidence and is the first to include clearly organized reasons (Human and LLM versions) directly linked to well-sourced evidence.

• Insightful Framework: We introduce a multi-role multimodal models framework 3MFact with a structured division of labor, where specific models are assigned tasks such as evidence retrieval, reasoning, and explanation. It addresses unseen multimodal misinformation by using text, video, and image data from both internal and external sources, ensuring transparent and thorough decision-making while reducing overlooking critical details.

- Innovative Standard: We establish novel metrics and a benchmark for multimodal fact-checking. Our metrics assess both accuracy and the quality of reasoning and evidence, setting a new standard for evaluating fact-checking systems in real-world scenarios.

## 2 Related Work

### 2.1 Video Fact-checking Datasets

Existing video datasets for fact-checking generally focus on truthfulness labels. For instance, the Checked dataset (Yang, Zhou, and Zafarani 2021) includes only truthfulness labels, while the FakeSV dataset (Qi et al. 2023a) offers a more comprehensive approach by incorporating social context, multimodal information, and debunking videos. The VMH dataset (Sung, Boyd-Graber, and Hassan 2023) provides some explanations but is primarily limited to misleading errors. The Mocheg dataset (Yao et al. 2023) pushes the boundaries by emphasizing multimodal fact-checking and text/image evidence collection. However, none of these datasets provide comprehensive explanations or supported evidence for general video fact-checking.

Our TRUE dataset addresses these gaps by integrating both human and LLM versions of rationales and evidence, sourced from credible fact-checking websites, targeting the video-post related claim, thereby enhancing interpretability, credibility, and comprehensiveness for video fact-checking evaluation. (see Table 1 for a detailed comparison).

### 2.2 Video Fact-checking Methods

Cross-modality learning improves fact-checking accuracy by integrating text, images, and videos. Models like SVFEND (Qi et al. 2023a) and BMR (Ying et al. 2023b) utilize feature fusion, while NEED (Qi et al. 2023b) leverages attention mechanisms. Traditional methods often rely on pre-trained language models like BERT and BART (Yao et al. 2023) for basic explanation generation. However, recent advancements in LLMs have significantly enhanced explanation generation and inference. For instance, QAcheck (Pan et al. 2023) and DAFND (Liu et al. 2024b) applies LLMs and exterior searching for multi-hop fact-checking, and HiSS (Zhang and Gao 2023) uses LLMs for claim decomposition and verification.

Despite these advancements, traditional methods often lack logical reasoning, failing to establish reliability, while LLM-based techniques face challenges with multimodal

content and exhibit unstable performance. Our 3MFact framework addresses these issues by integrating LMMs and LLMs for multi-modal analysis, incorporating credible online retrieval and multi-role analysis to enhance the robustness and effectiveness of video fact-checking.

## 3 Our TRUE Dataset

### 3.1 Dataset Construction

Data Collection. The dataset was sourced from Snopes $ ^{1} $, focusing on fact-checking articles containing videos, while excluding those with ambiguous labels like “unproven”. We extracted the relevant video information for the videos in the articles from platforms like YouTube and TikTok. Following specific guidelines, we manually selected the claim-sourced video posts (target videos). Transcripts for these target videos were generated using the Deepgram API $ ^{2} $. The raw dataset contains Snopes article texts, claim-related information, associated videos and video-related information, including dates, headlines, platforms, and transcripts (See the complete dataset fields on our Github).

Data Annotation. Our dataset takes a pioneering approach to enhancing the explainability and credibility of video fact-checking by being the first to specifically annotate rationales and corresponding evidence $ ^{3} $. We introduce two types of rationales: Expert-Crafted Rationale (ECR, also referred to as Original rationales) and LLM-Summary Rationale (LSR, also referred to as Summary rationales). ECR are extracted directly from Snopes articles by LLMs, preserving the original reasoning presented in the articles. Specifically, we extract both the main rationales for direct rating justification and additional supporting rationales. In contrast, LSR are generated by LLMs through synthesization of the article. Specifically, we transform the article into both concise yet comprehensive summaries and detailed reasons by decomposing the verification process and forming structured and traceable reasoning chains. We also collect Evidence that supports these rationales from the articles. Drawing from the Mocheq dataset's methodology (Yao et al. 2023), we extract textual evidence and external links from the tags in the source HTML of Snopes articles. Then, we employ LLMs to interpret the evidence and link it to the rationales. These annotations ensure that each claim is supported by well-defined rationales and evidence. The resulting dataset, as shown in Figure 1, showcases our detailed annotations.

Quality Assessment. To evaluate our dataset quality, we randomly selected 27 representative samples across different time periods and sub-labels. Each sample underwent evaluation by three independent annotators from a pool of seven experts, following a systematic framework with standardized scoring criteria across three critical dimensions (Originality, Accuracy, and Comprehensiveness), as detailed in Table 2. The evaluation results show that LSR achieves significantly higher accuracy (92.3%) and comprehensiveness (98.4%) compared to ECR (70.9% and 33.7% respectively). This performance gap stems from their different construction approaches: ECR captures the natural progression of human reasoning by extracting representative segments where reasoning points are progressively developed, while LSR excels in providing structured, comprehensive summaries through systematic synthesis of the entire content.

[None None was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Summary: This table reports the percentages for Originality, Accuracy, and Comprehensiveness for two methods or datasets (ECR and LSR). ECR achieves 100% Originality, 70.90% Accuracy, and 33.70% Comprehensiveness; LSR has a dash (indicating not applicable or not reported) for Originality, 92.30% Accuracy, and 98.40% Comprehensiveness.

Table LaTeX:

```latex
\begin{tabular}{llll}
\hline
 & Originality \$ \textasciicircum{}\{1\} \$ & Accuracy \$ \textasciicircum{}\{2\} \$ & Comprehensiveness \$ \textasciicircum{}\{3\} \$ \\
\hline
ECR & 100\% & 70.90\% & 33.70\% \\
LSR & - & 92.30\% & 98.40\% \\
\hline
\end{tabular}
```

$ ^{1} $ Originality: Unchanged from the original texts in the article.

$ ^{2} $ Accuracy: Consistent with the rationale’s definition and semantically accurate relative to the article content.

$ ^{3} $ Comprehensiveness: Covering all aspects of the reasons in the article content.

Table 2: Dataset Assessment Results

Benefits of Dual Rationales. This dual annotation approach—comprising both ECR and LSR—leverages the strengths of each (reliability vs. systematicity) and provides diverse perspectives for evaluating explainable fact-checking results. It also facilitates comparisons between human and AI-generated fact-checking, making the dataset valuable for a wide range of research and applications.

[Figure 2 was here. The original paper contained a figure at this position. Brief visual description: Figure 2 consists of two donut charts illustrating dataset statistics. Panel (a) displays the label composition, dividing entries into broad 'False' and 'True' categories, which are further subdivided into specific labels such as 'Miscaptioned', 'Mixture', 'Fake', 'MF', 'CA', 'MT', and 'True'. Panel (b) shows the source distribution, categorizing data sources into 'Social Media (SM)' and 'Official Media (OM)', with specific platforms like Instagram, TikTok, Facebook, X, YouTube, and news outlets like NYT, CNBC, and Fox News listed around the perimeter.]
Caption: Figure 2: (a) Label Composition: False labels include False (1062), Miscaptioned (268), Mixture (305), Fake (52), and Mostly False (MF, 126). True labels include True (798), Mostly True (MT, 100), and Correct Attribution (CA, 199). (b) Social Media (SM): Platforms include Instagram (IG), TikTok (TT), Facebook (FB), X (formerly Twitter), and YouTube (YT). Official Media (OM): Sources include the New York Times (NYT), CNBC, and Fox News (FN).
Key visible elements:
- Panel (a) Donut Chart: Visualizes label composition
- Panel (b) Donut Chart: Visualizes source distribution
- False Category: Major segment in Panel (a) containing false labels
- True Category: Major segment in Panel (a) containing true labels
- SM Category: Major segment in Panel (b) representing Social Media sources
- OM Category: Major segment in Panel (b) representing Official Media sources
- Sub-labels: Specific classes or sources (e.g., Miscaptioned, Fake, YT, FB)

### 3.2 Dataset Statistics

Claims in the TRUE dataset are from 2016 to 2024, with associated videos under 5 minutes in length. To preserve data authenticity, we have avoided any balancing operations, keeping the original distribution of true and false cases from controversial events. As shown in Figure 2(a), our dataset includes 1,097 true videos and 1,828 false videos, encompassing various types of misinformation. While some sub-labels appear less frequently, this distribution reflects the natural occurrence in real-world fact-checking sources (Snopes) and maintains ecological validity. The claim sources, as illustrated in Figure 2(b), include diverse official and social media platforms, enhancing the dataset's generalizability.

[Figure 3 was here. The original paper contained a figure at this position. Brief visual description: Figure 3 presents a comparative analysis of dataset statistics between true and false video claims across four metrics: claim length, claim entropy, scene transition counts, and the average count of fact-checking materials. Panels (a) and (b) utilize box plots to demonstrate that false claims are significantly shorter and have lower entropy than true claims. Panel (c) uses a bar chart to show that true claims involve significantly more scene transitions. Panel (d) displays grouped bar charts indicating that false claims generally require higher counts of Evidence (E), Expert-Crafted Rationale (ECR), and LLM-Summary-Rationale (LSR) for accurate judgment.]
Caption: Figure 3: (a) Average Claim Length: false claims are shorter in length. (b) Average Claim Entropy: false claims have lower information richness. (c) Scene Transition Counts: Videos associated with false claims tend to have more uniform scenes. (d) Average Count of Fact-Checking Materials (E: Evidence, ECR: the Expert-Crafted Rationale, LSR: LLM-Summary-Rationale.) per Claim: False claims require more evidence and rationales for accurate judgment.
Key visible elements:
- Panel (a): Box plot comparing Claim Length for True vs False claims
- Panel (b): Box plot comparing Claim Entropy for True vs False claims
- Panel (c): Bar chart comparing Scene Transition counts for True vs False claims
- Panel (d): Grouped bar chart comparing Average counts of E, ECR, and LSR for True vs False claims
- Statistical Annotations: Indicates significance levels (e.g., t-test_ind p <= 1e-5)
- Color Legend: Distinguishes 'True' (blue) from 'False' (orange)

### 3.3 Characteristics and Research Insights

Low Information Richness in Claims. To highlight the importance of incorporating new modalities and claim-related information, we conducted a preliminary analysis of the fact-checking dataset's conventional modalities—claims. Claims are typically considered key in fact-checking (Shu et al. 2017). We used entropy (Shannon 1948) to measure the information richness in claims, with lower entropy indicating higher redundancy. As shown in Figure 3(a) and Figure 3(b), false claims typically exhibit lower information richness and shorter lengths, encouraging the search for additional context or evidence for more accurate judgments.

Incomplete Video Context. With the rise of short video platforms, individual users can freely manipulate and upload videos, often resulting in incomplete footage, such as the removal of hypothetical earlier scenes. Previous work has highlighted the significant impact of editing traces in video-based fact-checking (Bu et al. 2023). On this dataset, we use PySceneDetect to identify scene transitions. As shown in Figure 3(c), false videos tend to have fewer scene transitions, suggesting a partial lack of temporal context.

These findings suggest that misinformation often lacks complete information, undermining the reliability of human and machine judgments. This underscores the importance of external information and emphasizes the role of rationales and evidence in clarifying misinformation for online consumers. We also count the average number of rationales and evidence in our dataset. As shown in Figure 3(d), both human and LLM-generated language require more information to clarify false claims compared to true ones.

## 4 Framework

We propose a multi-role collaboration framework, 3MFact, which systematically transfers data between roles to verify claims related to video posts. Inspired by the success of the Question-guided Multi-hop Structure in a text fact-checking system (Pan et al. 2023), our framework also incorporates a Question-guided process. Initially, the Video Descriptor converts the video into a textual description, which, along with the input claim and video information, is passed to the Claim Verifier. The Claim Verifier assesses the sufficiency of the existing available information. If sufficient, the data proceeds to the Reasoner to generate the final truthfulness, reasons, and evidence. If not, the Claim Verifier redirects the data to the Question Manager, initiating another cycle of inquiry and evidence gathering until adequate information is accumulated or a maximum number of cycles is reached.

### 4.1 Problem Definition

The problem of video fact-checking involves verifying a video-related claim c. The model takes as input the claim c, the video content v, and the video background information $ \mathcal{B} $ (such as the video title, release time, etc.). The primary output is a veracity rating y. Additionally, explainable fact-checking generates rationale r supported by evidence $ \mathcal{E}_r $.

### 4.2 Video Descriptor

The Video Descriptor converts video content v into a textual description t for analysis by LLM and LMM following the following steps: 1) VideoLMM generates a video-based textual description $ t_{video} $ from v. 2) ImageLMM produces a set of image descriptions $ \mathcal{T}_{image} = \{t_1, t_2, \ldots, t_n\} $ for the keyframes $ \mathcal{F} = \{f_1, f_2, \ldots, f_n\} $ of the video. Each $ t_i $ represents the textual description of keyframe $ f_i $. The keyframes are extracted by the Katna method. 3) The overall video content is synthesized by llm to produce the final textual description $ t $:

$$ \boldsymbol{t}=\mathrm{l l m}(\boldsymbol{t}_{v i d e o},\mathcal{T}_{i m a g e}). $$

This process ensures that the textual descriptions $ t $ are detailed and accurate, taking into account both visual details and the overall semantics of the video.

### 4.3 Claim Verifier

The Claim Verifier assesses the sufficiency of existing available information for verifying the claim c. This information includes the claim c, the textual video description t, the video background information B, and question-answer pairs with evidence P. This module helps the system establish a reliable judgment with high certainty, avoiding unnecessary reasoning. We use a Chain-of-Thought (CoT) strategy prompt to systematically guide the LLM in inference.

As the module’s output, $ \alpha \in \{0,1\} $ represents the judgment of information sufficiency, $ \delta \in [0,1] $ indicates the confidence level, and $ r_{cv} $ provides reasoning for the judgment. The process continues with the Question Generator in two cases: 1) when $ \alpha = 0 $, indicating insufficient information, or 2) when $ \alpha = 1 $ but $ \delta $ is below a certain threshold (default 0.93 for higher accuracy). Only when $ \alpha = 1 $ and $ \delta $ exceeds the threshold does the process proceed to the Reasoner.

[Figure 4 was here. The original paper contained a figure at this position. Brief visual description: The figure displays the architecture of the 3MFact framework, a multi-stage pipeline for video fact-checking. It features five main components: Video Descriptor, Claim Verifier, Question Manager, Information Retriever, and Reasoner. The process begins with input data (c, B, v) being processed by the Video Descriptor to generate a summary 't', which is then evaluated by the Claim Verifier. Based on the verifier's output (alpha), the system either routes the task directly to the Reasoner or to the Question Manager. The Question Manager interacts with the Information Retriever to fetch evidence and generates answers using either video analysis or retrieved text evidence.]
Caption: Figure 4: Overview of the proposed 3MFact framework, comprising five components: Video Descriptor (video-to-text conversion), Claim Verifier (assesses evidence sufficiency), Question Manager (generates questions and retrieves answers), Information Retriever (searches for evidence), and Reasoner (synthesizes judgment with rationale and evidence).
Key visible elements:
- Video Descriptor: Processes raw video input to create a textual summary
- Claim Verifier: Evaluates the claim against the video summary to determine sufficiency
- Reasoner: Synthesizes final judgment and rationale when evidence is sufficient
- Question Manager: Orchestrates question generation and answer synthesis when further investigation is needed
- Information Retriever: Searches for external evidence using tools like Google Search
- LLM Summary: Generates a summary from LMM analyses within the Video Descriptor block
- Answer Generation w/ Video LMM: Generates an answer based on visual analysis if deemed sufficient
- Answer w/ Evidences: Generates an answer based on retrieved textual evidence

### 4.4 Question Manager

The Question Manager formulates questions $ q $ when existing information is insufficient to verify the claim $ c $, subsequently deriving answers $ a $ and evidences $ \mathcal{E}_q $ via video content $ v $ or online retrieval, producing new QA&Evidence pairs $ p_N $ for further claim verification. Once a question $ q $ is generated, the Question Manager decides how to proceed:

• If $q$ pertains to the video content, the Question Manager forwards the question $q$ and video $v$ to VideoLMM. VideoLMM processes $v$ to generate both the answer and direct evidence from the video, resulting in a new QA&Evidence pair $p_N = (q, \text{videolmm}(v)).$

• If $q$ requires online retrieval, the Information Retriever module searches for relevant evidence based on the question $q$ and selects evidences $\mathcal{E}_q$. The Question Manager then uses $\mathcal{E}_q$ to generate an answer $a$, resulting in a new QA&Evidence pair $p_N = (q, a, \mathcal{E}_q)$.

The Question Manager validates the usefulness of the generated QA&Evidence pair $ p_N $ by outputting $ \beta \in \{0, 1\} $, enhancing the framework's accuracy and efficiency. If $ \beta = 0 $, the new $ p_N $ is not useful, so the process reiterates with a newly generated question $ q $. If $ \beta = 1 $, $ p_N $ is considered useful and passed to the Claim Verifier for further processing.

### 4.5 Information Retriever

The Information Retriever module extracts key search items from the raw question $ q $, facilitating the retrieval of diverse and credible evidence needed for a well-supported answer. Specifically, $ q $ is decomposed into key items $ \mathcal{K} = \{k_1, k_2, \ldots, k_m\} $, where $ m $ defaults to 2 to balance workload and result quality. The module then conducts parallel online searches, retrieving up to 10 evidence links per item and subsequently evaluating the results based on website quality, recency, and relevance, scoring each item according to the following criteria:

• Website Quality Score: $ s_{wq} $ evaluates the reliability and quality of the website content.

• Newness Score: $ s_{new} $ assesses the recency of the evidence, favoring more recent information to the claim.

• Relevance Score: $ s_{rlv} $ measures how closely the content matches the search query, focusing on relevant sentences within 250 tokens of the identified key phrases extracted from the Google search snippet.

The overall score s for each piece of raw evidence is:

$$ s=w_{1}\cdot s_{\mathrm{w q}}+w_{2}\cdot s_{\mathrm{n e w}}+w_{3}\cdot s_{\mathrm{r l v}}, $$

where $ w_{1}=0.25 $, $ w_{2}=0.25 $, and $ w_{3}=0.5 $. We prioritize relevance by assigning a higher weight to $ w_{3} $ to ensure the retrieved content closely aligns with the query.

After scoring, the Information Retriever selects the top 3 pieces of evidence $ \mathcal{E}_q $ to balance precision and coverage within LLMs' context length limits. Before passing them to the Question Manager, the relevance and usefulness of $ \mathcal{E}_q $ are additionally validated by assessing whether the evidence can adequately address the query. The validation output is $ \chi \in \{0,1\} $, where $ \chi = 1 $ means the evidence is valid and can be passed on. If $ \chi = 0 $, the evidence is deemed invalid, triggering a new retrieval cycle until a valid $ \mathcal{E}_q $ appears.

### 4.6 Reasoner

The Reasoner serves as the final decision-making module, tasked with determining the truthfulness and providing explanations based on the existing available information. The information encompasses the claim c, the textual video description t, the video background information B, and the set of question-answer-evidence pairs $ \mathcal{P} = \{p_1, p_2 \ldots p_N\} $, where N is the number of effective QA&Evidence pairs. The Reasoner is activated after the Claim Verifier has validated the sufficiency of information or when the system reaches the maximum allowable iterations, ensuring a definitive and informed judgment is rendered.

To guide the LLM in this crucial evaluation, we employ a meticulously designed prompt based on CoT strategy to enhance the reasoning capabilities, enabling it to integrate relevant information and obtain a well-substantiated decision with evidence cited for each rationale. The output of this module includes the binary truthfulness label $ y \in \{0,1\} $, where $ y = 1 $ indicates that the claim is true, and $ y = 0 $ indicates that it is false. Additionally, $ r $ provides the rationale

supporting the decision, and $ \mathcal{E}_{r} $ comprises the evidence that substantiates each rationale.

## 5 Experiments

### 5.1 Experiment Setup

Datasets and Evaluation Metrics. Our experiments are conducted on the TRUE dataset. Traditional datasets either focus solely on text claims or provide incomplete reasons, making them unsuitable for our study. For experimental evaluation, we randomly selected 433 claims (15% of the complete dataset) as our test set, ensuring temporal coverage and maintaining the original True/False ratio for unbiased representation and computational efficiency. The previous quality assessment results indicate that LLM-Summary Rationale(LSR) is more accurate and comprehensive, making it the comparison rationale used in the main experiments.

For evaluating veracity accuracy, we use standard metrics: Accuracy (Acc.), Recall, Precision (Prec.), and F1-Score. For explanation evaluation, in addition to traditional metrics like BLEU (B.) and ROUGE (RG.), following (Kim et al. 2024), we refer to G-Eval (Liu et al. 2023)(G-E.) with GPT-4o-mini (OpenAI et al. 2024) to comprehensively evaluate the quality of the generated explanations on new designed metrics. Specifically, we follow G-Eval's methodology where LLM evaluates outputs through carefully designed prompts to generate a 5-point score for each metric, taking claims, fact-checking results, and ground truth as inputs. The complete set of evaluation metrics can be found in Table 4 $ ^{4} $.

Implementation Details. We utilize the GPT-4o-mini as the LLM and MiniCPM-V 2.6 (Yao et al. 2024) as both the VideoLMM and ImageLMM. During framework development, we optimized prompts and hyperparameters through iterative experiments on modules and the overall framework, accounting for cascading effects. The Video Descriptor module takes 7 keyframes per video as input, to balance the need to capture essential content without overloading the model. To balance accuracy and efficiency, the Claim Verifier, Question Manager, and Information Retriever each limit their iterations to three rounds.

### 5.2 Model Comparison Experiments

The Model Comparison Experiments evaluate models on detection accuracy (Table 3) and explanation quality (Table 4). We compared 3MFact with traditional methods (e.g., SV-FEND) and standalone VideoLMMs (VideoLLaVa (Lin et al. 2024) and MiniCPM-V 2.6). SV-FEND performed poorly in accuracy and lacked explainability, highlighting the challenges of the TRUE dataset and was excluded from Table 4. MiniCPM-V 2.6 outperformed VideoLLaVa in accuracy, but both suffered from imbalanced recall, precision,

[Table 3 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 3: Comparison of Models for 2-Class Classification. Trad.: traditional methods; LLM.: LLM-based approaches. 3MFact $ _{(C+V)} $ uses CogVLM (Wang et al. 2023) as ImageLMM and VideoLLaVa as VideoLMM, and 3MFact $ _{(M+M)} $ uses MiniCPM-V 2.6 as both ImageLMM and VideoLMM.

Summary: This table reports performance metrics (Accuracy, Recall, Precision, F1) for five models on a 2-class classification task. Traditional methods (SV-FEND) and LLM-based approaches (VideoLLaVa, MiniCPM-V 2.6, two variants of 3MFact) are compared. The 3MFact model variant using MiniCPM-V for both image and video (3MFact_(M+M)) achieves the highest F1 score.

Table LaTeX:

```latex
\begin{tabular}{llllll}
\hline
Approach & Model & Acc. & Recall & Prec. & F1 \\
\hline
Trad. & SV-FEND & 62.80\% & 50.00\% & 31.40\% & 38.56\% \\
LLM. & VideoLLaVa & 44.11\% & 95.40\% & 41.50\% & 57.84\% \\
LLM. & MiniCPM-V 2.6 & 62.73\% & 20.81\% & 60.00\% & 30.90\% \\
LLM. & 3MFact \$ \_\{(C+V)\} \$ & 74.83\% & 72.41\% & 67.38\% & 69.81\% \\
LLM. & 3MFact \$ \_\{(M+M)\} \$ & 79.63\% & 87.23\% & 71.93\% & 73.85\% \\
\hline
\end{tabular}
```

[Table 4 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 4: Evaluation of Models on Explanation Metrics. R-O: Reasons/Evidence Omission, LC: Logical Consistency, SoE: Strength of Evidence, COM: Comprehensiveness, CUR: Currency, CON: Conciseness, FAI: Faithfulness, FH: Fact Hallucination.

Summary: Table 4 reports evaluation scores for four models (VideoLLaVa, MiniCPM-V, 3MFact_(C+V), 3MFact_(M+M)) on explanation metrics: B., RG., R-O, a combined metric (LC, SoE, COM, CUR, CON, FAI, FH), and an additional unnamed sixth column. 3MFact variants generally outperform the other two models across all metrics.

Table LaTeX:

```latex
\begin{tabular}{llllll}
\hline
Model & Trad. & Content Credibility Analysis (G-E.) & Content Credibility Analysis (G-E.) & Content Credibility Analysis (G-E.) & \\
\hline
Model & B. & RG. & R-O & LC SoE COM CUR CON FAI FH & \\
VideoLLaVa & 0.00 & 0.16 & 1.18 & 3.20 & 1.82 \\
MiniCPM-V & 0.00 & 0.16 & 1.43 & 3.69 & 2.08 \\
3MFact \$ \_\{(C+V)\} \$ & 0.03 & 0.26 & 2.24 & 4.48 & 3.98 \\
3MFact \$ \_\{(M+M)\} \$ & 0.04 & 0.25 & 2.37 & 4.47 & 3.96 \\
\hline
\end{tabular}
```

### 5.3 Ablation Study

The Ablation Study evaluates the impact of individual components within the 3MFact framework, identifying the key elements that significantly enhance model performance and explanation quality. The experimental results for detection accuracy (Table 5) and explanation quality (Table 6) demonstrate the contributions of these components.

Based on the ablation study results, we can broadly rank the impact of different modules or components on overall accuracy as follows: Information Retriever > Validation (including Validation of P and Validation of $ E_q $) > Question Answering with VideoLMM > Video Descriptor. This ranking reflects the relative importance of these modules in enhancing the framework's performance. Notably, due to the limitations in video descriptive capabilities, the Video Descriptor module shows limited impact on accuracy, despite advances in VideoLMMs(e.g., from VideoLLaVa to MiniCPM-V 2.6). As VideoLMM capabilities continue to improve, the Video Descriptor module is expected to bring more benefits to future video fact-checking field.

Regarding explainability, the 3MFact framework consistently provides strong and effective explanations across all configurations. Removing the VideoLMM component

from the Question Manager unexpectedly improves some evidence-related metrics. This could because excluding VideoLMM forced the framework to rely solely on the Information Retriever to gather evidences for answering questions, which places more focus on evidence-based explanations. Conversely, eliminating the Information Retriever leads to less detailed explanations and higher conciseness scores. The results reflect the optimal performance of the complete 3MFact, where all modules work together to achieve a balanced and comprehensive explanation quality.

[Table 5 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 5: Ablation Study on Different Modules. IR: Information Retriever module, QA $ _{LMM} $: Question Answering with VideoLMM, VD: Video Descriptor module, 2Val: Validation of P and Validation of $ E_{q} $

Summary: This ablation table reports the impact of removing individual modules (IR, QA_LMM, VD, 2Val) from the full 3MFact model on four metrics: Accuracy, Recall, Precision, and F1. The full model achieves the best performance, while removing the IR module causes the largest degradation across all metrics.

Table LaTeX:

```latex
\begin{tabular}{lllll}
\hline
Model & Acc. & Recall & Prec. & F1 \\
\hline
3MFact \$ \_\{(M+M)\} \$ & 79.63\% & 87.23\% & 71.93\% & 78.85\% \\
w/o IR & 69.98\% & 60.12\% & 63.03\% & 61.54\% \\
w/o QA \$ \_\{LMM\} \$ & 76.18\% & 81.58\% & 66.31\% & 73.16\% \\
w/o VD & 79.42\% & 85.71\% & 68.67\% & 76.25\% \\
w/o 2Val & 72.09\% & 74.85\% & 64.00\% & 69.00\% \\
\hline
\end{tabular}
```

### 5.4 The Performance on Different Rationales

In this section we compare the 3MFact framework on both Expert-Crafted Rationale (ECR) and LLM-Summary-Rationale (LSR). The results of the 3MFact $ _{(M+M)} $ framework are shown in Table 7. The results indicate that the 3MFact framework's performance on the original rationale is also fairly good, demonstrating its alignment with human explanations. The lower scores for the ECR compared to LSR may be due to some inaccuracies in the currently collected rationales (as shown in Table 2). Nevertheless, ECR can largely avoid fact-hallucination flaws while retaining the nuances of human reasoning. We believe that this part of the dataset can be further refined in the future, collaborating with LSR to provide a more reliable and comprehensive set of ground-truth rationales.

### 5.5 Case study

In addition to the quantitative analysis, we conducted a qualitative analysis with selected successful and unsuccessful cases to visually showcase the capabilities of our 3MFact framework. Figure 5(a) shows an example of successful detection. In this case, our framework successfully predicted the falsity of a claim by retrieving related news articles from the web that effectively refuted the claim. This demonstrates the framework's ability to utilize external evidence to challenge and verify claims. Conversely, as shown in Figure 5(b), a failed case is illustrated where an accurate claim was incorrectly classified by our framework. The error arose from inaccurate analysis of video details, leading to a miscued caption and a subsequent misjudgment. This highlights a current limitation in our framework's video detailed content analysis and the need for further improvements.

[Table 6 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 6: Ablation Study Evaluation on Explanation Metrics.

Summary: This table presents an ablation study comparing the full 3MFact model (with M+M variant) against four ablated variants (w/o IR, w/o QA_{LMM}, w/o VD, w/o 2Val) across five explanation metrics: B, RG, R-O, LC, and SoE. The values show that the full model achieves the highest R-O score and ties for the highest B score, while the w/o IR variant generally underperforms, especially on R-O and SoE.

Table LaTeX:

```latex
\begin{tabular}{llllll}
\hline
Model & Trad. & Content Credibility Analysis(G-E.) & Content Credibility Analysis(G-E.) & Content Credibility Analysis(G-E.) & Content Credibility Analysis(G-E.) \\
\hline
Model & B & RG & R-O & LC & SoE \\
3MFact \$ \_\{(M+M)\} \$ & 0.04 & 0.25 & 2.37 & 4.47 & 3.96 \\
w/o IR & 0.03 & 0.24 & 1.70 & 4.22 & 2.72 \\
w/o QA \$ \_\{LMM\} \$ & 0.03 & 0.25 & 2.30 & 4.46 & 4.24 \\
w/o VD & 0.03 & 0.26 & 2.25 & 4.41 & 4.01 \\
w/o 2Val & 0.04 & 0.25 & 2.17 & 4.41 & 3.92 \\
\hline
\end{tabular}
```

[Table 7 was here. The original paper contained a table at this position. The normalized LaTeX table is preserved below.]

Caption: Table 7: Comparison of Results on Different GT-R (Ground Truth Rationale) type. GT-R type: Ground Truth Rationale Type. ECR: the Expert-Crafted Rationale, LSR: the specific LLM-Summary-Rationale

Summary: This table compares the performance of three Ground Truth Rationale (GT-R) types—ECR+LSR, ECR, and LSR—on three evaluation metrics: BLEU-4, ROUGE-L, and R-O(G-E.). The values show that LSR achieves the highest scores across all metrics, while ECR has the lowest.

Table LaTeX:

```latex
\begin{tabular}{llll}
\hline
GT-R type & BLEU-4 & ROUGE-L & R-O(G-E.) \\
\hline
ECR+LSR & 0.024 & 0.242 & 2.157 \\
ECR & 0.021 & 0.194 & 2.166 \\
LSR & 0.037 & 0.252 & 2.370 \\
\hline
\end{tabular}
```

[Figure 5 was here. The original paper contained a figure at this position. Brief visual description: Figure 5 presents qualitative case studies of the 3MFact framework's claim detection performance. It displays two examples: the first demonstrates a successful detection where a false claim about Cristiano Ronaldo making his hotel available was correctly identified as 'False' with a supporting rationale and evidence snippet; the second example shows a claim labeled 'TRUE' regarding Putin's entourage and a UFC fighter, accompanied by video frames highlighting specific individuals.]
Caption: Figure 5: Examples of successful and failed claim detection by the 3MFact framework.
Key visible elements:
- Label: FALSE box: Indicates the ground truth label for the first claim
- Claim text (Ronaldo): The specific claim being evaluated in the first example
- Video frames (Ronaldo case): Visual context including rubble, Ronaldo's face, and a hotel room
- Framework Result box: Displays the model's output, including Rating (False), Rationale, and Evidence
- Label: TRUE box: Indicates the ground truth label for the second claim
- Claim text (Putin): The specific claim being evaluated in the second example
- Video frames (Putin case): Visual context showing Vladimir Putin and associates, with red circles highlighting entities

## 6 Conclusion

We explored the first dataset TRUE for explainable video fact-checking, which emphasizes the essential of resummarized rationale. It includes abundant multimodal information, providing an exploratory way for supporting explainable fact-checking research. The in-depth statistical analysis highlighted the necessity and practicality of TRUE. Based on this, we proposed an innovative multi-role structure 3MFact, tackling unseen misinformation among multimodals via diverse sources. We also established novel metrics to evaluate both accuracy and reasoning. Extensive experiments demonstrated the effectiveness of 3MFact in both misinformation detection and explanation. Nevertheless, significant room for improvement remains. LMMs, in particular, face challenges in fact-checking subtasks such as answering fact-related questions and generating video descriptions. Furthermore, the human-crafted rationales in our dataset require further refinement to enhance their completeness and accuracy.