Algorithm 1: Debate-Feedback

Input: LM,  $ E: S \to [0,1] $; n,  $ T \in N $;
 $ x \in S $;  $ t_{ne}, t_{po} : S \to S $;

Output: Final decision  $ y \in (0,1) $;
 $ O_0 \leftarrow LM(x) $;

for i  $ \leftarrow $ 1 to n do

// Debate Step

a:  $ a_{ne}, a_{po} \leftarrow t_{ne}(x), t_{po}(x) $;

e:  $ e_{ne}, e_{po} \leftarrow t_{ne}(x \oplus a_{po}), t_{po}(x \oplus a_{ne}) $;

// Verification Step

v:  $ v_{ne}, v_{po} \leftarrow \mathcal{E}(e_{ne}), \mathcal{E}(e_{po}) $;

sum = LM(a, e, v);

if Threshold(v) then

 $ O_i = $

 $ (1 - T) * O_{i-1} + T * LM(x, sum) $;

end

else

 $ O_i = LM(x) $;

end

end

 $ y \leftarrow O_n $;


<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td colspan="2">TrainingSet of Assistant model</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Training_X</td><td style='text-align: center; word-wrap: break-word;'>$ \{Case\_background + Debater&#x27;s opinion\} $</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Training_Y</td><td style='text-align: center; word-wrap: break-word;'>$ \{Ground\_truth XOR Debater&#x27;s position\} $</td></tr></table>

<div style="text-align: center;">Table 1: Dataset of assistant model.</div>


Reliability Analysis Through experiments, we observe that a simple debate model can sometimes lead to worse prediction results. This occurs because legal predictions differ from mathematical problems, as they often involve subjective tendencies. A straightforward example is when we guide multiple LLMs to debate from the perspectives of the plaintiff and defendant, it is challenging for them to reach a consensus. To address this issue, one of our solutions is to train an assistant model that learns from a large corpus of legal event annotations and assists in evaluating the reliability of different debate arguments, as shown in Table $ ^{[1]} $. Specifically, the training set for the assistant model is generated from multiple runs of the unassisted Debate-Feedback model, which we refer to as Debate-Feedback (single) in the subsequent experimental section.

Smoothing Operation To mitigate the impact of a "failed" debate where the main LLM generates incorrect answers, we apply a smoothing operation. This involves saving the results of each prediction and assigning them a certain weight. Specifically, let  $ LM(x) $ represent the predicted result of the i-th debate and T be the weighting factor. The updated result is calculated as:



 $$ O_{i}\leftarrow(1-T)*O_{i-1}+T*L M(x) $$ 

where  $ T \in [0, 1] $ represents the weight assigned to the latest prediction.

## 4 Experiment

### 4.1 Dateset and Baseline

Along with many influential LegalAI works, we also use CaseLaw as the main dataset. The CaseLaw dataset is a legal case dataset specifically used for natural language processing (NLP) and machine learning tasks in the legal field, especially in the fields of legal case retrieval and legal judgment prediction. This dataset contains a large number of court case texts that have been judged, usually including descriptions of legal facts, legal reasoning, and judgment results. In order to test the model's cross-language and cross-legal capabilities, we also used the Chinese dataset CAIL18 (Xiao et al., 2018; Zhong et al., 2018b).

We compare Debate-Feedback with both general large language models and legal domain models. GPT4o and GPT3.5-turbo are representative general large language models at present (OpenAI et al., 2024), and they have been proven to have strong text analysis and logical reasoning capabilities. LegalBert (Chalkidis et al., 2019) and Lawformer (Xiao et al., 2021) are well-known legal domain models, they're able to capture the association between legal terms and cases well. In addition, CNN (Lecun et al., 1998) is also used as a classifier for feature extraction in the baseline evaluation, with BERT (Devlin et al., 2019) serving as the text embedding layer.

Considering that the debate-feedback framework can essentially be seen as a large language model reasoning framework, we also compare it with classic reasoning methods, including Few-shot Learning, Chain of Thought(CoT) (Wei et al., 2023) and Reflexion (Shinn et al., 2023). We use gpt-4o mini as the baseline model in this part and verified them on a smaller subset on a smaller subset of the datasets (12,000 samples from CaseLaw and 3,000 samples from CAIL18).