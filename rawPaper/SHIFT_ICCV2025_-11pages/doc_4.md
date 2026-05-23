where  $ V_l $ is the feature vector of the  $ l $-th layer,  $ \alpha $ is an adjustment parameter, and  $ f(\cdot) $ is the transformer block. With the greedy decoding, the prediction of the next token  $ x_t^* $ is

 $$ x_{t}^{*}=\underset{x_{t}\in X}{\arg\max}softmax(\phi(F(V_{l^{*}}^{*}_{\mathrm{m u t a t i o n}})))_{x_{t}}, $$ 

where  $ F(\cdot) $ is the collection of the transformer blocks after the  $ L_{mutation}^{*} $-th layer.

### 4. Experiment

#### 4.1. Setup

##### 4.1.1. Models & Baselines

SHIFT is evaluated on three MLLMs, including LLaVA-1.5 [34], mPLUG-Owl2 [47], and InstructBlip [12]. For more comprehensive comparisons, we choose three decoding methods, including greedy, beam search, and nucleus sampling. Greedy decoding selects tokens with the highest probability in logits step by step. Beam search maintains a set of beams and selects the best token from them. Nucleus sampling focuses on the most significant probability mass at each time step by keeping a limited subset of the vocabulary. Besides the basic decoding methods, we also consider OPERA [19], which is implemented on beam search, and the contrastive decoding methods VCD [25] and ICD [46], which are improvements on nucleus sampling.

##### 4.1.2. Benchmark & Evaluation Metrics

CHAIR [38] The Caption Hallucination Assessment with Image Relevance (CHAIR) assesses hallucinations on the image captioning task. CHAIR works by generating ground-truth object labels for each image, and any object mentioned absent from the label set is deemed a hallucinated object. It comprises two metrics: sentence-level  $ (CHAIR_{S}) $ and image-level  $ (CHAIR_{I}) $, calculated as:

 $$ \mathrm{CHAIR}_{S}=\frac{\left|\left\{\mathrm{hallucinated\ objects}\right\}\right|}{\mathrm{all\ mentioned\ objects}}, $$ 

 $$ \mathrm{CHAIR}_{I}=\frac{\left|\left\{\text{captions with hallucinated objects}\right\}\right|}{\text{all captions}}. $$ 

We conduct CHAIR evaluation on the MSCOCO dataset [31], with the prompt “Please describe this image in detail”. Following the setup in [19], we randomly sample 500 images from the validate set for test.

POPE [29] The Polling-based Object Probing Evaluation (POPE) is a widely used benchmark for identifying object-level hallucinations. It utilizes a question-answering format such as "Is there a {} in the image", prompting MLLMs to assess whether a specified object is present in the image. We adopt two subsets for evaluation, including MSCOCO and GQA [20]. There encompasses three sampling settings for each subset: random, popular, and adversarial. MMHal-Bench [32] MMHal-Bench is another VQA-based evaluation on object hallucinations. Different from POPE, its questions contain logical considerations, which is more challenging for MLLMs. This benchmark includes 8 types of high-difficulty questions about object attributes: adversarial objects, comparisons, counting, spatial relations, environment, holistic descriptions, and others.



GPT-4v Assisted Evaluation CHAIR can only evaluate hallucinations based on the presence of objects in descriptions, struggling to assess physical attributes, positions, and other aspects. To provide a more comprehensive evaluation of hallucinations, following [19, 48], we use a GPT-4v-based assessment strategy. Specifically, we randomly sample 500 images from the MSCOCO dataset, and ask MLLMs to describe them. For each sample, the image together with responses from the vanilla decoding method and SHIFT are fed into GPT-4v, which is prompted to give judgements from 0-10 to assess the responses' qualities.

#### 4.2. Quantitive Results

##### 4.2.1. Results on CHAIR

We test SHIFT's performance when generating long sentences on the CHAIR benchmark, and the results are present in Table 2. As a decoding-independent method, SHIFT differs from the decoding-based baselines that primarily target at improving a specific decoding method. Therefore, we evaluate it on all of the three decoding methods, and the results show that SHIFT always outperforms baselines despite the decoding strategies. For example, when using the greedy decoding, SHIFT outperforms vanilla by up to 12.4% and 3.3% on sentence and image-level metrics, respectively. With the beam search decoding, SHIFT achieves CHAIR $ _{S} $ and CHAIR $ _{I} $ that exceed OPERA by 11.4% and 1.8%. When combined with the nucleus decoding, SHIFT also surpasses contrastive-decoding-based methods. The results demonstrate SHIFT's effectiveness in long description generation, which stems from its direct handling of information flow. By tracking differences in information between adjacent layers, it identifies mutated information and tunes it using continuous information from preceding layers. This process ensures the model's knowledge in deep layers remains faithful to the multimodal input, which effectively mitigates the influence of hallucinated knowledge may be contained in mutated information. Notably, SHIFT mitigates hallucinations without compromising the length of the generated text. In our experiments, the average sequence lengths when combined with the three decoding strategies are 99.8, 97.3, and 100.2, compared to 101.1, 98.5, and 100.7 for the vanilla model, respectively.

##### 4.2.2. Results on POPE

The experimental results on the POPE benchmark are shown in Table 3. It can be observed that SHIFT can