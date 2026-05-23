block. This reshaping divides the region into a grid of blocks, with m being the total number of blocks. Next, we flatten each block  $ r_j $ and project it into a  $ d^I $-dimensional vector. This projection is performed using a trainable linear projection matrix E, and the resulting embedded representation of block  $ r_j $ is denoted as  $ z_j = r_j E $. To incorporate contextual information and retain positional information, we prepend a [class] token embedding at the beginning of the patch sequence. Position embeddings are also appended to the patch embeddings, indicating their relative positions within the sequence. The input representation of each visual region  $ I_i $ is expressed as:

 $$ \boldsymbol{Z}_{i}=[\boldsymbol{z}_{[class]};\boldsymbol{z}_{1},...,\boldsymbol{z}_{m}]+\boldsymbol{E}_{pos} $$ 

where  $ Z_i $ represents the input matrix of image patches, and  $ E_{pos} $ denotes the position embedding matrix. Subsequently, we feed the input matrix  $ Z_i $ into the VGG16 encoder to obtain the visual region  $ I_i $ representation  $ h_i^v = VGG16(Z_i) $. Finally, the representation of the image I is defined as:

 $$ h_{I}=\{h^{c},h_{1}^{v},...,h_{m}^{v}\} $$ 

## Modal Fusion

The text and image of multimodal meme have the problem of weak correlation, and directly fusing the text and image may result in incorrect meme understanding. A good fusion solution should extract and integrate sufficient information from multimodal sequences while preserving the independence of each modality. Therefore, we propose a novel global-local cross-modal interaction model that not only considers interactions between modalities but also emphasizes the importance of each modality itself to enhance multimodal fusion at multiple granularities. Specifically, we devise an efficient mechanism called Cross-modal Attention Promotion (CAP) that leverages symmetric cross-modal attention to explore the inherent correlations between the two input feature sequences, promoting the exchange of beneficial information across the sequences. CAP utilizes self-attention to model the temporal dependencies within each feature sequence, enabling the integration of more information. The mechanism takes sequences  $ \pmb{h}^{T} $ and  $ \pmb{h}^{I} $ as inputs and generates their mutually reinforcing information  $ \pmb{h}_{T\rightarrow I} $ and  $ \pmb{h}_{I\rightarrow T} $. The computation of  $ CAP_{T\leftrightarrow I}(\pmb{h}^{T},\pmb{h}^{I}) $ is as follows:

 $$ \begin{aligned}\boldsymbol{h}^{\prime}_{T\rightarrow I}&=MCA(LN(\boldsymbol{h}_{T}),LN(\boldsymbol{h}_{I}))+\boldsymbol{h}_{T}\\\boldsymbol{h}^{\prime\prime}_{T\rightarrow I}&=MSA(LN(\boldsymbol{h}^{\prime}_{T\rightarrow I}))+\boldsymbol{h}^{\prime}_{T\rightarrow I}\\\boldsymbol{h}_{T\rightarrow I}&=FN(LN(\boldsymbol{h}^{\prime\prime}_{T\rightarrow I}))+\boldsymbol{h}^{\prime\prime}_{T\rightarrow I}\end{aligned} $$ 

where LN denotes layer normalization and FN is the feedforward neural network.  $ MSA(\cdot) $ refers to the output of the multi-head self-attention mechanism computation.  $ MCA(\cdot,\cdot) $ represents the result of the multiple cross-attention mechanism calculation. Similarly, we can obtain  $ CAP_{I\leftrightarrow T}(h_I,h_T) $.

The traditional cross-attention interaction requires two updates during the modal interaction process to achieve modal enhancement, which is inefficient and introduces redundant features into the sequence. Based on the historical experience from large-scale pretraining, it has been observed that a single token can represent the entire sequence, further improving the efficiency of modal interaction. Motivated by this observation, we propose a global-local cross-modal interaction model with linear computational cost to enhance efficiency. The discourse-level representation of each modality replaces the standard information and interacts with local unimodal features within a global multimodal context. This means that the representation of each modality not only relies on local features but also takes into account the influence of global context. This global-local interaction model reduces the introduction of redundant features and improves modal interaction effectiveness while maintaining efficiency.



We establish the global multimodal context information denoted as  $ \boldsymbol{g}^{i} = \text{concat}(\boldsymbol{h}_{T}^{i}, \boldsymbol{h}_{I}^{i}) $ by concatenating the representations of each modality at each layer of global-local interaction, where i represents the number of layers of global local interaction. By integrating the global context information and local modal information, and learning modal consistency and specificity, we ensure effective interaction and capture relevant information from both the global and local perspectives. The entire interaction process is as follows:

 $$ \begin{aligned}\boldsymbol{h}_{T}^{(i+1)},\boldsymbol{g}_{T\rightarrow G}^{(i)}&=C A P_{T\leftrightarrow G}^{(i)}(\boldsymbol{h}_{T}^{(i)},\boldsymbol{g}^{(i)})\\\boldsymbol{h}_{I}^{(i+1)},\boldsymbol{g}_{I\rightarrow G}^{(i)}&=C A P_{I\leftrightarrow G}^{(i)}(\boldsymbol{h}_{I}^{(i)},\boldsymbol{g}^{(i)})\\\end{aligned} $$ 

By stacking multiple layers, the global multimodal context and local unimodal features can mutually reinforce and progressively refine each other. We hierarchically handle the entire learning process, with each layer capturing different features corresponding to the model's main stages. The model initially learns shallow interaction features, gradually progressing to acquire higher-order semantic features in later stages. This hierarchical learning method successfully integrates information from various modalities by ingeniously designed aggregation blocks, providing the model with a more comprehensive and enriched representation of multimodal features. Through the model's interactions, information from different modalities can be combined in a deeper and more effective manner, enabling the acquisition of more advanced feature representations in subsequent hierarchical learning. Subsequently, we aggregate the features from both unimodal and multimodal sources to facilitate subsequent task predictions.

 $$ y_{m}=softmax(\boldsymbol{W}_{m}MSA([\boldsymbol{h}_{T}^{(L)},\boldsymbol{h}_{I}^{(L)},\boldsymbol{g}^{(L)}])+\boldsymbol{b}) $$ 

where  $ y_{m} $ is the feature output distribution after multimodal fusion,  $ W_{m} $ and b are trainable parameters.

Our approach focuses not only on the interactions between modalities but also on the individual feature representations of each modality. We separately use the unimodal features obtained from text and image encoders to predict subsequent tasks. This allows to capture the unique characteristics and information within each modality, thereby improving the accuracy and effectiveness of MMU.

 $$ \begin{aligned}\boldsymbol{y}_{T}&=softmax(\boldsymbol{W}_{t}\boldsymbol{h}_{T}+\boldsymbol{b})\\\boldsymbol{y}_{I}&=softmax(\boldsymbol{W}_{i}\boldsymbol{h}_{I}+\boldsymbol{b})\end{aligned} $$ 

Given the  $ y_{M}, y_{T} $, and  $ y_{I} $, we obtain the final prediction y:

 $$ \boldsymbol{y}=\boldsymbol{y}_{M}+\boldsymbol{y}_{T}+\boldsymbol{y}_{I} $$ 

where y can be considered as a comprehensive feature set encompassing multi-granular features, including text, image, and image-text modalities.