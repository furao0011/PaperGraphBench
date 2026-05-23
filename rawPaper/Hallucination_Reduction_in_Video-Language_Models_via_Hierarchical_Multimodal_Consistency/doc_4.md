the parameters of the Q-Former and the text encoder, while keeping the parameters of the visual encoder and the LLM frozen.

Injecting text semantic supervision offers several potential advantages: (1) It improves the learning process of visual encoding in Q-Former, enhancing its semantic discriminability. (2) As a standardized representation, text semantic features exhibit a stronger consistency with visual features, which effectively promotes the interaction of cross-modal information.

Multi-Level Alignment. Semantic alignment at different levels allows for the capture of text and visual information across varying degrees of abstraction. In long-term video understanding tasks, lower levels align basic visual features such as the clothing colors of characters and the geometric shapes of objects with the corresponding color and shape vocabulary in the text. At higher levels, complex visual semantics, like event flow and character relationships, can be matched with more abstract concepts in the text related to story development and character interactions.

To further enhance the semantic discriminability of visual encoding, we extend the semantic discriminative loss to a multi-level framework for improved semantic alignment. The multi-level semantic loss can be expressed as:

 $$ \begin{align*}\mathcal{L}_{multi}&=\sum_{l=1}^{L}\mathcal{L}_{semantic}^{l}\\&=-\frac{1}{N}\sum_{l=1}^{L}\sum_{i=1}^{N}\log\frac{\exp\left(f_{i}^{v}\cdot f_{i}^{t}/\tau\right)}{\sum_{j=1}^{N}\exp\left(f_{i}^{v}\cdot f_{j}^{t}/\tau\right)},\end{align*} $$ 

where L represents the number of levels. In the experiment, we adopted a two-level alignment scheme of aligning the output features of Q-Former and aligning the input features of the cross-attention mechanism. Finally, we achieve semantic alignment of features across all levels of visual and textual modalities using a multi-level semantic discriminative loss. This enables our model to capture both high-level and low-level semantic relations while effectively reducing hallucinations by establishing precise correspondences between video content and generated language.

### 3.3 Two-Stage Training

The training dataset is a significant factor contributing to hallucinations in video-language models. On one hand, the lack of diversity in some datasets results in the model having an inadequate understanding of certain visual concepts, complicating the alignment between video and text modalities. On the other hand, the uneven distribution of objects in the training set causes the video-language model to favor predicting common objects or frequently co-occurring combinations of objects. Therefore, we propose a two-stage training scheme that utilizes the extended dataset to improve the optimization of the multi-level semantic discriminative loss. These two stages are the auxiliary pre-training stage and the task-specific training stage respectively.

Auxiliary Pre-Training Stage. In this stage, we utilize a larger amount of data to infuse richer semantics into the training of the video-language model. Specifically, we initially conduct pre-training on the WebVid dataset. Through this process, the model can learn the general semantics between the visual and language modalities. This serves as an auxiliary for the training of the video-language model in specific tasks. The WebVid-5K dataset contains a vast variety of video clips with corresponding textual descriptions. By exposing the model to this extensive and diverse data source, it can capture a wide range of semantic relationships that exist in the real world. This helps the model to generalize better and build a more solid foundation for subsequent task-specific training.



Task-Specific Training Stage. Once the pre-training process of the auxiliary pre-training stage is completed, we proceed with further training on other datasets to achieve semantic alignment for specific task. This two-step approach is designed to leverage the knowledge acquired during the initial pre-training phase and fine-tune the video-language model according to the requirements of the specific task. This progressive learning process from large-scale general semantics to task-specific semantics allows the video-language model to continuously refine its semantic understanding. It gradually narrows down its focus from the broad semantic space learned during pre-training to the specific semantic domain of the target task. Through this iterative process of learning and adaptation, the model can capture more accurate cross-modal semantic relationships, which in turn leads to enhanced performance in generating high-quality outputs for the specific task.

### 3.4 Training Objectives

We input the output features of the Q-Former, which contains all sequential historical information at the final time step, into the LLM for text decoding. During training, given a labeled dataset consisting of video and text pairs, our model is supervised using the standard cross-entropy loss:

 $$ \mathcal{L}_{t e x t}=-\frac{1}{S}\sum_{i=1}^{S}\log P(w_{i}|w_{<i},V), $$ 

where V represents the input video and  $ w_{i} $ is the i-th ground-truth text token. In the auxiliary pre-training stage, we only use the semantic discriminative loss to train the Q-Former and text encoder, and do not use the LLM for text decoding. However, in the task-specific training stage, we carry out the training using two loss functions simultaneously. The overall loss function can be expressed as

 $$ \mathcal{L}=\lambda_{m u l t i}\cdot\mathcal{L}_{m u l t i}+\lambda_{t e x t}\cdot\mathcal{L}_{t e x t}, $$ 

where  $ \lambda_{multi} $ and  $ \lambda_{text} $ are hyper-parameters to trade off the two parts.

## 4 Experiments

### 4.1 Dataset

Experiments are conducted on two widely used long-term video datasets: The LVU dataset [Wu and Krahenbuhl, 2021] consists of more than 30,000 video clips, each ranging from 1 to 3 minutes, sourced from approximately 3,000 movies in diverse real-world contexts. The MSVD dataset [Chen and