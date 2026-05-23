#### 3.4.1 Metrics for Knowledge Retention

• Overall Accuracy (Overall Acc): The average correctness across all decisions, measuring how consistently the model maintains relevant knowledge and narrative coherence:

 $$ \mathrm{Accuracy}_{\mathrm{overall}}=\frac{1}{T}\sum_{t=1}^{T}c_{t}. $$ 

- First-Try Accuracy (First-Try Acc): The proportion of decision points at which the model selected the correct option on its first attempt. Let  $ f_t \in \{0,1\} $ be 1 if the model is correct on the first try at step t, then:

 $$ \mathrm{Accuracy}_{\mathrm{first-try}}=\frac{1}{T}\sum_{t=1}^{T}f_{t}. $$ 

• Longest Consecutive Correct Sequence (Longest Corr): The length of the longest contiguous subsequence of correct decisions:

 $$ \mathrm{Longest Corr}=\max_{1\leq i\leq j\leq T}\left(j-i+1\mid c_{k}=1\forall k\in\left[i,j\right]\right). $$ 

This reflects the model’s ability to sustain contextual consistency over extended intervals, though less critical than the above metrics.

#### 3.4.2 Metrics for Sequential Reasoning

• Accuracy by Difficulty (Easy/Hard Acc): To account for varying levels of memory and reasoning demand, we classify decisions into easy and hard categories. A decision is labeled as hard if it requires recalling information from a distant context, tracking latent state changes, or performing multi-step sequential reasoning; otherwise, it is considered easy. Let  $ \mathcal{E}_t $ and  $ \mathcal{H}_t $ denote easy and hard decision sets up to step  $ t $ (including retries), then:

 $$ \mathrm{A c c u r a c y}_{\mathrm{e a s y}}^{(t)}=\frac{1}{|\mathcal{E}_{t}|}\sum_{i\in\mathcal{E}_{t}}c_{i},\quad\mathrm{A c c u r a c y}_{\mathrm{h a r d}}^{(t)}=\frac{1}{|\mathcal{H}_{t}|}\sum_{i\in\mathcal{H}_{t}}c_{i}. $$ 

These metrics assess how well the model adapts to sequentially distributed and cognitively demanding decisions.

• Retry Count: Let  $ r_{t} $ denote the number of retries required before reaching a correct decision at step t. The total number of retries across the trajectory is:

 $$ \mathrm{Retry_{total}}=\sum_{t=1}^{T}r_{t}. $$ 

- Max Error per Choice (Max Err/Choice) and Thresholded Error Count: These metrics capture the worst-case and accumulated difficulty for the model in terms of repeated failures:

 $$  MaxError=\max_{1\leq t\leq T}r_{t},\quad ErrorCount\geq r_{thres}=\sum_{t=1}^{T}\mathbb{I}(r_{t}\geq r_{thres}), $$ 

Where  $ \mathbb{I}(\cdot) $ is the indicator function and  $ r_{\mathrm{thres}} $ is a predefined retry threshold (e.g., 9 in our experiments).

Finally, while not directly measuring memory accuracy, two auxiliary metrics provide additional perspective on the model's efficiency in handling long-horizon tasks: Runtime Cost reflects the inference efficiency of the memory system, while Token Consumption (Token Cons) indicates the model's reliance on contextual information.

Together, these metrics form a multi-faceted evaluation framework that jointly targets both the persistence of stored information and the model's ability to apply it dynamically within complex, sequentially structured environments. This ensures that memory is not only retained but also meaningfully used to navigate and reason through realistic multi-turn interactions.