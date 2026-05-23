In our initial experiments, we unexpectedly discovered that a simple smoothing operation was particularly useful in improving prediction accuracy. Specifically, we tested the Prediction Correction Rate and Prediction Degradation Rate with and without smoothing on a binary CaseLaw dataset containing 3000 samples, as shown in table[5].

• Prediction Correction: When the initial prediction of the model is wrong, and it is corrected by the debate-feedback framework.

• Prediction Degradation: When the initial prediction of the model is correct, but becomes incorrect due to the framework.

We found that the Prediction Degradation Rate was particularly high without smoothing, while the Prediction Correction Rate was about the same. This means the smoothing mechanism helps models avoid relying too heavily on the influence of a certain debater.