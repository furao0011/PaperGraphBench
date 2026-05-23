### 4.2 Regular LJP tasks

Trial Prediction The input for trial prediction includes a legal text, along with the opinions of the plaintiff and defendant. The predicted labels are Plaintiff wins, Defendant wins, Settlement, and Dismissed. Since Settlement and Dismissed are explicitly stated in the legal text, this can be reduced to a binary classification task with two labels: Plaintiff wins and Defendant wins. The CaseLaw dataset was used for this task, and Table[4] provides a sample.

Article Prediction Article prediction is a multilabel classification task. The model receives a description of legal facts and the prediction content contains multiple labels of different relevant law articles. CAIL18 dataset is used in this task.

### 4.3 Evaluation Metrics

In this study, we evaluate the model performance using two key metrics: accuracy and F1-score.

 $$ \mathrm{Accuracy}=\frac{\sum_{i=1}^{N}(y_{i}=y_{true,i})}{N} $$ 

Accuracy(Acc) is the proportion of correct predictions among all predictions. It is computed as:

where N is the total number of predictions,  $ y_{i} $ is the predicted label,  $ y_{true,i} $ is the actual label, and  $ (\cdot) $ is the indicator function that equals 1 when the condition is true and 0 otherwise.

F1-score(F1) is useful for imbalanced datasets as it balances precision and recall. In multi-class classification, F1-score is computed for each class and then averaged (macro F1-score). For a single class, F1-score is given by:

 $$ F1=2\times\frac{Precision\times Recall}{Precision+Recall} $$ 

Where precision and recall are defined as:

 $$  Precision=\frac{\sum_{i=1}^{N}1(y_{i}=c\land y_{true,i}=c)}{\sum_{i=1}^{N}1(y_{i}=c)} $$ 

 $$  Recall=\frac{\sum_{i=1}^{N}1(y_{i}=c\land y_{true,i}=c)}{\sum_{i=1}^{N}1(y_{true,i}=c)} $$ 

For multi-class classification, the macro F1-score is calculated as the average F1-scores for all classes:

 $$ F1_{macro}=\frac{1}{C}\sum_{c=1}^{C}F1_{c} $$ 

where C is the number of classes.


<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2">Model</td><td colspan="2">CaseLaw</td><td colspan="2">CAIL18</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Acc</td><td style='text-align: center; word-wrap: break-word;'>F1</td><td style='text-align: center; word-wrap: break-word;'>Acc</td><td style='text-align: center; word-wrap: break-word;'>F1</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>CNN(with BERT)</td><td style='text-align: center; word-wrap: break-word;'>0.58</td><td style='text-align: center; word-wrap: break-word;'>0.54</td><td style='text-align: center; word-wrap: break-word;'>0.39</td><td style='text-align: center; word-wrap: break-word;'>0.11</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Legal-BERT</td><td style='text-align: center; word-wrap: break-word;'>0.63</td><td style='text-align: center; word-wrap: break-word;'>0.61</td><td style='text-align: center; word-wrap: break-word;'>0.22</td><td style='text-align: center; word-wrap: break-word;'>0.03</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Lawformer</td><td style='text-align: center; word-wrap: break-word;'>0.53</td><td style='text-align: center; word-wrap: break-word;'>0.31</td><td style='text-align: center; word-wrap: break-word;'>0.38</td><td style='text-align: center; word-wrap: break-word;'>0.12</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GPT-3.5-turbo</td><td style='text-align: center; word-wrap: break-word;'>0.49</td><td style='text-align: center; word-wrap: break-word;'>0.27</td><td style='text-align: center; word-wrap: break-word;'>0.26</td><td style='text-align: center; word-wrap: break-word;'>0.04</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GPT-4o</td><td style='text-align: center; word-wrap: break-word;'>0.64</td><td style='text-align: center; word-wrap: break-word;'>0.64</td><td style='text-align: center; word-wrap: break-word;'>0.31</td><td style='text-align: center; word-wrap: break-word;'>0.05</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Debate-Feedback(single)</td><td style='text-align: center; word-wrap: break-word;'>0.66</td><td style='text-align: center; word-wrap: break-word;'>0.65</td><td style='text-align: center; word-wrap: break-word;'>0.42</td><td style='text-align: center; word-wrap: break-word;'>0.16</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Debate-Feedback(assistant)</td><td style='text-align: center; word-wrap: break-word;'>0.67</td><td style='text-align: center; word-wrap: break-word;'>0.66</td><td style='text-align: center; word-wrap: break-word;'>0.45</td><td style='text-align: center; word-wrap: break-word;'>0.16</td></tr></table>

<div style="text-align: center;">Table 2: Comparison of models on CaseLaw and CAIL18 datasets. All judge’s and debaters’ LMs in experiments are based on the GPT-4o model and T = 0.5.</div>


### 4.4 Experimental Results

The experimental results demonstrate the effectiveness of the Debate-Feedback model, with the inclusion of an assistant model in the feedback loop enhancing prediction reliability and providing more robust results compared to the single Debate-Feedback model. These results validate the strength of our approach in improving the accuracy and consistency of legal judgment predictions. Our experimental results are shown in Table[2], Figure[2] and Figure[3].

CaseLaw Dataset Performance For the CaseLaw dataset, the Debate-Feedback model outperformed GPT-4o, GPT-3.5-turbo, Legal-BERT, CNN and Lawformer. The model with the assistant achieved an accuracy of 0.67 and an F1-score of 0.66, while the single Debate-Feedback model obtained slightly lower performance with an accuracy of 0.66 and an F1-score of 0.65. These results show that our method improves the performance of pre-train legal domain models, which only achieved an accuracy of 0.63 and an F1-score of 0.61. The assistant model's inclusion in the feedback loop improves the reliability of predictions, making it more robust compared to the single model.

CAIL18 Dataset Performance On the Chinese legal dataset CAIL18, the Debate-Feedback model achieved a remarkable accuracy of 0.45, significantly surpassing GPT-4o (accuracy 0.31) and GPT-3.5-turbo (accuracy 0.26). The model with an assistant component further improved the F1-score to 0.16, highlighting the ability of the assistant model to refine predictions and correct any inconsistencies in the debate phase. These results also suggest that the Debate-Feedback model is more versatile in handling cross-linguistic challenges compared to other models.

Comparison with basic reasoning methods