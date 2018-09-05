# Twitter-Bot-DSGA1001
Twitter Bot Classification Project for DS-GA 1001 in Fall 2017.

# Bot Classification To Improve Advertising Quality on Twitter

### Authored by: Stephen Carrow, Chris Rogers,  Joe Sloan, Isaac Haberman

## Summary

For our project, we studied the case of bot classification on Twitter as a method to remove non-human users from Twitter’s advertising exchanges.  Using features derived from tweet sentiment analysis, account features, and tweet timing, we trained a number of classification models to minimize the cost of improving advertising quality.  We found that ensemble methods such as gradient boosting and random forests provided the best results in terms of both AUC and our stated cost matrix, with random forests slightly edging the other models. 

## Business Scenario

As data scientists working on Twitter's advertising team, our goal is to apply data  science methods and utilize large datasets to improve advertising on Twitter.  We have been receiving complaints from our advertising partners about delivering paid ads to non-human users in our ad exchange.  These complaints have made it clear that there is significant risk of decreased ad spending from some large clients if these quality issues continue.  Our task is to minimize overall profit loss by identifying and removing bot accounts from the ad exchange.  

We formulated our task as a classification problem on individual twitter accounts with two categories: human or bot, with bot as the positive case. Our threshold for removing likely bots  is determined using a cost-benefit matrix with estimated costs for failing to remove bot accounts and eliminating human accounts.  Through conversations with our advertising partners and subsequent analysis, we estimate that failing to remove bot accounts would lead to a loss in gross revenue on the order of $50M per year due to several key ad buyers significantly reducing spending if we continue to serve ads to non-human users.  Using this, we estimated a cost of -$0.06 per 1000 ad requests for each active non-human user (FN) in our exchange and a cost of -$0.03 per 1000 ad requests for removing an average user from our exchange (FP).

![Alt text](/images/botcosttable.png?raw=true)

Table 1: Cost-Benefit Matrix representing the expected change to revenue per 1000 ad requests per user.

## Data Collection and Preparation

In order to classify twitter users as non-human,  we looked at the two major sets of information available from Twitter for a user - account attributes and tweets.  Twitter provides an API for capturing this data, though without privileged access it is severely throttled to prevent abuse.  We consider a single data record to consist of a user’s account level information as well as aggregate information from their recent tweets.  

To gather this data we first needed to select a subset of users from Twitter’s reported 330M active users. As we realized this was a daunting task, we surveyed some of the existing research literature on the topic, relying particularly on “Online Human-Bot Interactions: Detection, Estimation, and Characterization” by Varol et al. 2017 and “Detecting Automation of Twitter Accounts: Are You a Human, Bot, or Cyborg? “ by Chu et al. 2012.
Since there was no existing bot label for this data, we needed to create labels for the accounts we sampled.  This label is not immediately obvious even to a human observer, and ideally label generation would be done by several people and the score averaged or aggregated in some other way.  This was not practical given our time and resources.  Fortunately, there were several public lists that are labeled, though the method for compiling the list and for labelling it vary widely.  We chose to use the list of users compiled by the authors of the Varol 2017 paper.  Their list was both rigorously compiled and relatively recent.  To train with a reasonable number of bot cases they oversampled the bot case (which they describe as ~10% of the population).

Once we had a list of twitter ids with an established bot or human label, we built a data scraping application using python and the public Twitter API (Both the Twython and Tweepy packages were used, the latter for its superior handling of Twitter’s throttling).  One concern with this method was that some accounts could have been disabled.  This did prove to be true, whether it was through privacy settings changes or account deletion, and did mean we 

![Alt text](/images/featuretable.png?raw=true)

Figure 1: Mutual information scores and direction of correlation for training data features generated from fully grown Decision Tree classifier with entropy criterion.

were unable to use some accounts from the list. Another possible concern is that an account could have switched from human to bot or vice-versa.  We assumed that this was unlikely to happen in significant quantities over only a few months and that a few such transitions would not invalidate our results.  

For each user we requested all account info and the last 200 tweets, storing them in user and tweet csv files respectively.  Two hundred tweets was chosen because it is the default for a single API call and we felt this was a reasonably large sample.  We selected a subset of these fields as features and engineered a number of other features from this data.  We removed several fields that were difficult to evaluate or codify (like profile_sidebar_boarder_color or unrestricted text fields) or that our original decision tree showed to add very little information. 

Outside the pre-existing fields chosen as features, three main types of engineered features were generated.   The first were features generated from follower counts, namely reputation and taste.  Reputation is the ratio of followers to users following and taste is the average reputation of a user’s followers.  These are two commonly cited (Varol et al. 2017, Chu et al. 2012, etc)  features that are generally viewed to be predictive because bots often randomly follow people (trying to attract attention and followers) but are rarely followed back except by other bots.  Generating taste required recursively investigating each user’s friends.  

![Alt text](/images/pcacharts.png?raw=true)

Figure 2: (Left) Scatter plot of first two principal components colored by class labels.  The classes show some differences. (Right) The cumulative explained variance of over principal components shows less than 20 components capture 90% of variance. 

This was the biggest challenge of the scraping process, as it required sampling hundred of times more user accounts (users could, and did, have up to 5000 friends).   Each team member generated app tokens that were used across multiple processes running in parallel.  

The next major feature was sentiment analysis of the tweet text.  Again, a python script was written to parse the raw tweets and generate a series of scores.  Two scoring procedures were used, which utilized different scoring dictionaries and produced scores relating to different aspects of tweet sentiment.  The VADER sentiment package, which is optimized for social media data, was used to generate valence scores that take into account punctuation, capitalization and emojis often used in tweets.  We also developed a python function to apply the scoring dictionary created by (Warriner, A.B., Kuperman, V. & Brysbaert, M. Behav Res 2013), which contained a large vocabulary that was used to generate valence, dominance and arousal scores.  Each user’s tweets were scored for these sentiment values (valence, dominance, arousal etc.), and aggregated to the account level using the 5-number summaries and standard deviation of each score, which were included as features in the final data set.
The last feature group was the interarrival time of users tweets.  It has been shown (Varol et. al. 2017) that human tweets tend to arrive at a more random interval, where bot tweets are often programmatically fixed, and thus evenly spaced (once an hour for instance).   A final python script was written to parse the tweets and compute the mean and standard deviation of the interarrival distribution for each user, as this could sufficiently distinguish between these users.  Parametric model evaluation was considered, but variation in distributions led us to favor the descriptive statistics used.

In the cases where we could not compute values (because there were no friends, or no tweets for instance) we imputed the value using an average and added a “missing value” field to look for any relationship in the missing data.

## Modeling and Evaluation

Our data consist of 2,426 instances of aggregated account level features available in the user profile data from the Twitter API, as well as account level summaries of tweet features such as tweet sentiment scores, temporal features of tweeting behavior and account reputation scores constructed using network data for each user.  The data were normalized and the feature space was reduced based on feature analysis performed prior to modeling.  We randomly split the 2,426 records into training (80%) and test (20%) splits.  Our modeling and evaluation covered two classes of algorithms, namely separating hyperplanes and decision trees, from which we selected Logistic Regression, Support Vector Machine, Gradient Boosted Trees and Random Forest for evaluation.  	

We decided against exploration of KNN and Neural Networks.  PCA analysis on our training data revealed that approximately 20 components were required to capture 90% of total variance, and KNN is unlikely to perform well given the relatively high dimensional input space.  Additionally, while Neural Networks are a powerful class of algorithms, they generally require large datasets to perform well and thus are not suited for our training data.

We suspected that Logistic Regression may perform well on our training set due to the relatively small data size.  However, (Varol et. al. 2017) found that Random Forests outperformed other algorithm classes on a similar problem, which motivated our exploration of Random Forest and Gradient Boosted Trees.  We also suspected these algorithms would perform well on our data due to the variance reducing qualities of their formulations and their ability to model non-linear functions.  While we expected these more complex models to have lower bias, training required tuning of additional hyper-parameters to avoid overfitting and was thus more challenging.  Finally, we evaluated SVM with non-linear kernels and Logistic Regression with polynomial features.  These experiments were intended to explore algorithms with few hyper-parameters to tune that could also have reduced bias compared to Logistic Regression due to training on non-linear features.

Our baseline model was Logistic Regression with default parameters from Scikit-Learn.  After training on our training-split this model achieved an AUC score of 0.817 on the test dataset.  While ROC curves and AUC allow us to evaluate each model's predictive performance, these analyses are insufficient evaluation methods for our business problem, which is to maximize profits while reducing non-human request volume in our ad exchange.  Therefore, our final evaluation of model performance is based on profit curves that incorporate a cost-benefit matrix developed during our understanding of the business problem and class priors estimated by (Varol et. al. 2017).  This approach more directly assess each model’s expected impact to Twitter’s bottom line.  Our baseline model achieved a maximum profit increase of $11.64M or 23.9%.

In order to improve upon this baseline performance we performed an iterative tuning procedure for the hyper-parameters of each algorithm selected for evaluation.  To identify the best model at each step we performed grid search and examined the cross-validated AUC score and standard errors across the range of hyper-parameter input values.  We then selected the parameter value that provided the greatest regularization and had strongly overlapping confidence intervals with the best cross-validated model. 

To facilitate the tuning process we developed a python class implementing a framework to reduce this iterative training method to a single function call and save all results automatically for the user.  A key component to this simplification was the automating of hyper-parameter selection, based on the methods described above.  To achieve this we implemented a selection method inspired by gradient descent's backtracking line search that incrementally searches for the most regularized parameter with confidence intervals within a predefined threshold of the best result.  We posit our automated selection technique is more robust than manual methods when collaboratively tuning algorithms.  Furthermore, This achieved results very similar to those performed manually.

![Alt text](/images/learningcurve1.png?raw=true)

Figure 3: Fitting curve of subsample from Gradient Boosting Classifier training.

We performed cross-validation of Logistic Regression with polynomial features and SVM with non-linear kernels across a wide range of values for C.  We also searched polynomial degrees of 1, 2 and 3 for Logistic Regression and the RBF and Sigmoid kernels for SVM.  Hyper parameter tuning on the Logistic regression did not lead to a significantly better model compared to baseline.  The AUC scores on the test-split for our selected SVM model was 0.844 and and expected profit increase of $15.13M or 31.1%.

The learning curves for Logistic Regression and SVM show a plateau in AUC performance towards the upper range of tested sample sizes (50 to 2000), suggesting that given our current training data, performance may be difficult to improve even if additional data was collected.  Restricted to these algorithms, we may consider investing further resources in feature engineering or in augmenting our data with features from new sources, however, other model classes outperform Logistic Regression and SVM on our current dataset, as described below.  Therefore, by changing model classes we can improve our system's performance above baseline without incurring the additional costs of data collection.

Tuning Gradient Boosted Trees required 7 iterative steps.  The order of hyper-parameter tuning was selected to codify the hyper-parameter values according to their impact to the model's structure.  We started with the baseline Gradient Boosting Classifier we proceeded to tune maximum depth, minimum samples per leaf, maximum features per split and subsample percentage of training instances per tree.  We then tuned the learning rate and the number of estimators in one iteration as these parameters are highly dependent.  Examining the results at this step revealed that a learning rate of 0.01 was optimal, however, AUC was improving at the top of our range for number of estimators.  Therefore, a final tuning iteration re-explored the number of estimators, which resulted selecting in many more estimators.  The tuned Gradient Boosting model achieved an AUC score of 0.871 on the test-split and maximum profit increase of $16.96M or 34.9%.

Finally, Random Forest  was tuned in a similar manner to Gradient Boosted Trees.  The order of tuning was, number of estimators, maximum features per split, minimum samples per leaf and minimum samples per split .  Notably, for many of our hyper-parameter experiments on Random Forest, the fitting curves were generally flat with random fluctuations.  This suggests that there was not a strong effect of changing these parameters.  The tuned Random Forest model achieved an AUC score of 0.865 on the test-split and a maximum profit increase of $19.02M or 39.1%.

![Alt text](/images/learningcurve2.png?raw=true)

Figure 4: Fitting curve of max features from Random Forest training.

We concluded in our Principal Component Analysis that approximately 90% of variance in our training data is captured by less than 50% of the principal components suggesting that we could apply PCA to reduce dimensionality and train simpler models.  To examine this hypothesis we tested training Gradient Boosting and Logistic Regression on PCA transformed data and performed grid search to select the optimal subset of components.  Unfortunately, neither of the PCA chained models, performed as well as our best model, Gradient Boosted Trees.

![Alt text](/images/comparisons.png?raw=true)

Figure 5: (Top Left) Profit curves showing expected profit percent change and (Top Right) dollar change from classifying none of the test samples as bots to 100% of the test samples as bots. (Bottom Left) ROC curves and (Bottom Right) learning curves for trained and baseline models.

A comparison of our models on the test-split showed that Random Forest and Gradient Boosting perform similarly and outperformed all other candidates on AUC and profit improvement -- generally dominating other models on all classification thresholds.  Random Forest is slightly better at ranking instances for high thresholds while Gradient Boosting is slightly better at ranking among lower thresholds.  Based on our estimated cost-benefit matrix and class priors we expect the Random Forest to improve profit by $19.02M or 39.1% compared a scenario where no model is used and to capture an additional $7.38M in estimated profit improvement compared to our baseline model.  

We note that our estimated costs and class priors are not exact and therefore evaluating expected profit at a single point on our curve is a risk.  However, the Random Forest model provides maximum profit improvement for a fairly wide range of thresholds surrounding the maximum profit point.  Due to its robust performance, along with its relative ease of training compared to Gradient Boosting, we selected the Random Forest method for deployment.

Our estimate of expected profit increase provides substantial motivation for deploying our solution, which will decrease request volume from non-human accounts in our ad exchange, thus alleviating concerns from our demand partners, and simultaneously improve Twitter’s profits.  Our solution achieves this result by more accurately ranking accounts on a score of “non-human account” when compared to randomized and baseline approaches and by selecting accounts for removal when the impact of doing so maximizes expected profit.

## Future Research and Deployment Strategies

While the results we saw from both Random Forests and Gradient-Boosted Trees were significant improvements over baseline, our initial data constraints imply further room for improvement.  As mentioned above, our dataset was derived from a list of accounts generated by Varol et. al (2017).  Our account list was, in practice, relatively small, and learning curves show that there is still at least a little more improvement in acquiring additional training data. The oversampling done in generating this list limits our ability to use metrics such as lift over baseline, precision, recall, or other useful evaluation metrics without adjusting to some assumed population rate for bots. Finally, we discussed earlier that there are some issues with possible change in labels or data accessibility over time that would continue to degrade without refreshing the data.  

These issues could be resolved by future work with larger and more representative account lists. Building this would require an initial investment in human-generated labels for Twitter accounts that meet a certain activity threshold and a continued investment in refreshing this data. Oversampling of non-human examples would still be implemented for the training dataset, but testing could be done on a more representative sample. Given that our models have shown success and reasonable return on investment at this stage, Twitter management could be confident at this point to invest more resources into data collection. 

Deployment becomes feasible after the investment in labeled account data.  As mentioned above, we do not expect labels to change significantly on a timescale of weeks or months.  This assumption is at least partially validated by our success with using a label list from early 2017 to to build accurate models.  Given this, our deployment strategy can afford to update labels quarterly or biannually.  Current reports show Twitter’s monthly active users at 330 million.  Given these numbers and assuming quarterly evaluation, we would need to evaluate approximately 20 users/second.  This would be a reasonable average throughput for the model evaluation if run in large batches, especially considering that Random Forests models are easily parallelized.  A larger potential bottleneck will be pulling the user and tweet data necessary for feature creation.  This can be done prior to model evaluation as a separate processing step, likely pulling from a reporting server that does not use realtime backups.  Nightly or even weekly snapshots would be sufficient for this model, so the primary concern would be choosing infrastructure that has the most available processing resources.  

Model evaluation would ideally occur monthly or quarterly, with people verifying the labels of a small subset of users. Approximately 100 million bot labels would be generated over each run of the entire user base, so these cannot all be checked.  If we plan to use data for future training and evaluation, human labels will also need to be verified.  A small team of 5-8 full time contractors could reasonably check around 1500 labels/week  (assuming around 10 minutes per label spread across a few independent checks). With a similar 80/20 split in training and testing data, a model could be trained quarterly with a fresh dataset twice the size of the data used in this report (after accounting for oversampling).  Model replacement would only occur if the new model outperforms the old labels by a noticeable margin on cost savings. Supporting this data validation and the necessary hardware to run the service would not be cheap (on the order of $1-2 million), but is still significantly less expensive than the projected costs of leaving bots unaddressed (around $19 million for best case Random Forest model).

One risk lies in advertisers not believing that we have actually addressed their concerns regarding impression quality.  This can be mitigated by pointing to outside metrics for validation of Twitter’s improvement in impression quality after the first run of labeling is complete.  There are no major ethical or legal risks to deploying this model for its intended purpose of removing bots from the advertising pool.  However, there may be a temptation for managers or other teams at Twitter to repurpose this model’s labels for other tasks involving bot detection. These problems may involve banning users or other outcomes with larger ethical implications.  Our model is not designed to handle these problems and its labels should not be used for such purposes.  Those projects can use our data and features as a starting point, but should ultimately build their own models and consider the implications of using them.

The largest risk, and the ultimate success of the deployment, comes back to the business case we established. If our understanding of the business issue is reasonably correct, especially the cost of false negatives that leave bots in the exchange, then our Random Forests model will generate significant revenue retention of almost $20 million at a cost of only $1-2 million.  If our cost-benefit matrix deviates significantly from real conditions then we may see significantly less savings.  The model’s results are competitive with other twitter bot classifiers given our limited dataset and feature space, but the success of the project will depend on the strength of the business case.

## Bibliography

Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay. Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research, 12, 2825-2830 (2011)

Wes McKinney. Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in Science Conference, 51-56 (2010)

Fernando Pérez and Brian E. Granger. IPython: A System for Interactive Scientific Computing, Computing in Science & Engineering, 9, 21-29 (2007)

Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux. The NumPy Array: A Structure for Efficient Numerical Computation, Computing in Science & Engineering, 13, 22-30 (2011)

Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

Warriner, A.B., Kuperman, V. & Brysbaert, M. Behav Res (2013) 45: 1191. Norms of valence, arousal, and dominance for 13,915 English lemmas [zipped csv]. Retrieved from https://doi.org/10.3758/s13428-012-0314-x

Tweepy - An easy-to-use Python library for accessing the Twitter API
Joshua Roesslein, et al. 2009
https://github.com/ryanmcgrath/twython

Twython - The premier Python library providing an easy way to access Twitter data.
Ryan Mcgrath, et al. 2013 
https://github.com/tweepy/tweepy

Varol Dataset - https://botometer.iuni.iu.edu/bot-repository/datasets/varol-2017/varol-2017.dat.gz

[Varol et al. 2017] Varol, 0; Ferrara, E; Davis, C; Menczer, F;  Flammin, 2017. Online Human-Bot Interactions: Detection, Estimation, and Characterization. arXiv preprint arXiv:1703.03107.
https://arxiv.org/pdf/1703.03107.pdf

[Chu et al. 2012] Chu, Z.; Gianvecchio, S.; Wang, H.; and Jajodia, S. 2012. Detecting automation of twitter accounts: Are you a human, bot, or cyborg? IEEE Tran Dependable & Secure Comput 9(6):811–824. 
http://www.cs.wm.edu/~hnw/paper/tdsc12b.pdf 

MoPub, Inc., MoPub, Inc. "Global Mobile Programmatic Trends A MoPub Marketplace Report | Q2 2016." Mopub.com. N.p., 16 Aug. 2016. Web. 6 Dec. 2017. Retrieved from
https://media.mopub.com/media/filer_public/22/b5/22b58fbf-b077-4c2c-ae41-d53d06d23dd9/mopub_global_mobile_programmatic_trends_report_-_q2_2016.pdf


