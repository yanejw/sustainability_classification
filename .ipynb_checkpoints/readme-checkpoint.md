# Capstone Project: Sustainability Classification of Fashion Products

## Background and problem statement

Sustainability is beoming increasingly more important in the global economy, with consumers indicating that they are [more likely to buy eco-friendly products](https://www.strategy-business.com/article/The-rise-of-the-eco-friendly-consumer). This means that it has become more important for businesses to show and/or prove that their products and business practices are sustainable. 

Although there are sustainability raters like [B Corp](https://www.bcorporation.net/en-us/certification/) that provide sustainability audits and certifications, the cost of the audits are often too expensive for small businesses. In this case, machine learning tools can create a sustainability classification model that provides businesses (and their consumers) a quick assessment of the sustainability of their products. 

The model may also be developed further to help businesses identify which part of their business practices are not the most sustainable, so that they can take steps to correct it.

-----------

For this project, we will focus on fashion businesses. In 2018, the fashion industry was estimated to be responsible for [10% of annual global carbon emissions](https://www.worldbank.org/en/news/feature/2019/09/23/costo-moda-medio-ambiente) - which was more than all international flights and maritime shipping combined. The fashion industry, especially fast fashion retailers, has also been called out many times for [greenwashing](https://www.straitstimes.com/multimedia/graphics/2022/11/fast-fashion-greenwashing-environmental-impact/index.html?shell). This emphasises a need for an independent and quick way to assess product sustainability in the industry.

## Project outline

Given the background and problem statement above, the project aims to do three things:
<br><br>

1. Train a machine learning model on a sample dataset to classify a variety of fashion items into five classes: Extremely non-sustainable (5), non-sustainable (4), medium sustainability (3), sustainable (2), extremely sustainable (1).
<br><br>
Different classification models, such as k-Nearest Neighbors, Random Forest, Support Vector Machine, etc. will be used.
<br><br>
As the sample dataset contains unbalanced classes, the F1-score will be used to evaluate the model.

2. Once trained, the model will be deployed on data scraped from one of Singapore's fashion e-retailer, GRAYE. GRAYE describes itself as having sustainable fashion concepts, and the model will inform us on how sustainable GRAYE's products actually are.

3. A streamlit app will also be deployed so that small businesses/consumers will be able to input details and get a prediction on whether their products are actually sustainable.

## Data selection

[Training and testing dataset](data/clothing_product_full_dataset.csv) from a 2022 research paper, [A Supervised Machine Learning Classification Framework for Clothing Products' Sustainability](https://www.mdpi.com/2071-1050/14/3/1334). <br>
[Independent dataset](data/graye_og.csv) scraped from a local eco-friendly fashion retailer, [GRAYE](https://grayestudio.com).

## Exploratory Data Analysis

### Inspecting the balance of classes

The distribution of classes is imbalanced, hence accuracy will not be a good metric to use in the evaluation of the model.

Since we want to minimise the false positives (precision) and false negatives (recall), we will use the F1 score metric (which takes into account both precision and recall) in the evaluation of the model.

In addition, we will use macro-averaging to calculate each metric in the evaluation of the model. This gives each class an equal weightage in the final evaluation, which is important given the imblanaced dataset.

### Inspecting the raw materials (numerical) features

The raw material is not normally distributed and has many outliers. This is too be expected as a garment is usually made out of at most four different materials, but the dataset is expected to capture as many different types of materials as possible. We will drop materials that are underrepresented in the dataset to reduce dimensionality, and reduce the outlier effect.

Given the outliers, we will use machine learning models (eg. Random Forest, Support Vector Machines, and k-Nearest Neighbours) that can deal with outliers to train and test our dataset. 
 
### Correlation of features

The Pearson correlation shows correlation between a few categorical features (`recylability label`, `reusability label`). However, as these labels are considered statistically signficant based on the chi-square test earlier, we will keep all features.

## Modelling Train and Test

We run five different models on the data. The following table summarises the train and test F1 scores of each model:

|Model|Train|Test|Remarks|
|-|-|-|-|
|k-Nearest Neighbors| 0.9986| 0.8661|Overfit|
|Support Vector Machine| 0.8891| 0.8893|Convergence issues|
|**Random Forest**|**0.942**| **0.922**|**Good fit**|
|Decision Trees| 0.9436| 0.9172|Good fit|
|Gradient Boost| 0.9916| 0.9267|Overfit|

For most models, the train score was higher than the test score, indicating some overfitting. Although the Support Vector Machine (SVM) model has the closest train and test scores, there were multiple convergence warnings even after tuning the hyperparameters. This indicates that SVM may not be the best model for the dataset.

The Random Forest and Decision Trees models had the best fitting train and test scores. Since interpretability is not an issue, we will select the Random Forest model as the best model, and use it for testing on the independent dataset.

### Feature Importance

The features that contribute most to the sustainability score seem to be the material it is made from (regenerated, lyocell, viscose, etc.) and the volume of materials taken to make the garment (based on type: a t-shirt would require less material to manufacture than a dress, for example).

However, it is important to note that the features importance function in random forest only shows the impact of a feature on the target variable, and not whether the impact is negative or positive. As such, we are unable to infer if the use of more organic cotton, polyester, etc. in production will lead to a higher or lower sustainability score.

For example, while natural materials like organic cotton and cotton can be considered more sustainable because they are biodegradable, a lot of water is used to grow and produce cotton for weaving. Synthetic materials like polyester use less water to produce and can be easily made from recyled materials (other recycled polyester), but they are generally not biodegradable.

## Testing on an independent dataset

We ran the model on an independent dataset scraped from local clothing brand GRAYE. The model predicted that most of GRAYE's garments were in the `Medium Sustainability` class (class 3). The model did not blindly predict according to the majority class (`Non-sustainable` - class 4) of the training and testing data set. Furthermore, GRAYE touts itself to be a sustainable clothing label, hence it would not be unreasonable for most of their products to be classified as medium sustainable.

Since the material and type of the garment contribute greatly to the classification (based on feature importance), we will analyse these features.

The table above seems to suggest that material, specifically linen (which was subsumed under the `Other_plants` feature during data cleaning), has the highest positive impact on whether a garment is sustainable. Garments that contained 100% linen were classified as extremely sustainable (class 1). Whereas garments with a majority average percentage of cotton were classified as medium sustainable (class 3). Based on domain knowledge, [linen is considered greener than cotton](https://www.treehugger.com/linen-vs-cotton-5116803#:~:text=In%20terms%20of%20raw%20material,cotton%20cultivated%20around%20the%20world.) because it requires less water and pesticides to produce.

Type (or volume) does not seem to have the highest impact on sustainability classes as there were no shorts classified as extremely sustainable (class 1) or sustainable (class 2) even though they use less material to make than trousers. This could be because it only requires about [1 more yard (90cm) to make trousers rather than shorts.](https://www.sewdiy.com/blog/video-how-to-decide-how-much-fabric-to-buy) Although the amount of material used has a impact on sustainability, it would be amplified by the number of garments made (ie, making 1000 quantities of a garment per month) rather than on the volume taken to make one garment.

Since we are able to rationalise the model's predictions with real world evidence, we are inclined to believe that the model's predictions are quite accurate, and it is ready to be deployed.

A simplified version of the model has also been deployed as a streamlit app and can be accessed [here](https://sustainabilityclassification.streamlit.app/)

## Future works

**For model improvement:**

1. Since the original dataset has imbalanced classes, perhaps future work could incorporate random sampling to balance the classes. Another possibility would be to collect more data so that the classes are balanced.
2. Based on the model's feature importance and classifications, the materials features have the most impact in the classification. Future works could experiment with more feature engineering on these kind of features to further improve the model if necessary.
3. To really ascertain the model's validity, it could be tested on another independent dataset from a clothing brand that does not claim itself to be sustainable.

**In the sustainability context:**

1. The model could be adapted for companies to improve their sustainability. For example, a company that wants to be greener could input different materials into the model, and check if the use of a particular material would make their products more sustainable. Or if using a certain manufacturing location would make the garments more sustaiable.
2. The model could also be adapted for other types of consumer goods, such as furniture, children's toys, footwear, etc.