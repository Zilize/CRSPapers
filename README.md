# CRS Papers

![](https://img.shields.io/github/last-commit/Zilize/CRSPapers?color=blue) ![](https://img.shields.io/badge/PaperNumber-89-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red)

A Conversational Recommender System (CRS) is defined by [Gao et al. (2021)](https://arxiv.org/pdf/2101.09459.pdf) as following:

> *A recommendation system that can elicit the dynamic preferences of users and take actions based on their current needs through real-time multi-turn interactions using natural language.*

### Contents

- [Quick-Start](#Quick-Start)
- [Survey and Tutorial](#Survey-and-Tutorial)
  - [Survey](#Survey)
  - [Tutorial](#Tutorial)
- [Toolkit and Dataset](#Toolkit-and-Dataset)
  - [Toolkit](#Toolkit)
  - [Dataset](#Dataset)
- [Model](#Model)
  - [Attribute-based](#Attribute-based)
  - [Generation-based](#Generation-based)
  - [Others](#Others)
- [Other](#Other)
- [Thesis](#Thesis)



## Quick-Start

> A quick-start paper list including survey, tutorial, toolkit and model papers.

1. "Deep Conversational Recommender Systems: A New Frontier for Goal-Oriented Dialogue Systems". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2004.13245.pdf)
2. "Tutorial on Conversational Recommendation Systems". `RecSys(2020)` [[PDF]](http://yongfeng.me/attach/fu-recsys2020.pdf) [[Homepage]](https://conversational-recsys.github.io/)
3. **CRSLab**: "CRSLab: An Open-Source Toolkit for Building Conversational Recommender System". `ACL(2021)` [[PDF]](https://arxiv.org/pdf/2101.00939.pdf) [[Homepage]](https://github.com/RUCAIBox/CRSLab)
4. **CRM**: "Conversational Recommender System". `SIGIR(2018)` [[PDF]](https://arxiv.org/pdf/1806.03277) [[Homepage]](https://github.com/ysun30/ConvRec)
5. **SAUR**: "Towards Conversational Search and Recommendation: System Ask, User Respond". `CIKM(2018)` [[PDF]](https://par.nsf.gov/servlets/purl/10090082) [[Dataset]](http://yongfeng.me/attach/conversation.zip)
6. **EAR**: "Estimation-Action-Reflection: Towards Deep Interaction Between Conversational and Recommender Systems". `WSDM(2020)` [[PDF]](https://arxiv.org/pdf/2002.09102) [[Homepage]](https://ear-conv-rec.github.io/)
7. **CPR**: "Interactive Path Reasoning on Graph for Conversational Recommendation". `KDD(2020)` [[PDF]](https://arxiv.org/pdf/2007.00194) [[Homepage]](https://cpr-conv-rec.github.io/)
8. **ReDial**: "Towards Deep Conversational Recommendations". `NeurIPS(2018)` [[PDF]](https://arxiv.org/pdf/1812.07617) [[Dataset]](https://redialdata.github.io/website/) [[Code]](https://github.com/RaymondLi0/conversational-recommendations)
9. **KBRD**: "Towards Knowledge-Based Recommender Dialog System". `EMNLP-IJCNLP(2019)` [[PDF]](https://arxiv.org/pdf/1908.05391.pdf) [[Code]](https://github.com/THUDM/KBRD)

10. **KGSF**: "Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion". `KDD(2020)` [[PDF]](https://arxiv.org/pdf/2007.04032) [[Code]](https://github.com/Lancelot39/KGSF)



## Survey and Tutorial

### Survey

1. "Deep Conversational Recommender Systems: A New Frontier for Goal-Oriented Dialogue Systems". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2004.13245.pdf)
2. "A survey on conversational recommender systems". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2004.00646.pdf)

3. "Advances and Challenges in Conversational Recommender Systems: A Survey". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2101.09459.pdf)

### Tutorial

1. "Tutorial on Conversational Recommendation Systems". [[Homepage]](https://conversational-recsys.github.io/)
   - `RecSys(2020)` [[PDF]](http://yongfeng.me/attach/fu-recsys2020.pdf)
   - `WSDM(2021)` [[PDF]](http://yongfeng.me/attach/fu-wsdm2021.pdf)
   - `IUI(2021)` [[PDF]](http://yongfeng.me/attach/fu-iui2021.pdf)

2. "Conversational Recommendation: Formulation, Methods, and Evaluation". `SIGIR(2020)` [[PDF]](http://staff.ustc.edu.cn/~hexn/papers/sigir20-tutorial.pdf) [[Slides]](http://staff.ustc.edu.cn/~hexn/slides/sigir20-tutorial-CRS-slides.pdf)



## Toolkit and Dataset

### Toolkit

1. **CRSLab**: "CRSLab: An Open-Source Toolkit for Building Conversational Recommender System". `ACL(2021)` [[PDF]](https://arxiv.org/pdf/2101.00939.pdf) [[Homepage]](https://github.com/RUCAIBox/CRSLab)

### Dataset

1. **ConvRec**: "Conversational Recommender System". `SIGIR(2018)` [[PDF]](https://arxiv.org/pdf/1806.03277) [[Homepage]](https://github.com/ysun30/ConvRec)

2. **SAUR**: "Towards Conversational Search and Recommendation: System Ask, User Respond". `CIKM(2018)` [[PDF]](https://par.nsf.gov/servlets/purl/10090082) [[Download]](http://yongfeng.me/attach/conversation.zip)
3. **EAR**: "Estimation-Action-Reflection: Towards Deep Interaction Between Conversational and Recommender Systems". `WSDM(2020)` [[PDF]](https://arxiv.org/pdf/2002.09102) [[Homepage]](https://ear-conv-rec.github.io/)
4. **CPR**: "Interactive Path Reasoning on Graph for Conversational Recommendation". `KDD(2020)` [[PDF]](https://arxiv.org/pdf/2007.00194) [[Homepage]](https://cpr-conv-rec.github.io/)
5. **ReDial**: "Towards Deep Conversational Recommendations". `NeurIPS(2018)` [[PDF]](https://arxiv.org/pdf/1812.07617) [[Homepage]](https://redialdata.github.io/website/)
6. **OpenDialKG**: "OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs". `ACL(2019)` [[PDF]](https://www.aclweb.org/anthology/P19-1081.pdf) [[Homepage]](https://github.com/facebookresearch/opendialkg)
7. **PersuasionForGood**: "Persuasion for Good: Towards a Personalized Persuasive Dialogue System for Social Good". `ACL(2019)` [[PDF]](https://arxiv.org/pdf/1906.06725.pdf) [[Homepage]](https://gitlab.com/ucdavisnlp/persuasionforgood)
8. **CCPE**: "Coached Conversational Preference Elicitation: A Case Study in Understanding Movie Preferences". `SIGDial(2019)` [[PDF]](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/54521b4011d0c2a19eaade8005ff4a499f754301.pdf) [[Homepage]](https://github.com/google-research-datasets/ccpe)
9. **TG-ReDial**: "Towards Topic-Guided Conversational Recommender System". `COLING(2020)` [[PDF]](https://arxiv.org/pdf/2010.04125) [[Homepage]](https://github.com/RUCAIBox/TG-ReDial)
10. **GoRecDial**: "Recommendation as a Communication Game: Self-Supervised Bot-Play for Goal-oriented Dialogue". `EMNLP(2019)` [[PDF]](https://arxiv.org/pdf/1909.03922) [[Download]](https://drive.google.com/drive/folders/1nilk6FUktW2VjNlATdM0VMehzSOPIvJ0?usp=sharing)
11. **DuRecDial**: "Towards Conversational Recommendation over Multi-Type Dialogs". `ACL(2020)` [[PDF]](https://arxiv.org/pdf/2005.03954) [[Download]](https://baidu-nlp.bj.bcebos.com/DuRecDial.zip)
12. **INSPIRED**: "INSPIRED: Toward Sociable Recommendation Dialogue Systems". `EMNLP(2020)` [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.654.pdf) [[Homepage]](https://github.com/sweetpeach/Inspired)
13. **MGConvRex**: "User Memory Reasoning for Conversational Recommendation". `ACL(2020)` [[PDF]](https://arxiv.org/pdf/2006.00184)
14. **COOKIE**: "COOKIE: A Dataset for Conversational Recommendation over Knowledge Graphs in E-commerce". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2008.09237) [[Homepage]](https://github.com/zuohuif/COOKIE)
15. **IARD**: "Predicting User Intents and Satisfaction with Dialogue-based Conversational Recommendations". `UMAP(2020)` [[PDF]](http://www.comp.hkbu.edu.hk/~lichen/download/Cai_UMAP20.pdf) [[Homepage]](https://wanlingcai.github.io/files/2020/UMAP2020_dataset_readme.html)
16. **DuRecDial 2.0**: "DuRecDial 2.0: A Bilingual Parallel Corpus for Conversational Recommendation". `EMNLP(2021)` [[PDF]](https://arxiv.org/pdf/2109.08877.pdf) [[Homepage]](https://github.com/liuzeming01/DuRecDial)
17. **MMConv**: "MMConv: An Environment for Multimodal Conversational Search across Multiple Domains". `SIGIR(2021)` [[PDF]](https://liziliao.github.io/papers/2021sigir_mmconv.pdf) [[Homepage]](https://github.com/liziliao/MMConv)
18. **INSPIRED2**: "INSPIRED2: An Improved Dataset for Sociable Conversational Recommendation." `RecSys(2022)` [[PDF]](https://arxiv.org/pdf/2208.04104.pdf) [[Homepage]](https://github.com/ahtsham58/INSPIRED2)


## Model

### Attribute-based

> Attribute-based CRSs typically capture user preferences by asking queries about item attributes and generates responses using pre-defined templates.

1. "Towards Conversational Recommender Systems". `KDD(2016)` [[PDF]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/rfp0063-christakopoulou.pdf)
2. **CRM**: "Conversational Recommender System". `SIGIR(2018)` [[PDF]](https://arxiv.org/pdf/1806.03277) [[Homepage]](https://github.com/ysun30/ConvRec)
3. **SAUR**: "Towards Conversational Search and Recommendation: System Ask, User Respond". `CIKM(2018)` [[PDF]](https://par.nsf.gov/servlets/purl/10090082) [[Dataset]](http://yongfeng.me/attach/conversation.zip)
4. **Q&R**: "Q&R: A Two-Stage Approach toward Interactive Recommendation". `KDD(2018)` [[PDF]](http://www.alexbeutel.com/papers/q-and-r-kdd2018.pdf)
5. "Dialogue based recommender system that flexibly mixes utterances and recommendations". `WI(2019)` [[Link]](https://ieeexplore.ieee.org/abstract/document/8909617)
6. **EAR**: "Estimation-Action-Reflection: Towards Deep Interaction Between Conversational and Recommender Systems". `WSDM(2020)` [[PDF]](https://arxiv.org/pdf/2002.09102) [[Homepage]](https://ear-conv-rec.github.io/)
7. **CPR**: "Interactive Path Reasoning on Graph for Conversational Recommendation". `KDD(2020)` [[PDF]](https://arxiv.org/pdf/2007.00194) [[Homepage]](https://cpr-conv-rec.github.io/)
8. **CRSAL**: "CRSAL: Conversational Recommender Systems with Adversarial Learning". `TOIS(2020)` [[PDF]](https://repository.kaust.edu.sa/bitstream/handle/10754/665725/TOIS.pdf?sequence=1&isAllowed=y) [[Code]](https://github.com/XuhuiRen/CRSAL)
9. **Qrec**: "Towards Question-Based Recommender Systems". `SIGIR(2020)` [[PDF]](https://arxiv.org/pdf/2005.14255.pdf) [[Code]](https://github.com/JieZouIR/Qrec)
10. **ConTS**: "Seamlessly Unifying Attributes and Items: Conversational Recommendation for Cold-Start Users". `TOIS(2021)` [[PDF]](https://arxiv.org/pdf/2005.12979) [[Code]](https://github.com/xiwenchao/conTS-TOIS-2021)
11. **UNICORN**: "Unified Conversational Recommendation Policy Learning via Graph-based Reinforcement Learning". `SIGIR(2021)` [[PDF]](https://arxiv.org/pdf/2105.09710.pdf) [[Code]](https://github.com/dengyang17/unicorn)
12. **KBQG**: "Learning to Ask Appropriate Questions in Conversational Recommendation". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2105.04774.pdf) [[Code]](https://github.com/XuhuiRen/KBQG)
13. **FPAN**: "Adapting User Preference to Online Feedback in Multi-round Conversational Recommendation". `WSDM(2021)` [[Link]](https://dl.acm.org/doi/abs/10.1145/3437963.3441791) [[Code]](https://github.com/xxkkrr/FPAN)

14. "Developing a Conversational Recommendation System for Navigating Limited Options". `CHI(2021)` [[PDF]](https://arxiv.org/pdf/2104.06552.pdf)
15. **MCMIPL**: "Multiple Choice Questions based Multi-Interest Policy Learning for Conversational Recommendation." `WWW(2022)` [[PDF]](https://arxiv.org/pdf/2112.11775.pdf) [[Code]](https://github.com/ZYM6-6/MCMIPL)
16. "Quantifying and Mitigating Popularity Bias in Conversational Recommender Systems." `CIKM(2022)` [[PDF]](https://arxiv.org/pdf/2208.03298.pdf)
17. **MINICORN**: "Minimalist and High-performance Conversational Recommendation with Uncertainty Estimation for User Preference." `arXiv(2022)` [[PDF]](https://arxiv.org/pdf/2206.14468.pdf)
18. **CRIF**: "Learning to Infer User Implicit Preference in Conversational Recommendation." `SIGIR(2022)` [[PDF]](https://dl.acm.org/doi/abs/10.1145/3477495.3531844)
19. **HICR**: "Conversational Recommendation via Hierarchical Information Modeling." `SIGIR(2022)` [[PDF]](https://dl.acm.org/doi/abs/10.1145/3477495.3531830)
20. **MetaCRS**: "Meta Policy Learning for Cold-Start Conversational Recommendation." `WSDM(2023)` [[PDF]](https://arxiv.org/pdf/2205.11788.pdf)

### Generation-based

> Compared to attribute-based CRSs, generation-based CRSs pay more attention to generate human-like responses in natural language.

1. **ReDial**: "Towards Deep Conversational Recommendations". `NeurIPS(2018)` [[PDF]](https://arxiv.org/pdf/1812.07617) [[Code]](https://github.com/RaymondLi0/conversational-recommendations) [[Dataset]](https://redialdata.github.io/website/)

2. **KBRD**: "Towards Knowledge-Based Recommender Dialog System". `EMNLP-IJCNLP(2019)` [[PDF]](https://arxiv.org/pdf/1908.05391.pdf) [[Code]](https://github.com/THUDM/KBRD)
3. **GoRecDial**: "Recommendation as a Communication Game: Self-Supervised Bot-Play for Goal-oriented Dialogue". `EMNLP(2019)` [[PDF]](https://arxiv.org/pdf/1909.03922) [[Code]](https://github.com/facebookresearch/ParlAI) [[Dataset]](https://drive.google.com/drive/folders/1nilk6FUktW2VjNlATdM0VMehzSOPIvJ0?usp=sharing)
4. **DialKG Walker**: "OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs". `ACL(2019)` [[PDF]](https://www.aclweb.org/anthology/P19-1081.pdf) [[Code]](https://github.com/madcpt/OpenDialKG) [[Dataset]](https://github.com/facebookresearch/opendialkg)
5. **DCR**: "Deep Conversational Recommender in Travel". `TKDE(2020)` [[PDF]](https://arxiv.org/pdf/1907.00710.pdf) [[Code]](https://github.com/truthless11/DCR)
6. **KGSF**: "Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion". `KDD(2020)` [[PDF]](https://arxiv.org/pdf/2007.04032) [[Code]](https://github.com/Lancelot39/KGSF)
7. **MGCG**: "Towards Conversational Recommendation over Multi-Type Dialogs". `ACL(2020)` [[PDF]](https://arxiv.org/pdf/2005.03954.pdf) [[Code]](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2020-DuRecDial) [[Dataset]](https://baidu-nlp.bj.bcebos.com/DuRecDial.zip)
8. **ECR**: "Towards Explainable Conversational Recommendation". `IJCAI(2020)` [[PDF]](https://www.ijcai.org/Proceedings/2020/0414.pdf)
9. **INSPIRED**: "INSPIRED: Toward Sociable Recommendation Dialogue Systems". `EMNLP(2020)` [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.654.pdf) [[Homepage]](https://github.com/sweetpeach/Inspired)
10. **TG-ReDial**: "Towards Topic-Guided Conversational Recommender System". `COLING(2020)` [[PDF]](https://arxiv.org/pdf/2010.04125) [[Homepage]](https://github.com/RUCAIBox/TG-ReDial)
11. **MGConvRex**: "User Memory Reasoning for Conversational Recommendation". `COLING(2020)` [[PDF]](https://arxiv.org/pdf/2006.00184)
12. **KGConvRec**: "Suggest me a movie for tonight: Leveraging Knowledge Graphs for Conversational Recommendation". `COLING(2020)` [[PDF]](https://www.aclweb.org/anthology/2020.coling-main.369.pdf) [[Code]](https://github.com/rajbsk/KG-conv-rec)
13. **CR-Walker**: "Bridging the Gap between Conversational Reasoning and Interactive Recommendation". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2010.10333.pdf) [[Code]](https://github.com/truthless11/CR-Walker)
14. **RevCore**: "RevCore: Review-augmented Conversational Recommendation". `ACL-Findings(2021)` [[PDF]](https://arxiv.org/pdf/2106.00957.pdf) [[Code]](https://github.com/JD-AI-Research-NLP/RevCore)
15. **KECRS**: "KECRS: Towards Knowledge-Enriched Conversational Recommendation System". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2105.08261.pdf)
16. "Category Aware Explainable Conversational Recommendation". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2103.08733.pdf)
17. **DuRecDial 2.0**: "DuRecDial 2.0: A Bilingual Parallel Corpus for Conversational Recommendation". `EMNLP(2021)` [[PDF]](https://arxiv.org/pdf/2109.08877.pdf) [[Dataset]](https://github.com/liuzeming01/DuRecDial)
18. **NTRD**: "Learning Neural Templates for Recommender Dialogue System." `EMNLP(2021)` [[PDF]](https://arxiv.org/pdf/2109.12302.pdf) [[Code]](https://github.com/jokieleung/NTRD)
19. **CRFR**: "CRFR: Improving Conversational Recommender Systems via Flexible Fragments Reasoning on Knowledge Graphs." `EMNLP(2021)` [[PDF]](https://aclanthology.org/2021.emnlp-main.355.pdf)
20. **RID**: "Finetuning Large-Scale Pre-trained Language Models for Conversational Recommendation with Knowledge Graph." `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2110.07477.pdf) [[Code]](https://github.com/Lingzhi-WANG/PLM-BasedCRS)
21. **RecInDial**: "RecInDial: A Unified Framework for Conversational Recommendation with Pretrained Language Models." `AACL(2022)` [[PDF]](https://arxiv.org/pdf/2110.07477.pdf) [[Code]](https://github.com/Lingzhi-WANG/PLM-BasedCRS)
22. **MESE**: "Improving Conversational Recommendation Systems’ Quality with Context-Aware Item Meta Information." `NAACL(2022)` [[PDF]](https://arxiv.org/pdf/2112.08140.pdf) [[Code]](https://github.com/by2299/MESE)
23. **C2-CRS**: "C2-CRS: Coarse-to-Fine Contrastive Learning for Conversational Recommender System." `WSDM(2022)` [[PDF]](https://arxiv.org/pdf/2201.02732.pdf) [[Code]](https://github.com/RUCAIBox/WSDM2022-C2CRS)
24. **BARCOR**: "BARCOR: Towards A Unified Framework for Conversational Recommendation Systems." `arXiv(2022)` [[PDF]](https://arxiv.org/pdf/2203.14257.pdf)
25. **UniMIND**: "A Unified Multi-task Learning Framework for Multi-goal Conversational Recommender Systems." `TOIS(2023)` [[PDF]](https://arxiv.org/pdf/2204.06923.pdf) [[Code]](https://github.com/dengyang17/unimind)
26. **UCCR**: "User-Centric Conversational Recommendation with Multi-Aspect User Modeling." `SIGIR(2022)` [[PDF]](https://arxiv.org/pdf/2204.09263.pdf) [[Code]](https://github.com/lisk123/UCCR)
27. **UPCR**: "Variational Reasoning about User Preferences for Conversational Recommendation." `SIGIR(2022)` [[PDF]](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/ren-2022-variational.pdf) [[Code]](https://github.com/tianz2020/UPCR)
28. **TSCR**: "Improving Conversational Recommender Systems via Transformer-based Sequential Modelling." `SIGIR(2022)` [[PDF]](https://dl.acm.org/doi/abs/10.1145/3477495.3531852)
29. **CCRS**: "Customized Conversational Recommender Systems." `ECML-PKDD(2022)` [[PDF]](https://arxiv.org/pdf/2207.00814.pdf)
30. **UniCRS**: "Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning." `KDD(2022)` [[PDF]](https://arxiv.org/pdf/2206.09363.pdf) [[Code]](https://github.com/RUCAIBox/UniCRS)
31. **EGCR**: "EGCR: Explanation Generation for Conversational Recommendation." `arXiv(2022)` [[PDF]](https://arxiv.org/pdf/2208.08035.pdf)
32. "Improving Conversational Recommender System via Contextual and Time-Aware Modeling with Less Domain-Specific Knowledge." `arXiv(2022)` [[PDF]](https://arxiv.org/pdf/2209.11386.pdf)
33. **DICR**: "Aligning Recommendation and Conversation via Dual Imitation." `arXiv(2022)` [[PDF]](https://arxiv.org/pdf/2211.02848.pdf)


### Others

1. **Converse-Et-Impera**: "Converse-Et-Impera: Exploiting Deep Learning and Hierarchical Reinforcement Learning for Conversational Recommender Systems". `AI*IA(2017)` [[PDF]](https://www.researchgate.net/profile/Alessandro-Suglia/publication/320875588_Converse-Et-Impera_Exploiting_Deep_Learning_and_Hierarchical_Reinforcement_Learning_for_Conversational_Recommender_Systems/links/5bf6ad1592851c6b27d27324/Converse-Et-Impera-Exploiting-Deep-Learning-and-Hierarchical-Reinforcement-Learning-for-Conversational-Recommender-Systems.pdf)

2. "A Model of Social Explanations for a Conversational Movie Recommendation System". `HAI(2019)` [[PDF]](https://eprints.gla.ac.uk/193937/7/193937.pdf)
3. "Dynamic Online Conversation Recommendation". `ACL(2020)` [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.305.pdf) [[Code]](https://github.com/zxshamson/dy-conv-rec)
4. **IAI MovieBot**: "IAI MovieBot: A Conversational Movie Recommender System". `CIKM(2020)` [[PDF]](https://arxiv.org/pdf/2009.03668.pdf) [[Code]](https://github.com/iai-group/moviebot)
5. **ConUCB**: "Conversational Contextual Bandit: Algorithm and Application". `WWW(2020)` [[PDF]](https://arxiv.org/pdf/1906.01219.pdf) [[Code]](https://github.com/Xiaoyinggit/ConUCB)
6. **Cora**: "A Socially-Aware Conversational Recommender System for Personalized Recipe Recommendations". `HAI(2020)` [[PDF]](https://www.researchgate.net/profile/Florian-Pecune/publication/346716927_A_Socially-Aware_Conversational_Recommender_System_for_Personalized_Recipe_Recommendations/links/5fcf621045851568d149d95e/A-Socially-Aware-Conversational-Recommender-System-for-Personalized-Recipe-Recommendations.pdf)
7. "Conversational Music Recommendation based on Bandits". `ICKG(2020)` [[Link]](https://ieeexplore.ieee.org/abstract/document/9194509/)
8. **n-by-p**: "Navigation-by-preference: a new conversational recommender with preference-based feedback". `IUI(2020)` [[PDF]](http://www.cs.ucc.ie/~dgb/papers/Rana-Bridge-2020.pdf)
9. "A Bayesian Approach to Conversational Recommendation Systems". `AAAI Workshop(2020)` [[PDF]](https://arxiv.org/pdf/2002.05063.pdf)
10. "Towards Retrieval-based Conversational Recommendation". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2109.02311.pdf)
11. ""It doesn’t look good for a date": Transforming Critiques into Preferences for Conversational Recommendation Systems". `EMNLP(2021)` [[PDF]](https://arxiv.org/pdf/2109.07576.pdf)



## Other

1. **CCPE**: "Coached Conversational Preference Elicitation: A Case Study in Understanding Movie Preferences". `SIGDial(2019)` [[PDF]](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/54521b4011d0c2a19eaade8005ff4a499f754301.pdf) [[Dataset]](https://github.com/google-research-datasets/ccpe)
2. "Leveraging Historical Interaction Data for Improving Conversational Recommender System". `CIKM(2020)` [[PDF]](https://arxiv.org/pdf/2008.08247.pdf) [[Code]](https://github.com/Lancelot39/Pre-CRS)
3. "Evaluating Conversational Recommender Systems via User Simulation". `KDD(2020)` [[PDF]](https://arxiv.org/pdf/2006.08732.pdf) [[Code]](https://github.com/iai-group/UserSimConvRec)
4. "End-to-End Learning for Conversational Recommendation: A Long Way to Go?". `RecSys(2020)` [[PDF]](http://ceur-ws.org/Vol-2682/short1.pdf) [[Material]](https://drive.google.com/drive/folders/10gPOmaiFrZjIULIa3LsdmuyvJvnCV_Xq)
5. "What Does BERT Know about Books, Movies and Music? Probing BERT for Conversational Recommendation". `RecSys(2020)` [[PDF]](https://arxiv.org/pdf/2007.15356.pdf) [[Code]](https://github.com/Guzpenha/ConvRecProbingBERT)
6. "Latent Linear Critiquing for Conversational Recommender Systems". `WWW(2020)` [[PDF]](http://www.inago.com/wp-content/uploads/2020/08/UofT-Sanner_www20_llc.pdf) [[Code]](https://github.com/k9luo/LatentLinearCritiquingforConvRecSys)
7. "A Ranking Optimization Approach to Latent Linear Critiquing for Conversational Recommender Systems". `RecSys(2020)` [[Link]](https://dl.acm.org/doi/abs/10.1145/3383313.3412240) [[Code]](https://github.com/litosly/RankingOptimizationApproachtoLLC)
8. "A Comparison of Explicit and Implicit Proactive Dialogue Strategies for Conversational Recommendation". `LREC(2020)` [[PDF]](https://www.aclweb.org/anthology/2020.lrec-1.54.pdf)
9. "Predicting User Intents and Satisfaction with Dialogue-based Conversational Recommendations". `UMAP(2020)` [[PDF]](http://www.comp.hkbu.edu.hk/~lichen/download/Cai_UMAP20.pdf) [[Dataset]](https://wanlingcai.github.io/files/2020/UMAP2020_dataset_readme.html)
10. **ConveRSE**: "Conversational Recommender Systems and natural language: A study through the ConveRSE framework". `Decision Support Systems(2020)` [[Link]](https://www.sciencedirect.com/science/article/pii/S0167923620300051) [[Dataset]](https://github.com/swapUniba/ConvRecSysDataset)

11. "On Estimating the Training Cost of Conversational Recommendation Systems". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2011.05302.pdf)



## Thesis

1. "Recommendation in Dialogue Systems". By [Yueming Sun](https://scholar.google.com/citations?user=UOYpBu4AAAAJ)(2019). [[PDF]](https://escholarship.org/content/qt4rs1s3ms/qt4rs1s3ms.pdf)

2. "Advanced Method Towards Conversational Recommendation". By [Yisong Miao](https://yisong.me/)(2020). [[PDF]](https://yisong.me/publications/Yisong_master_thesis-final.pdf)
