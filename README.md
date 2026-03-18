# Alzheimer's Disease Multi-Modal Classification with Federated Learning (基于联邦学习的阿尔茨海默症多模态分类研究)

## Project Overview
This project focuses on the early diagnosis of Alzheimer's Disease (AD) and addresses the key challenges of multi-modal heterogeneity, sensitive patient data privacy protection, and sample class imbalance in biomedical research. We propose a privacy-preserving federated learning framework based on knowledge distillation, integrating MRI brain structural imaging, DNA SNP genetic data, and mRNA transcriptomic data to implement three-class classification of AD (Cognitively Normal [CN], Mild Cognitive Impairment [MCI], AD).

For each data modality, we design and optimize independent task-specific models, then achieve cross-modal model aggregation through distributed local training and soft label knowledge distillation—without sharing any raw patient data or model parameters during the whole process. This framework not only ensures high diagnostic performance of the model but also strictly complies with international privacy protection regulations such as HIPAA and GDPR.

Validated on the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database, the federated distillation fusion model achieves 82%-83% average classification accuracy and 81%-82% macro F1-score across all three modalities, with performance significantly superior to traditional federated learning algorithms (e.g., FedAvg). This work provides a scalable, ethically compliant AI solution for cross-institutional precise diagnosis of neurodegenerative diseases.

本项目围绕阿尔茨海默症（AD）早期诊断展开，解决生物医学研究中多模态数据异质性、患者敏感数据隐私保护、样本类别不平衡三大核心问题，提出基于知识蒸馏的隐私保护联邦学习框架，整合MRI脑结构影像、DNA基因SNP数据、mRNA转录组数据三类核心生物医学数据，实现阿尔茨海默症的三分类诊断（认知正常CN、轻度认知障碍MCI、阿尔茨海默症AD）。

针对每种数据模态设计并优化独立的专属模型，通过分布式本地训练与软标签知识蒸馏完成跨模态模型聚合，全程不共享任何患者原始数据及模型参数，既保障模型的高诊断性能，又严格遵循HIPAA、GDPR等国际隐私保护规范。

基于阿尔茨海默症神经影像倡议（ADNI）数据库的实验验证表明，联邦蒸馏融合模型在三模态上均实现82%-83%的平均分类准确率与81%-82%的宏平均F1分数，性能显著优于FedAvg等传统联邦学习算法，为跨机构的神经退行性疾病精准诊断提供了可扩展、符合伦理的人工智能解决方案。

## Key Models & Experimental Performance

| Modality | Optimal Model       | Test Accuracy | Macro F1-Score | Key Optimization |
|:--------|:--------------------|:--------------|:---------------|:-----------------|
| MRI      | Tuned CNN           | 83.56%        | 0.84           | 64/96/48 filters, dropout=0.2, lr=7.35×10⁻⁵ |
| DNA      | Optimized Random Forest | 82.60%    | 0.810          | max_depth=None, max_features='log2' |
| mRNA     | Custom MLP          | 60.00%        | 0.390          | [512,256,128] hidden layers, dropout=0.2 |
| Fusion   | Federated Distillation | 82%-83%    | 81%-82%        | MRI=T8, DNA=T1, mRNA=T2 (distillation temperature) |

| 模态   | 最优模型           | 测试准确率  | 宏平均F1分数 | 核心优化点                     |
|:----- |:------------------ |:---------- |:------------ |:----------------------------- |
| MRI    | 调优后卷积神经网络 | 83.56%      | 0.84          | 64/96/48卷积核、dropout=0.2、学习率7.35×10⁻⁵ |
| DNA    | 优化后随机森林     | 82.60%      | 0.810         | 无最大深度限制、最大特征数='log2' |
| mRNA   | 自定义多层感知机   | 60.00%      | 0.390         | [512,256,128]隐藏层、dropout=0.2 |
| 融合   | 联邦蒸馏模型       | 82%-83%     | 81%-82%       | 蒸馏温度：MRI=T8、DNA=T1、mRNA=T2 |


## Core Features

1. Strict Privacy Protection: Only soft label probability outputs are transmitted between clients and the central server, avoiding data leakage risks from raw data/parameter sharing. 严格隐私保护：客户端与服务端仅传输软标签概率输出，避免原始数据/模型参数共享带来的泄露风险，符合国际隐私规范。

2. Heterogeneous Model Fusion: Seamlessly integrates CNN (for imaging), tree-based ensemble models (for genetics), and MLP (for transcriptomics) to adapt to different data characteristics. 异质模型融合：无缝整合适用于影像的CNN、适用于遗传学的树基集成模型、适用于转录组的MLP，适配不同数据特性。

3. Stable & Balanced Performance: Outperforms traditional federated averaging algorithms, with balanced classification results for CN/MCI/AD (solving sample imbalance issues). 性能稳定均衡：表现优于传统联邦平均算法，对CN/MCI/AD三类样本实现均衡分类，有效解决样本不平衡问题。

4. High Scalability: The framework is adaptable for cross-institutional biomedical data collaboration and can be transferred to other neurodegenerative disease research. 高度可扩展性：框架适用于跨机构生物医学数据协同研究，可迁移至帕金森、脑胶质瘤等其他神经退行性疾病研究。

5. Reproducible Pipeline: Standardized data preprocessing and model training steps, with complete experimental visualization and result analysis. 可复现研究流程：标准化的数据预处理与模型训练步骤，配套完整的实验可视化与结果分析。

## Data Source

All experimental data is sourced from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database (https://adni.loni.usc.edu), the most authoritative public dataset for AD research. All data is de-identified and complies with the FAIR data principles and the Declaration of Helsinki, ensuring strict adherence to research ethics.

本项目所有实验数据均来自阿尔茨海默症神经影像倡议（ADNI）数据库（https://adni.loni.usc.edu），这是目前阿尔茨海默症研究领域最权威的公共数据集。数据均经过去标识化处理，符合FAIR数据原则与赫尔辛基宣言，严格遵循科研伦理规范。

## Project Structure

```
Alzheimers-Disease/
├── DNA/                  # DNA-SNP data processing & modeling
├── MRI/                  # MRI imaging data processing & modeling
├── mRNA/                 # mRNA transcriptomic data processing & modeling
├── FederatedLearning/    # Federated learning framework with knowledge distillation
├── outputs/              # Experimental results & visualizations (confusion matrix, loss curves, etc.)
├── visualization.ipynb   # Unified multi-modal result analysis & visualization
└── README.md             # Project documentation (English/Chinese)
```

## Key Methodology

1. Data Preprocessing: Unified pipeline including missing value handling, normalization/standardization, SMOTE oversampling (for class imbalance), and feature filtering.

2. Single-Modal Modeling: Modality-specific model design based on data characteristics (CNN/ViT for MRI, RF/XGBoost/SVM for DNA, MLP/1D-CNN for mRNA).

3. Federated Distillation: Adopt FedDF-based framework, train local teacher models on each modality, distill knowledge to global student model via soft labels, no raw data sharing.

4. Model Evaluation: Comprehensive metrics including accuracy, macro F1-score, confusion matrix, loss/accuracy curves, and distillation temperature sensitivity analysis.




