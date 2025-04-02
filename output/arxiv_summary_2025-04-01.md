# arXiv Summary for 2025-04-01

Categories: `cs.AI, cs.CV, cs.CY, cs.LG`
Papers Found (published/updated yesterday): 134
Papers Processed: 134
--------------------

# Category Summaries

## cs.AI

Key themes from the cs.AI research papers released yesterday include:
1. **Evaluation of AI Models:** Several papers introduce new datasets and benchmarks to evaluate the performance of AI models in tasks such as summarization, knowledge representation, reasoning, and agent capabilities.
2. **Efficiency Improvements:** Methods for enhancing efficiency in AI systems, such as incremental reasoning, efficient explanation generation, and scalable verification of complex models, are highlighted.
3. **Interpretability and Explainability:** The importance of inner interpretability in neural networks is discussed, with frameworks proposed to analyze model behavior and generate explainable outputs.
4. **Behavior Analysis and Training:** AI is utilized to analyze human behaviors, uncover motivations, and train artificial agents to pursue goals effectively and comply with specified criteria.
5. **Security and Vulnerability:** Studies reveal vulnerabilities in reasoning large language models to subtle errors and manipulations, emphasizing the need for robustness and security in AI systems.

--------------------

## cs.CV

The recent research papers in the cs.CV category highlight several key themes:
1. **Advancements in 3D Scene Mapping**: Papers like [3] and [8] introduce novel mapping systems and generative approaches that improve reconstruction accuracy and completeness in complex environments through dynamic viewpoint selection and data-dependent flows for detail synthesis.
2. **Multi-Modal Reasoning and Intelligence Evaluation**: Studies like [4] introduce datasets and modeling strategies to evaluate multi-modal reasoning skills, intelligence, and critical thinking beyond memorization using images and text in cognitive reasoning tasks.
3. **Intrinsic Decomposition and Object Detection**: Papers like [12] focus on intrinsic decomposition of geometric and material properties, while [15] and [41] discuss oriented object detection challenges, methods, and frameworks for improved performance on benchmark datasets.
4. **Efficient Vision Transformer Models**: Research such as [19] and [22] explore vision transformer-based models for hyperspectral image interpretation and active perception, demonstrating superior performance in various tasks

--------------------

## cs.CY

The recent research papers in the cs.CY category highlight two main themes: 
1. The development of WeirdFlows, a network analysis-based pipeline for detecting fraudulent transactions and non-compliant agents in AFC investigations without the need for predefined patterns or training sets, showcasing its effectiveness in identifying suspicious activities in the context of EU economic sanctions post-February 2022 `[13]`.
2. The impact of data journalism (DJ) on online comments, indicating that DJ enhances user interactivity through transparency, multimedia, statistical information, sources, and visualizations, potentially fostering democratic processes by encouraging engagement in conversations `[77]`.

--------------------

## cs.LG

From the cs.LG papers released yesterday, key themes include the development of efficient and interpretable modeling techniques like SINDy-SHRED and Scalable Mechanistic Neural Network (S-MNN) for complex spatio-temporal data analysis ([1], [2]). Additionally, advancements in multi-omic models for biological tasks, federated domain incremental learning frameworks like RefFiL, and novel Federated Learning (FL) frameworks such as pFedMoAP were highlighted ([5], [6], [7], [17]). Other notable topics include global optimization strategies for posterior samples, graph-based retrieval frameworks like TOBUGraph, and explainability solutions for Bayesian Optimization (BO) such as TNTRules ([16], [20], [27]). Lastly, papers also addressed the ongoing debate around optimal decision tree (ODT) methods, crime spatiotemporal prediction models like LGSTime, and the importance of addressing biases in training data for AI models ([18], [30], [

--------------------

# Paper Details

## 1. **[OCEAN RELATED]** Sparse identification of nonlinear dynamics and Koopman operators with   Shallow Recurrent Decoder Networks

*   **Category:** `cs.LG`
*   **Authors:** Mars Liyao Gao, Jan P. Williams, J. Nathan Kutz
*   **Published:** 2025-01-23 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2501.13329v2](http://arxiv.org/abs/2501.13329v2)
*   **PDF Link:** [http://arxiv.org/pdf/2501.13329v2.pdf](http://arxiv.org/pdf/2501.13329v2.pdf)
*   **Summary:** The paper introduces SINDy-SHRED, a method combining Sparse Identification of Nonlinear Dynamics with SHallow REcurrent Decoder networks for modeling complex spatio-temporal data efficiently. The approach learns interpretable generative models from sparse sensor measurements, discovers new physics models, and achieves superior accuracy and efficiency compared to baseline deep learning models in various experiments including turbulent flows and video data.

--------------------

## 2. Scalable Mechanistic Neural Networks for Differential Equations and   Machine Learning

*   **Category:** `cs.LG`
*   **Authors:** Jiale Chen, Dingling Yao, Adeel Pervez, Dan Alistarh, Francesco Locatello
*   **Published:** 2024-10-08 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2410.06074v3](http://arxiv.org/abs/2410.06074v3)
*   **PDF Link:** [http://arxiv.org/pdf/2410.06074v3.pdf](http://arxiv.org/pdf/2410.06074v3.pdf)
*   **Summary:** The Scalable Mechanistic Neural Network (S-MNN) is a refined neural network framework for scientific machine learning with long temporal sequences, improving on the original Mechanistic Neural Network (MNN) by reducing computational complexities to linear from cubic and quadratic, allowing for efficient modeling of long-term dynamics without compromising accuracy. Experimental results show that S-MNN maintains precision while significantly cutting down on computational resources, making it a practical replacement for MNN in applications that integrate mechanistic bottlenecks into neural network models

--------------------

## 3. ActiveGAMER: Active GAussian Mapping through Efficient Rendering

*   **Category:** `cs.CV`
*   **Authors:** Liyan Chen, Huangying Zhan, Kevin Chen, Xiangyu Xu, Qingan Yan, Changjiang Cai, Yi Xu
*   **Published:** 2025-01-12 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2501.06897v2](http://arxiv.org/abs/2501.06897v2)
*   **PDF Link:** [http://arxiv.org/pdf/2501.06897v2.pdf](http://arxiv.org/pdf/2501.06897v2.pdf)
*   **Summary:** ActiveGAMER is introduced as an active mapping system using 3D Gaussian Splatting for real-time scene mapping, offering efficient exploration in complex environments by dynamically selecting informative viewpoints and employing a balanced framework for enhanced reconstruction accuracy and completeness, outperforming existing methods in geometric and photometric reconstruction on benchmark datasets like Replica and MP3D.

--------------------

## 4. NTSEBENCH: Cognitive Reasoning Benchmark for Vision Language Models

*   **Category:** `cs.CV`
*   **Authors:** Pranshu Pandya, Vatsal Gupta, Agney S Talwarr, Tushar Kataria, Dan Roth, Vivek Gupta
*   **Published:** 2024-07-15 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2407.10380v3](http://arxiv.org/abs/2407.10380v3)
*   **PDF Link:** [http://arxiv.org/pdf/2407.10380v3.pdf](http://arxiv.org/pdf/2407.10380v3.pdf)
*   **Summary:** The abstract discusses the challenges of cognitive reasoning tasks and the limitations of large language and vision models (LLMs and VLMs) in more complex reasoning. It introduces a new dataset, NTSEBench, containing multiple-choice questions with images to evaluate multi-modal reasoning skills, intelligence, and critical thinking beyond memorization, and proposes modeling strategies for handling text and image modalities.

--------------------

## 5. Large-Scale Multi-omic Biosequence Transformers for Modeling   Protein-Nucleic Acid Interactions

*   **Category:** `cs.LG`
*   **Authors:** Sully F. Chen, Robert J. Steele, Glen M. Hocky, Beakal Lemeneh, Shivanand P. Lad, Eric K. Oermann
*   **Published:** 2024-08-29 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2408.16245v3](http://arxiv.org/abs/2408.16245v3)
*   **PDF Link:** [http://arxiv.org/pdf/2408.16245v3.pdf](http://arxiv.org/pdf/2408.16245v3.pdf)
*   **Summary:** The paper discusses the development of a large-scale multi-omic foundation model that can efficiently model protein-nucleic acid interactions. These multi-omic models can learn joint representations between different biological domains and achieve state-of-the-art results in predicting Gibbs free energy changes in binding interactions, showcasing their ability to learn structural information without explicit training. The study suggests that multi-omic models outperform single-omic models in various biological tasks, indicating a more comprehensive approach to building biosequence transformers.

--------------------

## 6. Rehearsal-free Federated Domain-incremental Learning

*   **Category:** `cs.LG`
*   **Authors:** Rui Sun, Haoran Duan, Jiahua Dong, Varun Ojha, Tejal Shah, Rajiv Ranjan
*   **Published:** 2024-05-22 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2405.13900v2](http://arxiv.org/abs/2405.13900v2)
*   **PDF Link:** [http://arxiv.org/pdf/2405.13900v2.pdf](http://arxiv.org/pdf/2405.13900v2.pdf)
*   **Summary:** RefFiL is a rehearsal-free federated domain incremental learning framework that tackles catastrophic forgetting in federated learning by sharing prompts globally and generating local prompts to maintain domain-specific knowledge boundaries. It includes a domain-specific prompt contrastive learning loss to enhance precision and effectiveness, making it suitable for resource-constrained devices without compromising privacy.

--------------------

## 7. Identifying Predictions That Influence the Future: Detecting   Performative Concept Drift in Data Streams

*   **Category:** `cs.LG`
*   **Authors:** Brandon Gower-Winter, Georg Krempl, Sergey Dragomiretskiy, Tineke Jelsma, Arno Siebes
*   **Published:** 2024-12-13 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2412.10545v2](http://arxiv.org/abs/2412.10545v2)
*   **PDF Link:** [http://arxiv.org/pdf/2412.10545v2.pdf](http://arxiv.org/pdf/2412.10545v2.pdf)
*   **Summary:** The paper investigates performative drift in stream learning, where the model's predictions can induce concept drift. They introduce a novel drift detection approach, CheckerBoard Performative Drift Detection (CB-PDD), which effectively identifies performative drift in data streams and is resilient to intrinsic drift. The study emphasizes the importance of distinguishing performative drift from other causes of drift and discusses the implications and limitations of CB-PDD.

--------------------

## 8. DetailGen3D: Generative 3D Geometry Enhancement via Data-Dependent Flow

*   **Category:** `cs.CV`
*   **Authors:** Ken Deng, Yuan-Chen Guo, Jingxiang Sun, Zi-Xin Zou, Yangguang Li, Xin Cai, Yan-Pei Cao, Yebin Liu, Ding Liang
*   **Published:** 2024-11-25 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2411.16820v3](http://arxiv.org/abs/2411.16820v3)
*   **PDF Link:** [http://arxiv.org/pdf/2411.16820v3.pdf](http://arxiv.org/pdf/2411.16820v3.pdf)
*   **Summary:** DetailGen3D is a generative approach designed to enhance 3D shapes generated from sparse or single views by modeling coarse-to-fine transformations through data-dependent flows in latent space. It introduces a token matching strategy for accurate spatial correspondence during refinement, enabling local detail synthesis while preserving global structure, and achieves high-fidelity geometric detail synthesis efficiently in training.

--------------------

## 9. STORYSUMM: Evaluating Faithfulness in Story Summarization

*   **Category:** `cs.AI`
*   **Authors:** Melanie Subbiah, Faisal Ladhak, Akankshya Mishra, Griffin Adams, Lydia B. Chilton, Kathleen McKeown
*   **Published:** 2024-07-09 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2407.06501v3](http://arxiv.org/abs/2407.06501v3)
*   **PDF Link:** [http://arxiv.org/pdf/2407.06501v3.pdf](http://arxiv.org/pdf/2407.06501v3.pdf)
*   **Summary:** The study introduces the STORYSUMM dataset for evaluating faithfulness in abstractive summarization of narrative texts, highlighting the limitations of human evaluation and the need for diverse methods to establish ground truth. Automatic metrics tested on this dataset show limited success, indicating a challenging benchmark for future research in faithfulness evaluation.

--------------------

## 10. ASP-based Multi-shot Reasoning via DLV2 with Incremental Grounding

*   **Category:** `cs.AI`
*   **Authors:** Francesco Calimeri, Giovambattista Ianni, Francesco Pacenza, Simona Perri, Jessica Zangari
*   **Published:** 2024-12-22 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2412.17143v4](http://arxiv.org/abs/2412.17143v4)
*   **PDF Link:** [http://arxiv.org/pdf/2412.17143v4.pdf](http://arxiv.org/pdf/2412.17143v4.pdf)
*   **Summary:** DLV2 is an AI tool for Knowledge Representation and Reasoning using Answer Set Programming, supporting logic-based computational problem solving by generating answer sets. The paper introduces an incremental reasoner derived from DLV2 to enable multi-shot reasoning without restarting computations, improving efficiency and applicability in dynamic data environments.

--------------------

## 11. A Survey on Unlearnable Data

*   **Category:** `cs.LG`
*   **Authors:** Jiahao Li, Yiqiang Chen, Yunbing Xing, Yang Gu, Xiangyuan Lan
*   **Published:** 2025-03-30 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23536v2](http://arxiv.org/abs/2503.23536v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23536v2.pdf](http://arxiv.org/pdf/2503.23536v2.pdf)
*   **Summary:** The paper provides a comprehensive review of unlearnable data (ULD), a defense technique that protects data privacy by degrading model performance through perturbations. It evaluates ULD generation methods, benchmarks, metrics, and applications, highlighting challenges like balancing imperceptibility with model degradation and computational complexity, and suggesting future research directions to enhance ULD effectiveness in data protection.

--------------------

## 12. IDArb: Intrinsic Decomposition for Arbitrary Number of Input Views and   Illuminations

*   **Category:** `cs.CV`
*   **Authors:** Zhibing Li, Tong Wu, Jing Tan, Mengchen Zhang, Jiaqi Wang, Dahua Lin
*   **Published:** 2024-12-16 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2412.12083v3](http://arxiv.org/abs/2412.12083v3)
*   **PDF Link:** [http://arxiv.org/pdf/2412.12083v3.pdf](http://arxiv.org/pdf/2412.12083v3.pdf)
*   **Summary:** This paper introduces IDArb, a diffusion-based model for intrinsic decomposition of geometric and material properties from images under varying illuminations. The method achieves accurate estimation of surface normals and material properties across multiple views, outperforming existing methods and supporting various downstream tasks in 3D content creation.

--------------------

## 13. FlowSeries: Anomaly Detection in Financial Transaction Flows

*   **Category:** `cs.CY`
*   **Authors:** Arthur Capozzi, Salvatore Vilella, Dario Moncalvo, Marco Fornasiero, Valeria Ricci, Silvia Ronchiadin, Giancarlo Ruffo
*   **Published:** 2025-03-20 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.15896v2](http://arxiv.org/abs/2503.15896v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.15896v2.pdf](http://arxiv.org/pdf/2503.15896v2.pdf)
*   **Summary:** The paper introduces WeirdFlows, a network analysis-based pipeline for detecting potentially fraudulent transactions and non-compliant agents in AFC investigations. It does not require predefined patterns or a training set, offering interpretability for AFC analysts. Evaluation on an ISP bank dataset shows its effectiveness in identifying suspicious activities, especially in the context of EU economic sanctions post-February 2022.

--------------------

## 14. NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model   Internals

*   **Category:** `cs.LG`
*   **Authors:** Jaden Fiotto-Kaufman, Alexander R. Loftus, Eric Todd, Jannik Brinkmann, Koyena Pal, Dmitrii Troitskii, Michael Ripa, Adam Belfki, Can Rager, Caden Juang, Aaron Mueller, Samuel Marks, Arnab Sen Sharma, Francesca Lucchetti, Nikhil Prakash, Carla Brodley, Arjun Guha, Jonathan Bell, Byron C. Wallace, David Bau
*   **Published:** 2024-07-18 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2407.14561v4](http://arxiv.org/abs/2407.14561v4)
*   **PDF Link:** [http://arxiv.org/pdf/2407.14561v4.pdf](http://arxiv.org/pdf/2407.14561v4.pdf)
*   **Summary:** NNsight and NDIF are technologies designed to facilitate the study of large neural networks by introducing deferred remote execution and a scalable inference service, respectively. These tools, along with the Intervention Graph architecture, allow for transparent access to deep neural network internals, enabling various research methods on very large language models without the complexity of hosting customized models individually. Performance benchmarks and resources are available at https://nnsight.net/.

--------------------

## 15. Oriented Object Detection in Optical Remote Sensing Images using Deep   Learning: A Survey

*   **Category:** `cs.CV`
*   **Authors:** Kun Wang, Zi Wang, Zhang Li, Ang Su, Xichao Teng, Erting Pan, Minhao Liu, Qifeng Yu
*   **Published:** 2023-02-21 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2302.10473v5](http://arxiv.org/abs/2302.10473v5)
*   **PDF Link:** [http://arxiv.org/pdf/2302.10473v5.pdf](http://arxiv.org/pdf/2302.10473v5.pdf)
*   **Summary:** This paper provides a detailed survey of recent advancements in oriented object detection in remote sensing, discussing the evolution from horizontal to oriented object detection, challenges such as feature and spatial misalignment, OBB regression issues, categorization of methods, dataset availability, evaluation protocols, comparison of state-of-the-art methods, and future research directions in the field.

--------------------

## 16. Optimizing Posterior Samples for Bayesian Optimization via Rootfinding

*   **Category:** `cs.LG`
*   **Authors:** Taiwo A. Adebiyi, Bach Do, Ruda Zhang
*   **Published:** 2024-10-29 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2410.22322v4](http://arxiv.org/abs/2410.22322v4)
*   **PDF Link:** [http://arxiv.org/pdf/2410.22322v4.pdf](http://arxiv.org/pdf/2410.22322v4.pdf)
*   **Summary:** The paper introduces an efficient global optimization strategy for posterior samples using global rootfinding, allowing gradient-based optimizers to find the global optimum with just one point from each of two sets of starting points most of the time, even in high dimensions. This approach significantly improves the performance of Gaussian process Thompson sampling (GP-TS) and other posterior sample-based acquisition functions, such as entropy search variants, and offers a sample-average formulation of GP-TS with explicit exploitation control.

--------------------

## 17. Mixture of Experts Made Personalized: Federated Prompt Learning for   Vision-Language Models

*   **Category:** `cs.LG`
*   **Authors:** Jun Luo, Chen Chen, Shandong Wu
*   **Published:** 2024-10-14 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2410.10114v4](http://arxiv.org/abs/2410.10114v4)
*   **PDF Link:** [http://arxiv.org/pdf/2410.10114v4.pdf](http://arxiv.org/pdf/2410.10114v4.pdf)
*   **Summary:** The paper introduces pFedMoAP, a novel Federated Learning (FL) framework that allows clients to download multiple pre-aggregated prompts as non-local experts, enhancing prompt learning through a Mixture of Experts (MoE) approach. Experimental results across 9 datasets validate the efficacy of pFedMoAP in improving text features alignment with local image data in federated settings. The code for pFedMoAP is accessible at https://github.com/ljaiverson/pFedMoAP.

--------------------

## 18. Optimal or Greedy Decision Trees? Revisiting their Objectives, Tuning,   and Performance

*   **Category:** `cs.LG`
*   **Authors:** Jacobus G. M. van der Linden, Daniël Vos, Mathijs M. de Weerdt, Sicco Verwer, Emir Demirović
*   **Published:** 2024-09-19 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2409.12788v2](http://arxiv.org/abs/2409.12788v2)
*   **PDF Link:** [http://arxiv.org/pdf/2409.12788v2.pdf](http://arxiv.org/pdf/2409.12788v2.pdf)
*   **Summary:** The abstract discusses the ongoing debate regarding the performance of optimal decision tree (ODT) methods compared to traditional greedy approaches. The study conducts an extensive experimental analysis on 180 datasets to explore the impact of different objective functions and tuning techniques on ODTs, providing insights and recommendations for researchers and practitioners.

--------------------

## 19. HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model

*   **Category:** `cs.CV`
*   **Authors:** Di Wang, Meiqi Hu, Yao Jin, Yuchun Miao, Jiaqi Yang, Yichu Xu, Xiaolei Qin, Jiaqi Ma, Lingyu Sun, Chenxing Li, Chuan Fu, Hongruixuan Chen, Chengxi Han, Naoto Yokoya, Jing Zhang, Minqiang Xu, Lin Liu, Lefei Zhang, Chen Wu, Bo Du, Dacheng Tao, Liangpei Zhang
*   **Published:** 2024-06-17 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2406.11519v2](http://arxiv.org/abs/2406.11519v2)
*   **PDF Link:** [http://arxiv.org/pdf/2406.11519v2.pdf](http://arxiv.org/pdf/2406.11519v2.pdf)
*   **Summary:** The paper introduces HyperSIGMA, a vision transformer-based model for hyperspectral image interpretation that unifies tasks and scenes, overcoming limitations of existing methods. It incorporates a sparse sampling attention mechanism to address spectral and spatial redundancy, integrates spatial and spectral features, and utilizes a large-scale dataset for pre-training. HyperSIGMA demonstrates superior performance in various HSI tasks, scalability, robustness, cross-modal transferring capability, real-world applicability, and computational efficiency compared to current state-of-the-art methods. The

--------------------

## 20. Explainable Bayesian Optimization

*   **Category:** `cs.LG`
*   **Authors:** Tanmay Chakraborty, Christian Wirth, Christin Seifert
*   **Published:** 2024-01-24 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2401.13334v2](http://arxiv.org/abs/2401.13334v2)
*   **PDF Link:** [http://arxiv.org/pdf/2401.13334v2.pdf](http://arxiv.org/pdf/2401.13334v2.pdf)
*   **Summary:** This paper introduces TNTRules, a novel algorithm designed to address the post-hoc explainability problem of Bayesian Optimization (BO) for cyber-physical systems. TNTRules provides both global and local explanations for BO recommendations through actionable rules and visual graphs, outperforming three baselines on various testing functions and hyperparameter tuning problems in terms of explanation quality.

--------------------

## 21. BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games

*   **Category:** `cs.AI`
*   **Authors:** Davide Paglieri, Bartłomiej Cupiał, Samuel Coward, Ulyana Piterbarg, Maciej Wolczyk, Akbir Khan, Eduardo Pignatelli, Łukasz Kuciński, Lerrel Pinto, Rob Fergus, Jakob Nicolaus Foerster, Jack Parker-Holder, Tim Rocktäschel
*   **Published:** 2024-11-20 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2411.13543v2](http://arxiv.org/abs/2411.13543v2)
*   **PDF Link:** [http://arxiv.org/pdf/2411.13543v2.pdf](http://arxiv.org/pdf/2411.13543v2.pdf)
*   **Summary:** BALROG is a new benchmark to evaluate the agentic capabilities of Large Language Models (LLMs) and Vision Language Models (VLMs) through challenging games, including tasks ranging from easily solvable to extremely difficult ones like the NetHack Learning Environment. Current models show partial success in simpler games but struggle with more complex tasks, particularly in vision-based decision-making. The benchmark aims to support future research and development in the agentic community and is available at balrogai.com.

--------------------

## 22. Mind the GAP: Glimpse-based Active Perception improves generalization   and sample efficiency of visual reasoning

*   **Category:** `cs.CV`
*   **Authors:** Oleh Kolner, Thomas Ortner, Stanisław Woźniak, Angeliki Pantazi
*   **Published:** 2024-09-30 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2409.20213v2](http://arxiv.org/abs/2409.20213v2)
*   **PDF Link:** [http://arxiv.org/pdf/2409.20213v2.pdf](http://arxiv.org/pdf/2409.20213v2.pdf)
*   **Summary:** Human visual understanding surpasses that of AI systems, especially with novel objects, due to active vision theories suggesting learning visual relations through eye movements. A novel Glimpse-based Active Perception (GAP) system is developed, outperforming prior models on visual reasoning tasks by leveraging eye movement locations and visual content to represent relations in images effectively.

--------------------

## 23. RedMotion: Motion Prediction via Redundancy Reduction

*   **Category:** `cs.CV`
*   **Authors:** Royden Wagner, Omer Sahin Tas, Marvin Klemp, Carlos Fernandez, Christoph Stiller
*   **Published:** 2023-06-19 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2306.10840v4](http://arxiv.org/abs/2306.10840v4)
*   **PDF Link:** [http://arxiv.org/pdf/2306.10840v4.pdf](http://arxiv.org/pdf/2306.10840v4.pdf)
*   **Summary:** RedMotion is a transformer model for motion prediction in self-driving vehicles that utilizes redundancy reduction to learn environment representations effectively. Through internal transformer decoders and self-supervised learning, it outperforms existing methods like PreTraM, Traj-MAE, and GraphDINO in semi-supervised settings and achieves competitive results in the Waymo Motion Prediction Challenge. The open-source implementation can be found at: https://github.com/kit-mrt/future-motion.

--------------------

## 24. Knowledge-Aware Iterative Retrieval for Multi-Agent Systems

*   **Category:** `cs.AI`
*   **Authors:** Seyoung Song
*   **Published:** 2025-03-17 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.13275v2](http://arxiv.org/abs/2503.13275v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.13275v2.pdf](http://arxiv.org/pdf/2503.13275v2.pdf)
*   **Summary:** The abstract introduces a novel large language model-driven agent framework that refines queries and filters evidence using evolving knowledge, decoupling external sources from an internal cache to mitigate bias and enable dynamic search paths. The system outperforms single-step baselines and iterative retrieval methods in diverse question-answering tasks, showcasing advantages in complex scenarios through evidence-based reasoning and efficiency, supporting multi-agent collaboration for enhanced performance as task difficulty rises.

--------------------

## 25. The Computational Complexity of Circuit Discovery for Inner   Interpretability

*   **Category:** `cs.AI`
*   **Authors:** Federico Adolfi, Martina G. Vilas, Todd Wareham
*   **Published:** 2024-10-10 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2410.08025v3](http://arxiv.org/abs/2410.08025v3)
*   **PDF Link:** [http://arxiv.org/pdf/2410.08025v3.pdf](http://arxiv.org/pdf/2410.08025v3.pdf)
*   **Summary:** The abstract discusses the importance of inner interpretability in neural networks and proposes a framework using computational complexity theory to analyze circuit discovery queries for mechanistic explanation. The study reveals a challenging complexity landscape with many intractable queries, but also identifies transformations to tackle some hard problems and proves the tractability of more modest queries.

--------------------

## 26. Fine-Grained Behavior and Lane Constraints Guided Trajectory Prediction   Method

*   **Category:** `cs.CV`
*   **Authors:** Wenyi Xiong, Jian Chen, Ziheng Qi
*   **Published:** 2025-03-27 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.21477v2](http://arxiv.org/abs/2503.21477v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.21477v2.pdf](http://arxiv.org/pdf/2503.21477v2.pdf)
*   **Summary:** The paper introduces BLNet, a dual-stream architecture for trajectory prediction in autonomous driving systems, which combines behavioral intention recognition and lane constraint modeling using parallel attention mechanisms. The model generates fine-grained behavior state and lane queries, followed by a two-stage decoder for trajectory proposals and refinement, outperforming existing algorithms on nuScenes and Argoverse datasets.

--------------------

## 27. TOBUGraph: Knowledge Graph-Based Retrieval for Enhanced LLM Performance   Beyond RAG

*   **Category:** `cs.LG`
*   **Authors:** Savini Kashmira, Jayanaka L. Dantanarayana, Joshua Brodsky, Ashish Mahendra, Yiping Kang, Krisztian Flautner, Lingjia Tang, Jason Mars
*   **Published:** 2024-12-06 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2412.05447v2](http://arxiv.org/abs/2412.05447v2)
*   **PDF Link:** [http://arxiv.org/pdf/2412.05447v2.pdf](http://arxiv.org/pdf/2412.05447v2.pdf)
*   **Summary:** TOBUGraph is proposed as a graph-based retrieval framework that leverages knowledge graphs constructed from unstructured data to enhance retrieval accuracy beyond the limitations of Retrieval-Augmented Generation (RAG). By utilizing structured knowledge and relationships, TOBUGraph eliminates the need for chunking strategies, reduces hallucinations, and improves precision and recall compared to multiple RAG implementations in a real-world application for personal memory organization and retrieval called TOBU.

--------------------

## 28. RePoseD: Efficient Relative Pose Estimation With Known Depth Information

*   **Category:** `cs.CV`
*   **Authors:** Yaqing Ding, Viktor Kocur, Václav Vávra, Jian Yang, Torsten Sattler, Zuzana Kukelova
*   **Published:** 2025-01-13 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2501.07742v2](http://arxiv.org/abs/2501.07742v2)
*   **PDF Link:** [http://arxiv.org/pdf/2501.07742v2.pdf](http://arxiv.org/pdf/2501.07742v2.pdf)
*   **Summary:** This paper explores the use of monocular depth estimates for relative pose estimation between two cameras, comparing them to traditional point-based methods. The proposed framework includes efficient solvers for different camera configurations, achieving superior speed and accuracy compared to existing depth-aware solvers, as demonstrated through real experiments on multiple datasets with various monocular depth estimation methods.

--------------------

## 29. Enhanced Controllability of Diffusion Models via Feature Disentanglement   and Realism-Enhanced Sampling Methods

*   **Category:** `cs.CV`
*   **Authors:** Wonwoong Cho, Hareesh Ravi, Midhun Harikumar, Vinh Khuc, Krishna Kumar Singh, Jingwan Lu, David I. Inouye, Ajinkya Kale
*   **Published:** 2023-02-28 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2302.14368v4](http://arxiv.org/abs/2302.14368v4)
*   **PDF Link:** [http://arxiv.org/pdf/2302.14368v4.pdf](http://arxiv.org/pdf/2302.14368v4.pdf)
*   **Summary:** This paper introduces a training framework called FDiff for feature disentanglement of Diffusion Models, focusing on training models with disentangled latent spaces and incorporating disentangled conditions during sampling. The proposed sampling methods, including Composable Diffusion Models and timestep-dependent weight scheduling, enhance realism and controllability in image generation, manipulation, and translation compared to existing methods.

--------------------

## 30. Innovative LSGTime Model for Crime Spatiotemporal Prediction Based on   MindSpore Framework

*   **Category:** `cs.LG`
*   **Authors:** Zhenkai Qin, BaoZhong Wei, Caifeng Gao
*   **Published:** 2025-03-26 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.20136v3](http://arxiv.org/abs/2503.20136v3)
*   **PDF Link:** [http://arxiv.org/pdf/2503.20136v3.pdf](http://arxiv.org/pdf/2503.20136v3.pdf)
*   **Summary:** This paper introduces LGSTime, a crime spatiotemporal prediction model combining LSTM, GRU, and Multi-head Sparse Self-attention mechanisms to accurately predict crime distribution. The model effectively captures long-term dependencies and temporal-spatial features, demonstrating superior performance compared to a CNN model across various crime datasets.

--------------------

## 31. Att-Adapter: A Robust and Precise Domain-Specific Multi-Attributes T2I   Diffusion Adapter via Conditional Variational Autoencoder

*   **Category:** `cs.CV`
*   **Authors:** Wonwoong Cho, Yan-Ying Chen, Matthew Klenk, David I. Inouye, Yanxia Zhang
*   **Published:** 2025-03-15 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.11937v2](http://arxiv.org/abs/2503.11937v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.11937v2.pdf](http://arxiv.org/pdf/2503.11937v2.pdf)
*   **Summary:** The paper introduces the Attribute Adapter (Att-Adapter), a plug-and-play module designed to enable precise control of multiple continuous attributes in pretrained diffusion models using text guidance. The Att-Adapter leverages a decoupled cross attention module and a Conditional Variational Autoencoder (CVAE) to enhance control and mitigate overfitting, outperforming existing baselines in controlling attributes and improving disentanglement in image generation tasks without requiring paired synthetic data for training.

--------------------

## 32. Improving Vector-Quantized Image Modeling with Latent   Consistency-Matching Diffusion

*   **Category:** `cs.LG`
*   **Authors:** Bac Nguyen, Chieh-Hsin Lai, Yuhta Takida, Naoki Murata, Toshimitsu Uesaka, Stefano Ermon, Yuki Mitsufuji
*   **Published:** 2024-10-18 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2410.14758v2](http://arxiv.org/abs/2410.14758v2)
*   **PDF Link:** [http://arxiv.org/pdf/2410.14758v2.pdf](http://arxiv.org/pdf/2410.14758v2.pdf)
*   **Summary:** The paper introduces VQ-LCMD, a continuous-space latent diffusion framework that stabilizes training by combining a joint embedding-diffusion variational lower bound with a consistency-matching loss, alongside a shifted cosine noise schedule and random dropping strategy. Experimental results demonstrate that VQ-LCMD outperforms discrete-state latent diffusion models on various benchmarks, achieving an FID of 6.81 for class-conditional image generation on ImageNet with 50 steps.

--------------------

## 33. Statistically Testing Training Data for Unwanted Error Patterns using   Rule-Oriented Regression

*   **Category:** `cs.LG`
*   **Authors:** Stefan Rass, Martin Dallinger
*   **Published:** 2025-03-24 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.18497v2](http://arxiv.org/abs/2503.18497v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.18497v2.pdf](http://arxiv.org/pdf/2503.18497v2.pdf)
*   **Summary:** The abstract discusses the importance of addressing biases in training data for artificial intelligence models and proposes a method to test data for flaws before training machine learning models, aiming to establish a trustworthy ground-truth. The method combines fuzzy inference with regression modeling to detect hidden error patterns and can be applied to small datasets without requiring large amounts of data like deep learning methods.

--------------------

## 34. MSCMNet: Multi-scale Semantic Correlation Mining for Visible-Infrared   Person Re-Identification

*   **Category:** `cs.LG`
*   **Authors:** Xuecheng Hua, Ke Cheng, Hu Lu, Juanjuan Tu, Yuanquan Wang, Shitong Wang
*   **Published:** 2023-11-24 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2311.14395v2](http://arxiv.org/abs/2311.14395v2)
*   **PDF Link:** [http://arxiv.org/pdf/2311.14395v2.pdf](http://arxiv.org/pdf/2311.14395v2.pdf)
*   **Summary:** The abstract introduces the Multi-scale Semantic Correlation Mining network (MSCMNet) for Visible-Infrared Person Re-Identification (VI-ReID), aiming to extract discriminative features from different modalities by leveraging semantic correlations at multiple scales and minimizing modality information loss. The network includes a Multi-scale Information Correlation Mining Block (MIMB), a quadruple-stream feature extractor (QFE), and a Quadruple Center Triplet Loss (QCT), resulting in improved accuracy on various datasets.

--------------------

## 35. Class-Dependent Perturbation Effects in Evaluating Time Series   Attributions

*   **Category:** `cs.LG`
*   **Authors:** Gregor Baer, Isel Grau, Chao Zhang, Pieter Van Gorp
*   **Published:** 2025-02-24 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2502.17022v2](http://arxiv.org/abs/2502.17022v2)
*   **PDF Link:** [http://arxiv.org/pdf/2502.17022v2.pdf](http://arxiv.org/pdf/2502.17022v2.pdf)
*   **Summary:** The abstract discusses the importance of Explainable Artificial Intelligence (XAI) methods in understanding predictions of machine learning models in time series applications. The study reveals class-dependent effects in feature attribution methods, indicating varying effectiveness across classes and proposing an evaluation framework with a class-aware penalty term to address these effects, especially beneficial for class-imbalanced datasets in time series classification and potentially other structured data domains.

--------------------

## 36. Efficient Semantic Segmentation via Lightweight Multiple-Information   Interaction Network

*   **Category:** `cs.CV`
*   **Authors:** Yangyang Qiu, Guoan Xu, Guangwei Gao, Zhenhua Guo, Yi Yu, Chia-Wen Lin
*   **Published:** 2024-10-03 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2410.02224v2](http://arxiv.org/abs/2410.02224v2)
*   **PDF Link:** [http://arxiv.org/pdf/2410.02224v2.pdf](http://arxiv.org/pdf/2410.02224v2.pdf)
*   **Summary:** The paper introduces LMIINet, a Lightweight Multiple-Information Interaction Network for real-time semantic segmentation that combines CNNs and Transformers efficiently to balance accuracy and efficiency. LMIINet achieves 72.0% mIoU at 100 FPS on Cityscapes and 69.94% mIoU at 160 FPS on CamVid with only 0.72M parameters and 11.74G FLOPs on a single RTX2080Ti GPU, addressing the

--------------------

## 37. Optimization Insights into Deep Diagonal Linear Networks

*   **Category:** `cs.LG`
*   **Authors:** Hippolyte Labarrière, Cesare Molinari, Lorenzo Rosasco, Silvia Villa, Cristian Vega
*   **Published:** 2024-12-21 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2412.16765v2](http://arxiv.org/abs/2412.16765v2)
*   **PDF Link:** [http://arxiv.org/pdf/2412.16765v2.pdf](http://arxiv.org/pdf/2412.16765v2.pdf)
*   **Summary:** This paper explores the implicit regularization properties of gradient flow for training deep diagonal neural networks, revealing a bias towards specific solutions determined by network initialization, with findings shedding light on the optimization dynamics and properties of overparameterized models trained with gradient descent.

--------------------

## 38. Towards Robust and Parameter-Efficient Knowledge Unlearning for LLMs

*   **Category:** `cs.LG`
*   **Authors:** Sungmin Cha, Sungjun Cho, Dasol Hwang, Moontae Lee
*   **Published:** 2024-08-13 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2408.06621v4](http://arxiv.org/abs/2408.06621v4)
*   **PDF Link:** [http://arxiv.org/pdf/2408.06621v4.pdf](http://arxiv.org/pdf/2408.06621v4.pdf)
*   **Summary:** The abstract introduces the challenge of efficiently unlearning sensitive data from Large Language Models (LLMs) to address privacy and copyright concerns. The proposed Low-rank Knowledge Unlearning (LoKU) framework combines Gradient Ascent with low-rank adaptation and introduces Inverted Hinge Loss and data-adaptive initialization for effective unlearning without sacrificing generative performance, as demonstrated through experiments on GPT-Neo models. The implementation of LoKU can be accessed at https://github.com/csm9493

--------------------

## 39. Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards   for Reasoning-Enhanced Text-to-SQL

*   **Category:** `cs.LG`
*   **Authors:** Mohammadreza Pourreza, Shayan Talaei, Ruoxi Sun, Xingchen Wan, Hailong Li, Azalia Mirhoseini, Amin Saberi, Sercan "O. Arik
*   **Published:** 2025-03-29 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23157v2](http://arxiv.org/abs/2503.23157v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23157v2.pdf](http://arxiv.org/pdf/2503.23157v2.pdf)
*   **Summary:** The paper introduces a novel set of partial rewards tailored for Text-to-SQL tasks, addressing reward sparsity in reinforcement learning. By leveraging group relative policy optimization (GRPO), the proposed approach encourages large language models to develop intrinsic reasoning skills, resulting in higher accuracy and superior generalization compared to supervised fine-tuning. The RL-trained 14B-parameter model outperforms larger proprietary models on the BIRD benchmark, demonstrating the effectiveness of the proposed RL-training framework for enhancing accuracy and reasoning capabilities in Text

--------------------

## 40. Generative Data Assimilation of Sparse Weather Station Observations at   Kilometer Scales

*   **Category:** `cs.LG`
*   **Authors:** Peter Manshausen, Yair Cohen, Peter Harrington, Jaideep Pathak, Mike Pritchard, Piyush Garg, Morteza Mardani, Karthik Kashinath, Simon Byrne, Noah Brenowitz
*   **Published:** 2024-06-19 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2406.16947v3](http://arxiv.org/abs/2406.16947v3)
*   **PDF Link:** [http://arxiv.org/pdf/2406.16947v3.pdf](http://arxiv.org/pdf/2406.16947v3.pdf)
*   **Summary:** The study demonstrates the effectiveness of score-based data assimilation in generating realistic km-scale weather fields by training a diffusion model on High Resolution Rapid Refresh data and incorporating sparse weather station observations. The approach shows improved performance compared to a baseline system, with 10% lower RMSEs on left-out stations, indicating its potential for accelerating data assimilation processes in regional weather models.

--------------------

## 41. ConsistencyDet: A Few-step Denoising Framework for Object Detection   Using the Consistency Model

*   **Category:** `cs.CV`
*   **Authors:** Lifan Jiang, Zhihui Wang, Changmiao Wang, Ming Li, Jiaxu Leng, Xindong Wu
*   **Published:** 2024-04-11 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2404.07773v4](http://arxiv.org/abs/2404.07773v4)
*   **PDF Link:** [http://arxiv.org/pdf/2404.07773v4.pdf](http://arxiv.org/pdf/2404.07773v4.pdf)
*   **Summary:** The study introduces ConsistencyDet, a novel framework for object detection using a denoising diffusion process with a self-consistency feature that enhances operational efficiency by mapping distorted information back to its original state. ConsistencyDet outperforms other detectors on benchmarks like MS-COCO and LVIS, with code available at the provided link.

--------------------

## 42. SVInvNet: A Densely Connected Encoder-Decoder Architecture for Seismic   Velocity Inversion

*   **Category:** `cs.LG`
*   **Authors:** Mojtaba Najafi Khatounabad, Hacer Yalim Keles, Selma Kadioglu
*   **Published:** 2023-12-13 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2312.08194v2](http://arxiv.org/abs/2312.08194v2)
*   **PDF Link:** [http://arxiv.org/pdf/2312.08194v2.pdf](http://arxiv.org/pdf/2312.08194v2.pdf)
*   **Summary:** This study introduces SVInvNet, a deep learning model for seismic velocity inversion, designed to handle noisy and noiseless datasets of different sizes. SVInvNet's unique architecture, tailored for time series data processing, outperforms baseline models across various seismic velocity models and noise types, demonstrating its effectiveness in seismic inversion tasks.

--------------------

## 43. Self-Supervised Pretraining for Aerial Road Extraction

*   **Category:** `cs.CV`
*   **Authors:** Rupert Polley, Sai Vignesh Abishek Deenadayalan, J. Marius Zöllner
*   **Published:** 2025-03-31 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.24326v2](http://arxiv.org/abs/2503.24326v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.24326v2.pdf](http://arxiv.org/pdf/2503.24326v2.pdf)
*   **Summary:** A self-supervised pretraining method is proposed to improve aerial image segmentation by training models to reconstruct missing regions in images before fine-tuning for road extraction. This approach enhances segmentation accuracy, generalization, and robustness to domain shifts, especially in low-data scenarios, offering a scalable solution for aerial image analysis.

--------------------

## 44. Sharp Rates in Dependent Learning Theory: Avoiding Sample Size Deflation   for the Square Loss

*   **Category:** `cs.LG`
*   **Authors:** Ingvar Ziemann, Stephen Tu, George J. Pappas, Nikolai Matni
*   **Published:** 2024-02-08 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2402.05928v4](http://arxiv.org/abs/2402.05928v4)
*   **PDF Link:** [http://arxiv.org/pdf/2402.05928v4.pdf](http://arxiv.org/pdf/2402.05928v4.pdf)
*   **Summary:** The study focuses on statistical learning with dependent data and square loss in a hypothesis class $\mathscr{F}$, investigating the impact of the mixing time of the underlying covariates process on variance proxies in the absence of realizability assumptions. The research establishes that under certain conditions, the empirical risk minimizer achieves a rate primarily determined by the class complexity and second order statistics, presenting a "near mixing-free rate" where direct dependence on mixing is only reflected in a higher order term.

--------------------

## 45. DG-TTA: Out-of-domain Medical Image Segmentation through Augmentation   and Descriptor-driven Domain Generalization and Test-Time Adaptation

*   **Category:** `cs.CV`
*   **Authors:** Christian Weihsbach, Christian N. Kruse, Alexander Bigalke, Mattias P. Heinrich
*   **Published:** 2023-12-11 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2312.06275v4](http://arxiv.org/abs/2312.06275v4)
*   **PDF Link:** [http://arxiv.org/pdf/2312.06275v4.pdf](http://arxiv.org/pdf/2312.06275v4.pdf)
*   **Summary:** The study proposes a method using a generalizing descriptor and augmentation for domain-generalized pre-training and test-time adaptation of deep learning segmentation models, achieving high-quality segmentation in unseen domains. Results show significant improvements in cross-domain prediction for abdominal, spine, and cardiac imaging scenarios.

--------------------

## 46. Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation

*   **Category:** `cs.CV`
*   **Authors:** Chuandong Liu, Xingxing Weng, Shuguo Jiang, Pengcheng Li, Lei Yu, Gui-Song Xia
*   **Published:** 2024-08-21 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2408.11280v3](http://arxiv.org/abs/2408.11280v3)
*   **PDF Link:** [http://arxiv.org/pdf/2408.11280v3.pdf](http://arxiv.org/pdf/2408.11280v3.pdf)
*   **Summary:** This paper introduces AIScene, a method for semi-supervised LiDAR semantic segmentation in driving scenes that utilizes a teacher-student training approach. By removing points without pseudo-labels during training and incorporating patch-based data augmentation, AIScene effectively improves segmentation model performance, outperforming previous methods on popular benchmarks with notable improvements in challenging low-data scenarios.

--------------------

## 47. Illuminating the Diversity-Fitness Trade-Off in Black-Box Optimization

*   **Category:** `cs.LG`
*   **Authors:** Maria Laura Santoni, Elena Raponi, Aneta Neumann, Frank Neumann, Mike Preuss, Carola Doerr
*   **Published:** 2024-08-29 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2408.16393v2](http://arxiv.org/abs/2408.16393v2)
*   **PDF Link:** [http://arxiv.org/pdf/2408.16393v2.pdf](http://arxiv.org/pdf/2408.16393v2.pdf)
*   **Summary:** This paper explores the challenge of selecting a fixed number of solutions with a specified pairwise distance threshold while maximizing their average quality, focusing on existing search heuristics' ability to balance diversity and quality. The study reveals that simple random sampling often performs as well as specialized heuristics in this context, suggesting a need for algorithms that can generate diverse high-quality solutions.

--------------------

## 48. Introducing the Short-Time Fourier Kolmogorov Arnold Network: A Dynamic   Graph CNN Approach for Tree Species Classification in 3D Point Clouds

*   **Category:** `cs.CV`
*   **Authors:** Said Ohamouddou, Mohamed Ohamouddou, Hanaa El Afia, Abdellatif El Afia, Rafik Lasri, Raddouane Chiheb
*   **Published:** 2025-03-31 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23647v2](http://arxiv.org/abs/2503.23647v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23647v2.pdf](http://arxiv.org/pdf/2503.23647v2.pdf)
*   **Summary:** The paper introduces STFT-KAN, a novel network integrating the Short-Time Fourier Transform within a lightweight DGCNN model to classify tree species using TLS data, achieving competitive results with reduced parameter count compared to existing models. STFT-KAN also outperforms other KAN variants and a hybrid architecture combining MLP and STFT-KAN shows promising results in balancing model complexity and performance.

--------------------

## 49. Machine Unlearning Fails to Remove Data Poisoning Attacks

*   **Category:** `cs.LG`
*   **Authors:** Martin Pawelczyk, Jimmy Z. Di, Yiwei Lu, Ayush Sekhari, Gautam Kamath, Seth Neel
*   **Published:** 2024-06-25 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2406.17216v2](http://arxiv.org/abs/2406.17216v2)
*   **PDF Link:** [http://arxiv.org/pdf/2406.17216v2.pdf](http://arxiv.org/pdf/2406.17216v2.pdf)
*   **Summary:** The study evaluates practical methods for approximate machine unlearning in deep learning, finding that existing methods struggle to effectively remove the effects of data poisoning attacks across various scenarios and models. New evaluation metrics are introduced to assess unlearning efficacy, highlighting the need for a more comprehensive approach to avoid overconfidence in unlearning procedures for deep learning without guaranteed results. Although unlearning methods may offer some efficiency in removing poisoned data without retraining, the study suggests that they are not yet as effective as retraining.

--------------------

## 50. Causal Concept Graph Models: Beyond Causal Opacity in Deep Learning

*   **Category:** `cs.LG`
*   **Authors:** Gabriele Dominici, Pietro Barbiero, Mateo Espinosa Zarlenga, Alberto Termine, Martin Gjoreski, Giuseppe Marra, Marc Langheinrich
*   **Published:** 2024-05-26 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2405.16507v6](http://arxiv.org/abs/2405.16507v6)
*   **PDF Link:** [http://arxiv.org/pdf/2405.16507v6.pdf](http://arxiv.org/pdf/2405.16507v6.pdf)
*   **Summary:** The paper addresses the challenge of causal opacity in deep neural networks by introducing Causal Concept Graph Models (Causal CGMs), which are interpretable models with a transparent decision-making process. The experiments demonstrate that Causal CGMs can match the performance of opaque models, allow human corrections to reasoning steps, and improve causal interpretability and model verification.

--------------------

## 51. Leveraging Joint Predictive Embedding and Bayesian Inference in Graph   Self Supervised Learning

*   **Category:** `cs.LG`
*   **Authors:** Srinitish Srinivasan, Omkumar CU
*   **Published:** 2025-02-02 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2502.01684v3](http://arxiv.org/abs/2502.01684v3)
*   **PDF Link:** [http://arxiv.org/pdf/2502.01684v3.pdf](http://arxiv.org/pdf/2502.01684v3.pdf)
*   **Summary:** The paper introduces a novel joint embedding predictive framework for graph self-supervised learning (SSL) that eliminates the need for contrastive objectives and negative sampling while maintaining semantic and structural information. By incorporating a semantic-aware objective term utilizing pseudo-labels from Gaussian Mixture Models (GMMs), the proposed framework enhances node discriminability and outperforms existing methods in graph SSL tasks without complex decoders. The approach offers a computationally efficient and collapse-resistant paradigm that integrates spatial and semantic graph features for improved performance

--------------------

## 52. ExMAG: Learning of Maximally Ancestral Graphs

*   **Category:** `cs.LG`
*   **Authors:** Petr Ryšavý, Pavel Rytíř, Xiaoyu He, Georgios Korpas, Jakub Mareček
*   **Published:** 2025-03-11 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.08245v2](http://arxiv.org/abs/2503.08245v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.08245v2.pdf](http://arxiv.org/pdf/2503.08245v2.pdf)
*   **Summary:** The paper introduces a score-based learning algorithm for maximizing ancestral graphs with confounders, using a mixed-integer quadratic program and a branch-and-cut method to avoid generating exponentially many constraints. The proposed approach demonstrates improved accuracy compared to existing methods on small to medium-sized synthetic datasets with up to 25 variables.

--------------------

## 53. DoubleDiffusion: Combining Heat Diffusion with Denoising Diffusion for   Texture Generation on 3D Meshes

*   **Category:** `cs.CV`
*   **Authors:** Xuyang Wang, Ziang Cheng, Zhenyu Li, Jiayu Yang, Haorui Ji, Pan Ji, Mehrtash Harandi, Richard Hartley, Hongdong Li
*   **Published:** 2025-01-06 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2501.03397v5](http://arxiv.org/abs/2501.03397v5)
*   **PDF Link:** [http://arxiv.org/pdf/2501.03397v5.pdf](http://arxiv.org/pdf/2501.03397v5.pdf)
*   **Summary:** This paper introduces DoubleDiffusion, a novel method for generating textures directly on 3D meshes by leveraging heat dissipation diffusion, which improves efficiency compared to existing approaches that rely on image diffusion models. By combining heat dissipation diffusion with denoising diffusion, this approach enables generative learning on 3D mesh surfaces.

--------------------

## 54. Attention-Guided Multi-scale Interaction Network for Face   Super-Resolution

*   **Category:** `cs.CV`
*   **Authors:** Xujie Wan, Wenjie Li, Guangwei Gao, Huimin Lu, Jian Yang, Chia-Wen Lin
*   **Published:** 2024-09-01 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2409.00591v2](http://arxiv.org/abs/2409.00591v2)
*   **PDF Link:** [http://arxiv.org/pdf/2409.00591v2.pdf](http://arxiv.org/pdf/2409.00591v2.pdf)
*   **Summary:** The study introduces AMINet, an attention-guided Multi-scale interaction network for face super-resolution, aiming to enhance feature fusion and complementarity by incorporating local and global feature interactions through LGFI and SKAF modules, resulting in improved FSR performance with efficient computational consumption and faster inference.

--------------------

## 55. MTL-LoRA: Low-Rank Adaptation for Multi-Task Learning

*   **Category:** `cs.LG`
*   **Authors:** Yaming Yang, Dilxat Muhtar, Yelong Shen, Yuefeng Zhan, Jianfeng Liu, Yujing Wang, Hao Sun, Denvy Deng, Feng Sun, Qi Zhang, Weizhu Chen, Yunhai Tong
*   **Published:** 2024-10-12 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2410.09437v3](http://arxiv.org/abs/2410.09437v3)
*   **PDF Link:** [http://arxiv.org/pdf/2410.09437v3.pdf](http://arxiv.org/pdf/2410.09437v3.pdf)
*   **Summary:** MTL-LoRA is proposed to address task interference in multi-task learning scenarios by incorporating task-adaptive parameters to enhance both low-rank adaptation and multi-task learning capabilities. Experimental results show that MTL-LoRA outperforms LoRA and its variants in adapting to different target domains with comparable or fewer trainable parameters.

--------------------

## 56. UniGS: Modeling Unitary 3D Gaussians for Novel View Synthesis from   Sparse-view Images

*   **Category:** `cs.CV`
*   **Authors:** Jiamin Wu, Kenkun Liu, Yukai Shi, Xiaoke Jiang, Yuan Yao, Lei Zhang
*   **Published:** 2024-10-17 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2410.13195v3](http://arxiv.org/abs/2410.13195v3)
*   **PDF Link:** [http://arxiv.org/pdf/2410.13195v3.pdf](http://arxiv.org/pdf/2410.13195v3.pdf)
*   **Summary:** The paper introduces UniGS, a 3D Gaussian reconstruction and novel view synthesis model that predicts high-quality 3D Gaussians from sparse-view images using a DETR-like framework, updating unitary 3D Gaussians in world space layer by layer to avoid ghosting issues and efficiently allocate resources to complex regions. The method outperforms existing approaches, showing a 4.2 dB improvement in PSNR when trained on Objaverse and tested on the GSO benchmark. The code will be

--------------------

## 57. Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred   Image

*   **Category:** `cs.CV`
*   **Authors:** Jerred Chen, Ronald Clark
*   **Published:** 2025-03-21 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.17358v3](http://arxiv.org/abs/2503.17358v3)
*   **PDF Link:** [http://arxiv.org/pdf/2503.17358v3.pdf](http://arxiv.org/pdf/2503.17358v3.pdf)
*   **Summary:** This paper introduces a novel framework that uses motion blur as a cue for motion estimation in robotics and VR/AR applications. By predicting motion flow and depth from a single motion-blurred image, the method recovers camera velocity, yielding robust results for fast camera movements, surpassing existing methods like MASt3R and COLMAP in accuracy.

--------------------

## 58. Think or Not Think: A Study of Explicit Thinking inRule-Based Visual   Reinforcement Fine-Tuning

*   **Category:** `cs.CV`
*   **Authors:** Ming Li, Jike Zhong, Shitian Zhao, Yuxiang Lai, Kaipeng Zhang
*   **Published:** 2025-03-20 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.16188v2](http://arxiv.org/abs/2503.16188v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.16188v2.pdf](http://arxiv.org/pdf/2503.16188v2.pdf)
*   **Summary:** This paper explores rule-based reinforcement learning fine-tuning for visual classification using large language models, comparing the effectiveness of explicit thinking in CLS-RL with a novel approach, No-Thinking-RL. The study finds that No-Thinking-RL outperforms CLS-RL in in-domain performance and generalization, challenging the assumption that complex reasoning is always beneficial in fine-tuning MLLMs for visual tasks.

--------------------

## 59. FastRM: An efficient and automatic explainability framework for   multimodal generative models

*   **Category:** `cs.AI`
*   **Authors:** Gabriela Ben-Melech Stan, Estelle Aflalo, Man Luo, Shachar Rosenman, Tiep Le, Sayak Paul, Shao-Yen Tseng, Vasudev Lal
*   **Published:** 2024-12-02 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2412.01487v3](http://arxiv.org/abs/2412.01487v3)
*   **PDF Link:** [http://arxiv.org/pdf/2412.01487v3.pdf](http://arxiv.org/pdf/2412.01487v3.pdf)
*   **Summary:** This paper introduces FastRM, an efficient method for generating explainable Relevancy Maps of Large Vision Language Models (LVLMs), offering both quantitative and qualitative assessment of model confidence. FastRM significantly reduces computation time and memory footprint compared to traditional methods, making explainable AI more practical and scalable for real-world applications, enhancing the evaluation of model outputs.

--------------------

## 60. PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded   Text-to-Video Generation

*   **Category:** `cs.CV`
*   **Authors:** Qiyao Xue, Xiangyu Yin, Boyuan Yang, Wei Gao
*   **Published:** 2024-11-30 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2412.00596v2](http://arxiv.org/abs/2412.00596v2)
*   **PDF Link:** [http://arxiv.org/pdf/2412.00596v2.pdf](http://arxiv.org/pdf/2412.00596v2.pdf)
*   **Summary:** This paper introduces PhyT2V, a novel data-independent technique for text-to-video generation that enhances adherence to real-world physical rules by enabling chain-of-thought and step-back reasoning in T2V prompting. PhyT2V outperforms existing T2V models in maintaining physical realism and achieving generalizability to out-of-distribution domains.

--------------------

## 61. FisherTune: Fisher-Guided Robust Tuning of Vision Foundation Models for   Domain Generalized Segmentation

*   **Category:** `cs.CV`
*   **Authors:** Dong Zhao, Jinlong Li, Shuang Wang, Mengyao Wu, Qi Zang, Nicu Sebe, Zhun Zhong
*   **Published:** 2025-03-23 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.17940v2](http://arxiv.org/abs/2503.17940v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.17940v2.pdf](http://arxiv.org/pdf/2503.17940v2.pdf)
*   **Summary:** The paper introduces FisherTune, a fine-tuning method for Vision Foundation Models (VFMs) in Domain Generalized Semantic Segmentation (DGSS) tasks. By utilizing the Domain-Related Fisher Information Matrix (DR-FIM) to selectively update domain-sensitive parameters, FisherTune enhances adaptability in DGSS while preserving generalization, outperforming existing methods in cross-domain segmentation.

--------------------

## 62. When Counterfactual Reasoning Fails: Chaos and Real-World Complexity

*   **Category:** `cs.LG`
*   **Authors:** Yahya Aalaila, Gerrit Großmann, Sumantrak Mukherjee, Jonas Wahl, Sebastian Vollmer
*   **Published:** 2025-03-31 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23820v2](http://arxiv.org/abs/2503.23820v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23820v2.pdf](http://arxiv.org/pdf/2503.23820v2.pdf)
*   **Summary:** The paper explores the limitations of counterfactual reasoning in real-world causal modeling with uncertainties and chaotic behavior, highlighting cases where it can become unreliable, leading to significant deviations between predicted and true counterfactual trajectories. The findings caution against applying counterfactual reasoning in chaotic and uncertain settings, suggesting fundamental limitations on answering counterfactual questions about certain systems.

--------------------

## 63. Lie Detector: Unified Backdoor Detection via Cross-Examination Framework

*   **Category:** `cs.LG`
*   **Authors:** Xuan Wang, Siyuan Liang, Dongping Liao, Han Fang, Aishan Liu, Xiaochun Cao, Yu-liang Lu, Ee-Chien Chang, Xitong Gao
*   **Published:** 2025-03-21 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.16872v2](http://arxiv.org/abs/2503.16872v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.16872v2.pdf](http://arxiv.org/pdf/2503.16872v2.pdf)
*   **Summary:** The paper introduces a unified backdoor detection framework in a semi-honest setting that utilizes cross-examination of model inconsistencies between two independent service providers. By integrating central kernel alignment and backdoor fine-tuning sensitivity analysis, the proposed method achieves superior detection performance compared to existing approaches across various learning paradigms, including supervised, semi-supervised, and autoregressive tasks, with particular success in detecting backdoors in multimodal large language models.

--------------------

## 64. FedORGP: Guiding Heterogeneous Federated Learning with Orthogonality   Regularization on Global Prototypes

*   **Category:** `cs.LG`
*   **Authors:** Fucheng Guo, Zeyu Luan, Qing Li, Dan Zhao, Yong Jiang
*   **Published:** 2025-02-22 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2502.16119v2](http://arxiv.org/abs/2502.16119v2)
*   **PDF Link:** [http://arxiv.org/pdf/2502.16119v2.pdf](http://arxiv.org/pdf/2502.16119v2.pdf)
*   **Summary:** This paper introduces FedORGP, a novel Heterogeneous Federated Learning (HtFL) algorithm that improves global prototype separation through orthogonality regularization to address statistical and model heterogeneity challenges. FedORGP encourages intra-class prototype similarity and enhances inter-class angular separation, outperforming seven state-of-the-art baselines with up to a 10.12% accuracy improvement in scenarios with heterogeneity.

--------------------

## 65. An End-to-End Robust Point Cloud Semantic Segmentation Network with   Single-Step Conditional Diffusion Models

*   **Category:** `cs.CV`
*   **Authors:** Wentao Qu, Jing Wang, YongShun Gong, Xiaoshui Huang, Liang Xiao
*   **Published:** 2024-11-25 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2411.16308v4](http://arxiv.org/abs/2411.16308v4)
*   **PDF Link:** [http://arxiv.org/pdf/2411.16308v4.pdf](http://arxiv.org/pdf/2411.16308v4.pdf)
*   **Summary:** This paper introduces CDSegNet, an end-to-end semantic Segmentation Network based on a Conditional-Noise Framework (CNF) of DDPMs, which improves 3D scene understanding tasks by modeling the Noise Network as a learnable noise-feature generator. CDSegNet achieves state-of-the-art performance on public indoor and outdoor benchmarks by enhancing generalization in unseen scenes and exhibiting strong noise and sparsity robustness in experiments.

--------------------

## 66. Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model

*   **Category:** `cs.CV`
*   **Authors:** Yuxuan Zhang, Yirui Yuan, Yiren Song, Jiaming Liu
*   **Published:** 2024-03-12 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2403.07764v2](http://arxiv.org/abs/2403.07764v2)
*   **PDF Link:** [http://arxiv.org/pdf/2403.07764v2.pdf](http://arxiv.org/pdf/2403.07764v2.pdf)
*   **Summary:** This paper introduces Stable-Makeup, a diffusion-based method that can transfer a wide range of real-world makeup styles onto user-provided faces using a pre-trained diffusion model, Detail-Preserving makeup encoder, content and structural control modules, and makeup cross-attention layers in U-Net. The method demonstrates strong robustness and generalizability, outperforming existing makeup transfer methods and showing promise for applications in cross-domain makeup transfer and makeup-guided text-to-image generation.

--------------------

## 67. Local Information Matters: Inference Acceleration For Grounded   Conversation Generation Models Through Adaptive Local-Aware Token Pruning

*   **Category:** `cs.CV`
*   **Authors:** Bizhe Bai, Jianjian Cao, Yadan Luo, Tao Chen
*   **Published:** 2025-03-31 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23959v2](http://arxiv.org/abs/2503.23959v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23959v2.pdf](http://arxiv.org/pdf/2503.23959v2.pdf)
*   **Summary:** The paper introduces Adaptive Local-Aware Token Pruning (ALTP) to improve Grounded Conversation Generation (GCG) models by prioritizing local object information through Detail Density Capture (DDC) and Dynamic Density Formation (DDF). ALTP outperforms existing token pruning methods, achieving significant performance gains on GLaMM and OMG-LLaVA models with reduced computational costs.

--------------------

## 68. Mr. DETR: Instructive Multi-Route Training for Detection Transformers

*   **Category:** `cs.CV`
*   **Authors:** Chang-Bin Zhang, Yujie Zhong, Kai Han
*   **Published:** 2024-12-13 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2412.10028v2](http://arxiv.org/abs/2412.10028v2)
*   **PDF Link:** [http://arxiv.org/pdf/2412.10028v2.pdf](http://arxiv.org/pdf/2412.10028v2.pdf)
*   **Summary:** The study introduces a multi-task framework for training detection transformers by simultaneously predicting one-to-one and one-to-many assignments. Results show that components in the transformer decoder can effectively learn both tasks, leading to a proposed multi-route training mechanism with instructive self-attention for one-to-many prediction, resulting in consistent performance improvements across various baselines.

--------------------

## 69. ControlSR: Taming Diffusion Models for Consistent Real-World Image Super   Resolution

*   **Category:** `cs.CV`
*   **Authors:** Yuhao Wan, Peng-Tao Jiang, Qibin Hou, Hao Zhang, Jinwei Chen, Ming-Ming Cheng, Bo Li
*   **Published:** 2024-10-18 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2410.14279v2](http://arxiv.org/abs/2410.14279v2)
*   **PDF Link:** [http://arxiv.org/pdf/2410.14279v2.pdf](http://arxiv.org/pdf/2410.14279v2.pdf)
*   **Summary:** ControlSR is introduced as a method to enhance Diffusion Models for real-world image super-resolution (Real-ISR) by effectively utilizing low-resolution (LR) information to improve control signals in the latent space, resulting in higher-quality and more consistent super-resolution results compared to existing methods. The proposed approach demonstrates improved performance across various metrics and generates clearer super-resolution outcomes aligned with LR images.

--------------------

## 70. Prior Learning in Introspective VAEs

*   **Category:** `cs.LG`
*   **Authors:** Ioannis Athanasiadis, Fredrik Lindsten, Michael Felsberg
*   **Published:** 2024-08-25 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2408.13805v2](http://arxiv.org/abs/2408.13805v2)
*   **PDF Link:** [http://arxiv.org/pdf/2408.13805v2.pdf](http://arxiv.org/pdf/2408.13805v2.pdf)
*   **Summary:** This study explores incorporating a multimodal and learnable prior into the Soft-IntroVAE (S-IntroVAE) framework, enhancing prior learning in VAEs. The addition of a third player as the prior, along with new regularizations, improves generation and representation learning in experiments on 2D density estimation and image generation tasks using (F)-MNIST and CIFAR-10 datasets.

--------------------

## 71. StarGen: A Spatiotemporal Autoregression Framework with Video Diffusion   Model for Scalable and Controllable Scene Generation

*   **Category:** `cs.CV`
*   **Authors:** Shangjin Zhai, Zhichao Ye, Jialin Liu, Weijian Xie, Jiaqi Hu, Zhen Peng, Hua Xue, Danpeng Chen, Xiaomeng Wang, Lei Yang, Nan Wang, Haomin Liu, Guofeng Zhang
*   **Published:** 2025-01-10 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2501.05763v3](http://arxiv.org/abs/2501.05763v3)
*   **PDF Link:** [http://arxiv.org/pdf/2501.05763v3.pdf](http://arxiv.org/pdf/2501.05763v3.pdf)
*   **Summary:** The paper introduces StarGen, a novel framework utilizing a pre-trained video diffusion model for long-range scene generation by conditioning the generation of each video clip on 3D spatial warping of adjacent images and temporally overlapping images from previously generated clips, enhancing spatiotemporal consistency and pose control. StarGen demonstrates superior scalability, fidelity, and pose accuracy in tasks such as sparse view interpolation, perpetual view generation, and layout-conditioned city generation compared to existing methods.

--------------------

## 72. AnyTouch: Learning Unified Static-Dynamic Representation across Multiple   Visuo-tactile Sensors

*   **Category:** `cs.LG`
*   **Authors:** Ruoxuan Feng, Jiangyu Hu, Wenke Xia, Tianci Gao, Ao Shen, Yuhao Sun, Bin Fang, Di Hu
*   **Published:** 2025-02-15 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2502.12191v3](http://arxiv.org/abs/2502.12191v3)
*   **PDF Link:** [http://arxiv.org/pdf/2502.12191v3.pdf](http://arxiv.org/pdf/2502.12191v3.pdf)
*   **Summary:** The abstract discusses the challenges of integrating diverse visuo-tactile sensors in robotic systems and proposes TacQuad, a multi-modal dataset, and AnyTouch, a framework for learning unified multi-sensor representations from static and dynamic perspectives to enhance perception and transferability across sensors. Experimental results demonstrate the effectiveness of the proposed method in improving perception capabilities for tasks like real-world pouring.

--------------------

## 73. A Clustering Method with Graph Maximum Decoding Information

*   **Category:** `cs.LG`
*   **Authors:** Xinrun Xu, Manying Lv, Zhanbiao Lian, Yurong Wu, Jin Yan, Shan Jiang, Zhiming Ding
*   **Published:** 2024-03-18 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2403.13846v3](http://arxiv.org/abs/2403.13846v3)
*   **PDF Link:** [http://arxiv.org/pdf/2403.13846v3.pdf](http://arxiv.org/pdf/2403.13846v3.pdf)
*   **Summary:** The abstract introduces CMDI, a novel Clustering method for Maximizing Decoding Information in graph-based models, addressing the uncertainty associated with random walk access between nodes and embedded structural information. CMDI outperforms classical baseline methods in terms of decoding information ratio and computational efficiency, making it a valuable tool for enhancing decoding information quality in graph-based clustering analyses.

--------------------

## 74. Evaluating machine learning models for predicting pesticides toxicity to   honey bees

*   **Category:** `cs.LG`
*   **Authors:** Jakub Adamczyk, Jakub Poziemski, Pawel Siedlecki
*   **Published:** 2025-03-31 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.24305v2](http://arxiv.org/abs/2503.24305v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.24305v2.pdf](http://arxiv.org/pdf/2503.24305v2.pdf)
*   **Summary:** This study introduces ApisTox, a comprehensive dataset of experimentally validated chemical toxicity to honey bees, and evaluates it using various machine learning methods. The findings suggest that models trained on biomedical data may not generalize well to agrochemical datasets, emphasizing the importance of diverse data and specialized model development in the agrochemical domain.

--------------------

## 75. VFX Creator: Animated Visual Effect Generation with Controllable   Diffusion Transformer

*   **Category:** `cs.CV`
*   **Authors:** Xinyu Liu, Ailing Zeng, Wei Xue, Harry Yang, Wenhan Luo, Qifeng Liu, Yike Guo
*   **Published:** 2025-02-09 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2502.05979v4](http://arxiv.org/abs/2502.05979v4)
*   **PDF Link:** [http://arxiv.org/pdf/2502.05979v4.pdf](http://arxiv.org/pdf/2502.05979v4.pdf)
*   **Summary:** This work introduces a novel paradigm for animated VFX generation called VFX Creator, utilizing a Video Diffusion Transformer framework to generate dynamic effects from textual descriptions and reference images. The proposed system, trained on the Open-VFX dataset, demonstrates superior performance in generating realistic and controllable visual effects, enhancing accessibility to high-quality VFX for a wider audience.

--------------------

## 76. GaussianRoom: Improving 3D Gaussian Splatting with SDF Guidance and   Monocular Cues for Indoor Scene Reconstruction

*   **Category:** `cs.CV`
*   **Authors:** Haodong Xiang, Xinghui Li, Kai Cheng, Xiansong Lai, Wanting Zhang, Zhichao Liao, Long Zeng, Xueping Liu
*   **Published:** 2024-05-30 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2405.19671v2](http://arxiv.org/abs/2405.19671v2)
*   **PDF Link:** [http://arxiv.org/pdf/2405.19671v2.pdf](http://arxiv.org/pdf/2405.19671v2.pdf)
*   **Summary:** The paper introduces a unified optimization framework that combines neural signed distance fields (SDFs) with 3D Gaussian Splatting (3DGS) to improve geometry reconstruction and real-time rendering in indoor scenes, addressing challenges like incomplete and noisy reconstructions. By integrating neural SDFs with 3DGS, the proposed method achieves state-of-the-art performance in surface reconstruction and novel view synthesis, particularly in textureless areas, through the use of regularization terms and improved point cloud initialization.

--------------------

## 77. Conversations with Data: How Data Journalism Affects Online Comments in   the New York Times

*   **Category:** `cs.CY`
*   **Authors:** Avner Kantor, Sheizaf Rafaeli
*   **Published:** 2024-11-04 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2411.02045v2](http://arxiv.org/abs/2411.02045v2)
*   **PDF Link:** [http://arxiv.org/pdf/2411.02045v2.pdf](http://arxiv.org/pdf/2411.02045v2.pdf)
*   **Summary:** This study explores how data journalism (DJ) influences online comments by allowing users to interact with data through transparency and multimedia. Results show that DJ is linked to increased interactivity among users, mediated by statistical information, information sources, and static visualizations, leading to engagement in conversation and potentially fostering democratic processes.

--------------------

## 78. ResNLS: An Improved Model for Stock Price Forecasting

*   **Category:** `cs.LG`
*   **Authors:** Yuanzhe Jia, Ali Anaissi, Basem Suleiman
*   **Published:** 2023-12-02 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2312.01020v2](http://arxiv.org/abs/2312.01020v2)
*   **PDF Link:** [http://arxiv.org/pdf/2312.01020v2.pdf](http://arxiv.org/pdf/2312.01020v2.pdf)
*   **Summary:** This paper introduces a hybrid model, ResNLS, combining ResNet and LSTM neural architectures to improve stock price prediction by emphasizing dependencies between adjacent prices. The model performs optimally when using closing price data for the previous 5 consecutive trading days as input, showing at least a 20% improvement over current baselines. Backtesting results suggest that ResNLS-5 can help clients avoid risks and earn profits in the stock market.

--------------------

## 79. Content-decoupled Contrastive Learning-based Implicit Degradation   Modeling for Blind Image Super-Resolution

*   **Category:** `cs.CV`
*   **Authors:** Jiang Yuan, Ji Ma, Bo Wang, Weiming Hu
*   **Published:** 2024-08-10 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2408.05440v2](http://arxiv.org/abs/2408.05440v2)
*   **PDF Link:** [http://arxiv.org/pdf/2408.05440v2.pdf](http://arxiv.org/pdf/2408.05440v2.pdf)
*   **Summary:** This paper introduces a Content-decoupled Contrastive Learning-based blind image super-resolution (CdCL) framework that utilizes a negative-free contrastive learning technique to model implicit degradation representation. The framework includes a detail-aware implicit degradation adapting module to enhance the adaptation of degradation representations to specific low-resolution features, resulting in competitive results across various degradation settings with reduced model complexity and computational costs.

--------------------

## 80. Holistic analysis on the sustainability of Federated Learning across AI   product lifecycle

*   **Category:** `cs.LG`
*   **Authors:** Hongliu Cao
*   **Published:** 2023-12-22 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2312.14628v2](http://arxiv.org/abs/2312.14628v2)
*   **PDF Link:** [http://arxiv.org/pdf/2312.14628v2.pdf](http://arxiv.org/pdf/2312.14628v2.pdf)
*   **Summary:** This study evaluates the sustainability of Cross-Silo Federated Learning (FL) in comparison to traditional centralized approaches throughout the AI product lifecycle, highlighting comparable energy consumption and costs in model training but significant CO2 emissions savings due to reduced data transfer and storage requirements. The research also introduces a data and application management system to enhance the sustainability and economic efficiency of IT enterprises utilizing Cross-Silo FL.

--------------------

## 81. Video-T1: Test-Time Scaling for Video Generation

*   **Category:** `cs.CV`
*   **Authors:** Fangfu Liu, Hanyang Wang, Yimo Cai, Kaiyan Zhang, Xiaohang Zhan, Yueqi Duan
*   **Published:** 2025-03-24 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.18942v2](http://arxiv.org/abs/2503.18942v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.18942v2.pdf](http://arxiv.org/pdf/2503.18942v2.pdf)
*   **Summary:** The paper explores the concept of Test-Time Scaling (TTS) in video generation, aiming to improve generation quality by allowing video models to use more inference-time computation. By reinterpreting test-time scaling as a search problem in Gaussian noise space, the authors propose a method called Tree-of-Frames (ToF) that adaptively expands and prunes video branches to enhance video quality based on text prompts. Extensive experiments show that increasing test-time compute leads to substantial improvements in video quality.

--------------------

## 82. MolGround: A Benchmark for Molecular Grounding

*   **Category:** `cs.AI`
*   **Authors:** Jiaxin Wu, Ting Zhang, Rubing Chen, Wengyu Zhang, Chen Jason Zhang, Xiaoyong Wei, Li Qing
*   **Published:** 2025-03-31 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23668v2](http://arxiv.org/abs/2503.23668v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23668v2.pdf](http://arxiv.org/pdf/2503.23668v2.pdf)
*   **Summary:** The study introduces a molecular grounding benchmark to evaluate models' ability to link molecular concepts with specific structural components, aligning with conventions in NLP and molecular science. A multi-agent grounding prototype developed in this study outperforms existing models and enhances tasks like molecular captioning and ATC classification.

--------------------

## 83. Generalizable Prompt Learning of CLIP: A Brief Overview

*   **Category:** `cs.CV`
*   **Authors:** Fangming Cui, Yonggang Zhang, Xuan Wang, Xule Wang, Liang Xiao
*   **Published:** 2025-03-03 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.01263v4](http://arxiv.org/abs/2503.01263v4)
*   **PDF Link:** [http://arxiv.org/pdf/2503.01263v4.pdf](http://arxiv.org/pdf/2503.01263v4.pdf)
*   **Summary:** The abstract discusses how vision-language models like CLIP excel in generalizing across tasks by combining visual and textual information. It provides insights into CLIP's few-shot prompt learning, experimental results, and technical aspects to guide researchers new to generalizable prompting through few-shot training and encourage integration of this approach in various downstream tasks.

--------------------

## 84. ALLVB: All-in-One Long Video Understanding Benchmark

*   **Category:** `cs.CV`
*   **Authors:** Xichen Tan, Yuanjing Luo, Yunfan Ye, Fang Liu, Zhiping Cai
*   **Published:** 2025-03-10 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.07298v2](http://arxiv.org/abs/2503.07298v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.07298v2.pdf](http://arxiv.org/pdf/2503.07298v2.pdf)
*   **Summary:** The abstract introduces the ALLVB benchmark, an extensive long video understanding benchmark that integrates 9 major video tasks into a single evaluation platform. This benchmark includes 1,376 videos across 16 categories, each with an average duration of nearly 2 hours, and a total of 252k QAs, making it the largest benchmark of its kind. Testing on mainstream MLLMs reveals significant room for improvement, highlighting the challenging nature of the benchmark and the potential for advancements in long video understanding.

--------------------

## 85. Zero-Shot Visual Concept Blending Without Text Guidance

*   **Category:** `cs.CV`
*   **Authors:** Hiroya Makino, Takahiro Yamaguchi, Hiroyuki Sakai
*   **Published:** 2025-03-27 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.21277v2](http://arxiv.org/abs/2503.21277v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.21277v2.pdf](http://arxiv.org/pdf/2503.21277v2.pdf)
*   **Summary:** The paper introduces a zero-shot image generation method named "Visual Concept Blending," allowing precise control over transferring features from multiple reference images to a source image. Operating within a CLIP embedding space, the approach enables diverse transformations like style transfer and form metamorphosis without extra training, demonstrating effectiveness in combining subtle visual attributes for creative applications like art and design.

--------------------

## 86. 1-2-3-Go! Policy Synthesis for Parameterized Markov Decision Processes   via Decision-Tree Learning and Generalization

*   **Category:** `cs.AI`
*   **Authors:** Muqsit Azeem, Debraj Chakraborty, Sudeep Kanav, Jan Kretinsky, Mohammadsadegh Mohagheghi, Stefanie Mohr, Maximilian Weininger
*   **Published:** 2024-10-23 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2410.18293v2](http://arxiv.org/abs/2410.18293v2)
*   **PDF Link:** [http://arxiv.org/pdf/2410.18293v2.pdf](http://arxiv.org/pdf/2410.18293v2.pdf)
*   **Summary:** The paper addresses scalability issues in verifying parameterized Markov decision processes (MDPs) by proposing a learning-based approach to synthesize policies for huge MDPs. By generalizing optimal policies from small instances using decision-tree learning, the method avoids explicit state-space exploration of large models, offering a practical solution to the state-space explosion problem and demonstrating effectiveness through extensive experimentation.

--------------------

## 87. Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular   Videos

*   **Category:** `cs.CV`
*   **Authors:** Hanxue Liang, Jiawei Ren, Ashkan Mirzaei, Antonio Torralba, Ziwei Liu, Igor Gilitschenski, Sanja Fidler, Cengiz Oztireli, Huan Ling, Zan Gojcic, Jiahui Huang
*   **Published:** 2024-12-04 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2412.03526v2](http://arxiv.org/abs/2412.03526v2)
*   **PDF Link:** [http://arxiv.org/pdf/2412.03526v2.pdf](http://arxiv.org/pdf/2412.03526v2.pdf)
*   **Summary:** The paper introduces BTimer, a motion-aware feed-forward model for real-time reconstruction and novel view synthesis of dynamic scenes, achieving scalability and generalization by combining information from static and dynamic scene datasets. BTimer reconstructs a bullet-time scene in 150ms from a casual monocular dynamic video, outperforming optimization-based methods on both static and dynamic scene datasets.

--------------------

## 88. Designing Heterogeneous GNNs with Desired Permutation Properties for   Wireless Resource Allocation

*   **Category:** `cs.LG`
*   **Authors:** Jianyu Zhao, Chenyang Yang, Tingting Liu
*   **Published:** 2022-03-08 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2203.03906v3](http://arxiv.org/abs/2203.03906v3)
*   **PDF Link:** [http://arxiv.org/pdf/2203.03906v3.pdf](http://arxiv.org/pdf/2203.03906v3.pdf)
*   **Summary:** This paper introduces a systematic approach for designing heterogeneous graph neural networks (HetGNNs) to learn wireless policies with complex permutation properties. It proposes a method for constructing appropriate graphs and outlines three conditions for designing HetGNNs to satisfy desired permutation properties, demonstrated through power allocation and hybrid precoding policy examples.

--------------------

## 89. Diffusion Models in 3D Vision: A Survey

*   **Category:** `cs.CV`
*   **Authors:** Zhen Wang, Dongyuan Li, Yaozu Wu, Tianyu He, Jiang Bian, Renhe Jiang
*   **Published:** 2024-10-07 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2410.04738v3](http://arxiv.org/abs/2410.04738v3)
*   **PDF Link:** [http://arxiv.org/pdf/2410.04738v3.pdf](http://arxiv.org/pdf/2410.04738v3.pdf)
*   **Summary:** This paper explores the use of diffusion models in 3D vision tasks, such as object generation and point-cloud reconstruction, highlighting their potential for capturing variability and uncertainty in real-world 3D data. The authors discuss the challenges of applying diffusion models to 3D vision, including handling occlusions and high-dimensional data, and propose solutions like improving computational efficiency and exploring large-scale pretraining for better generalization across 3D tasks.

--------------------

## 90. Using Language Models to Decipher the Motivation Behind Human Behaviors

*   **Category:** `cs.AI`
*   **Authors:** Yutong Xie, Qiaozhu Mei, Walter Yuan, Matthew O. Jackson
*   **Published:** 2025-03-20 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.15752v2](http://arxiv.org/abs/2503.15752v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.15752v2.pdf](http://arxiv.org/pdf/2503.15752v2.pdf)
*   **Summary:** AI is used to analyze human behaviors by varying prompts to a language model, revealing motivations behind behaviors in economic games and uncovering relationships between different scenarios. This method can help understand behavioral tendencies across populations.

--------------------

## 91. Unveiling the Mist over 3D Vision-Language Understanding: Object-centric   Evaluation with Chain-of-Analysis

*   **Category:** `cs.CV`
*   **Authors:** Jiangyong Huang, Baoxiong Jia, Yan Wang, Ziyu Zhu, Xiongkun Linghu, Qing Li, Song-Chun Zhu, Siyuan Huang
*   **Published:** 2025-03-28 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.22420v2](http://arxiv.org/abs/2503.22420v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.22420v2.pdf](http://arxiv.org/pdf/2503.22420v2.pdf)
*   **Summary:** The abstract discusses the limitations of current 3D vision-language benchmarks and proposes Beacon3D, a new benchmark that addresses these limitations by providing high-quality test data, object-centric evaluation, and a chain-of-analysis paradigm to assess language robustness and model coherence in grounding and question answering tasks. Evaluation of state-of-the-art models on Beacon3D reveals weaknesses in generalization, coherence between grounding and question answering, and the impact of incorporating large language models on model capabilities.

--------------------

## 92. MagicPose4D: Crafting Articulated Models with Appearance and Motion   Control

*   **Category:** `cs.CV`
*   **Authors:** Hao Zhang, Di Chang, Fang Li, Mohammad Soleymani, Narendra Ahuja
*   **Published:** 2024-05-22 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2405.14017v2](http://arxiv.org/abs/2405.14017v2)
*   **PDF Link:** [http://arxiv.org/pdf/2405.14017v2.pdf](http://arxiv.org/pdf/2405.14017v2.pdf)
*   **Summary:** The paper introduces MagicPose4D, a novel framework for generating 4D content with refined control over appearance and motion. Unlike existing methods, MagicPose4D accepts monocular videos or mesh sequences as motion prompts, enabling precise and customizable motion control. Through two key modules, it achieves accurate 4D reconstruction and cross-category motion transfer, outperforming existing methods in benchmarks.

--------------------

## 93. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning   for Web Agents

*   **Category:** `cs.AI`
*   **Authors:** Yu Gu, Kai Zhang, Yuting Ning, Boyuan Zheng, Boyu Gou, Tianci Xue, Cheng Chang, Sanjari Srivastava, Yanan Xie, Peng Qi, Huan Sun, Yu Su
*   **Published:** 2024-11-10 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2411.06559v2](http://arxiv.org/abs/2411.06559v2)
*   **PDF Link:** [http://arxiv.org/pdf/2411.06559v2.pdf](http://arxiv.org/pdf/2411.06559v2.pdf)
*   **Summary:** The paper introduces WebDreamer, a model-based planning framework for web agents that uses large language models (LLMs) as world models and value functions to simulate and deliberate candidate actions before committing to them. Empirical results show that WebDreamer outperforms reactive baselines and is competitive with tree search in sandbox environments, while being 4-5 times more efficient. Additionally, the trained world model, Dreamer-7B, performs comparably to GPT-4o, showcasing the

--------------------

## 94. Phase-shifted remote photoplethysmography for estimating heart rate and   blood pressure from facial video

*   **Category:** `cs.CV`
*   **Authors:** Gyutae Hwang, Sang Jun Lee
*   **Published:** 2024-01-09 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2401.04560v4](http://arxiv.org/abs/2401.04560v4)
*   **PDF Link:** [http://arxiv.org/pdf/2401.04560v4.pdf](http://arxiv.org/pdf/2401.04560v4.pdf)
*   **Summary:** This thesis proposes a vision-based method using deep learning networks, DRP-Net and BBP-Net, to estimate heart rate and blood pressure by analyzing remote photoplethysmography signals. The method achieved a mean absolute error of 1.78 BPM for heart rate estimation and 10.19 mmHg and 7.09 mmHg for systolic and diastolic blood pressure estimation, respectively, on the MMSE-HR dataset. On the V4V dataset

--------------------

## 95. VRM: Knowledge Distillation via Virtual Relation Matching

*   **Category:** `cs.CV`
*   **Authors:** Weijia Zhang, Fei Xie, Weidong Cai, Chao Ma
*   **Published:** 2025-02-28 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2502.20760v2](http://arxiv.org/abs/2502.20760v2)
*   **PDF Link:** [http://arxiv.org/pdf/2502.20760v2.pdf](http://arxiv.org/pdf/2502.20760v2.pdf)
*   **Summary:** This paper introduces a novel relational knowledge distillation method called virtual relation matching (VRM) to address issues in relation-based methods, such as overfitting and spurious responses. By transferring affinity graphs that capture inter-sample, inter-class, and inter-view correlations, VRM provides richer guidance signals and stronger regularization for the student model, leading to superior performance on CIFAR-100 and ImageNet datasets compared to existing methods.

--------------------

## 96. ZETA: Leveraging Z-order Curves for Efficient Top-k Attention

*   **Category:** `cs.LG`
*   **Authors:** Qiuhao Zeng, Jerry Huang, Peng Lu, Gezheng Xu, Boxing Chen, Charles Ling, Boyu Wang
*   **Published:** 2025-01-24 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2501.14577v3](http://arxiv.org/abs/2501.14577v3)
*   **PDF Link:** [http://arxiv.org/pdf/2501.14577v3.pdf](http://arxiv.org/pdf/2501.14577v3.pdf)
*   **Summary:** The paper introduces ZETA, a method utilizing Z-Order Curves for Efficient Top-k Attention to enable parallel querying of past tokens for entire sequences, improving training efficiency and reducing space and time complexity to O(N log N). Experimental results show that ZETA matches standard attention performance on the Multi-Query Associative Recall task and outperforms attention and its variants on Long Range Arena and WikiText-103 language modeling.

--------------------

## 97. Controllable Human Image Generation with Personalized Multi-Garments

*   **Category:** `cs.CV`
*   **Authors:** Yisol Choi, Sangkyung Kwak, Sihyun Yu, Hyungwon Choi, Jinwoo Shin
*   **Published:** 2024-11-25 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2411.16801v3](http://arxiv.org/abs/2411.16801v3)
*   **PDF Link:** [http://arxiv.org/pdf/2411.16801v3.pdf](http://arxiv.org/pdf/2411.16801v3.pdf)
*   **Summary:** BootComp is introduced as a framework for controllable human image generation using text-to-image diffusion models with multiple reference garments. The framework addresses the challenge of data acquisition by proposing a data generation pipeline to create a synthetic dataset of human and garment pairs, along with a filtering strategy to ensure data quality, ultimately training a diffusion model for generating human images with fine details while considering multiple garment conditions.

--------------------

## 98. Buyer-Initiated Auction Mechanism for Data Redemption in Machine   Unlearning

*   **Category:** `cs.LG`
*   **Authors:** Bin Han, Di Feng, Jie Wang, Hans D. Schotten
*   **Published:** 2025-03-29 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23001v2](http://arxiv.org/abs/2503.23001v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23001v2.pdf](http://arxiv.org/pdf/2503.23001v2.pdf)
*   **Summary:** The abstract discusses the privacy concerns related to AI and user data, proposing a buyer-initiated auction mechanism for data redemption to balance cost and privacy protection without prior knowledge of users' preferences, ultimately maximizing social welfare.

--------------------

## 99. VisRL: Intention-Driven Visual Perception via Reinforced Reasoning

*   **Category:** `cs.CV`
*   **Authors:** Zhangquan Chen, Xufang Luo, Dongsheng Li
*   **Published:** 2025-03-10 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.07523v2](http://arxiv.org/abs/2503.07523v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.07523v2.pdf](http://arxiv.org/pdf/2503.07523v2.pdf)
*   **Summary:** The abstract discusses the importance of intention-driven visual understanding and introduces VisRL, a novel framework that uses reinforcement learning to optimize the visual reasoning process without relying on annotated bounding boxes. VisRL outperforms existing methods on various benchmarks and aligns more closely with human perception learning processes. The code for VisRL is available on GitHub.

--------------------

## 100. Diffusion State-Guided Projected Gradient for Inverse Problems

*   **Category:** `cs.LG`
*   **Authors:** Rayhan Zirvi, Bahareh Tolooshams, Anima Anandkumar
*   **Published:** 2024-10-04 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2410.03463v5](http://arxiv.org/abs/2410.03463v5)
*   **PDF Link:** [http://arxiv.org/pdf/2410.03463v5.pdf](http://arxiv.org/pdf/2410.03463v5.pdf)
*   **Summary:** Diffusion models have shown promise in learning data priors for inverse problems, but approximations are needed due to intractable measurement likelihoods, leading to inaccuracies and artifacts in applications like image restoration. To address this, the Diffusion State-Guided Projected Gradient (DiffStateGrad) is proposed, which enhances diffusion models by projecting the measurement gradient onto a low-rank subspace of the diffusion process, improving robustness and performance in solving inverse problems such as image restoration. The Diff

--------------------

## 101. Retrieval-augmented Few-shot Medical Image Segmentation with Foundation   Models

*   **Category:** `cs.CV`
*   **Authors:** Lin Zhao, Xiao Chen, Eric Z. Chen, Yikang Liu, Terrence Chen, Shanhui Sun
*   **Published:** 2024-08-16 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2408.08813v2](http://arxiv.org/abs/2408.08813v2)
*   **PDF Link:** [http://arxiv.org/pdf/2408.08813v2.pdf](http://arxiv.org/pdf/2408.08813v2.pdf)
*   **Summary:** The paper introduces a novel method for few-shot medical image segmentation by combining DINOv2 and SAM 2 models. By utilizing DINOv2's features for sample retrieval and SAM 2's memory attention mechanism, accurate segmentation results are achieved across different modalities without requiring retraining, showcasing superior performance and generalizability for clinical applications.

--------------------

## 102. CodingTeachLLM: Empowering LLM's Coding Ability via AST Prior Knowledge

*   **Category:** `cs.LG`
*   **Authors:** Zhangquan Chen, Chunjiang Liu, Haobin Duan
*   **Published:** 2024-03-13 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2403.15426v2](http://arxiv.org/abs/2403.15426v2)
*   **PDF Link:** [http://arxiv.org/pdf/2403.15426v2.pdf](http://arxiv.org/pdf/2403.15426v2.pdf)
*   **Summary:** This paper introduces CodingTeachLLM, a large language model tailored for coding education, which outperforms traditional fine-tuning methods by employing an end-to-end prior-based three-phases supervised fine-tuned model. The model enhances coding ability through structural disassembly, incremental guided output of educational knowledge, and achieves state-of-the-art results in code abilities and conversational capabilities in various benchmarks.

--------------------

## 103. VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning

*   **Category:** `cs.CV`
*   **Authors:** Ye Liu, Kevin Qinghong Lin, Chang Wen Chen, Mike Zheng Shou
*   **Published:** 2025-03-17 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.13444v2](http://arxiv.org/abs/2503.13444v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.13444v2.pdf](http://arxiv.org/pdf/2503.13444v2.pdf)
*   **Summary:** This work introduces VideoMind, a video-language agent designed for temporal-grounded video understanding, incorporating a role-based agentic workflow and a Chain-of-LoRA strategy to efficiently integrate various roles. Extensive experiments show that VideoMind achieves state-of-the-art performance on diverse video understanding tasks, highlighting its effectiveness in advancing video agent and long-form temporal reasoning.

--------------------

## 104. Without Paired Labeled Data: An End-to-End Self-Supervised Paradigm for   UAV-View Geo-Localization

*   **Category:** `cs.CV`
*   **Authors:** Zhongwei Chen, Zhao-Xu Yang, Hai-Jun Rong
*   **Published:** 2025-02-17 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2502.11381v2](http://arxiv.org/abs/2502.11381v2)
*   **PDF Link:** [http://arxiv.org/pdf/2502.11381v2.pdf](http://arxiv.org/pdf/2502.11381v2.pdf)
*   **Summary:** This paper introduces a self-supervised method for UAV-View Geo-Localization (UVGL) that does not require pre-paired UAV-satellite images for training. The proposed method utilizes clustering, contrastive learning, memory learning, and information consistency evolution to improve feature representation and outperforms existing self-supervised and supervised methods on benchmark datasets.

--------------------

## 105. HumanDreamer: Generating Controllable Human-Motion Videos via Decoupled   Generation

*   **Category:** `cs.CV`
*   **Authors:** Boyuan Wang, Xiaofeng Wang, Chaojun Ni, Guosheng Zhao, Zhiqin Yang, Zheng Zhu, Muyang Zhang, Yukun Zhou, Xinze Chen, Guan Huang, Lihong Liu, Xingang Wang
*   **Published:** 2025-03-31 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.24026v2](http://arxiv.org/abs/2503.24026v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.24026v2.pdf](http://arxiv.org/pdf/2503.24026v2.pdf)
*   **Summary:** The paper introduces HumanDreamer, a framework for human video generation that generates diverse poses from text prompts and then creates human-motion videos based on these poses. The proposed MotionVid dataset and MotionDiT model, along with the LAMA loss, significantly improve FID and R-precision metrics, enhancing both Text-to-Pose control accuracy and video quality. The method produces diverse, high-quality human-motion videos and can aid in pose sequence prediction and 2D-3D motion lifting tasks.

--------------------

## 106. Time-Series Forecasting via Topological Information Supervised Framework   with Efficient Topological Feature Learning

*   **Category:** `cs.LG`
*   **Authors:** ZiXin Lin, Nur Fariha Syaqina Zulkepli
*   **Published:** 2025-03-31 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23757v2](http://arxiv.org/abs/2503.23757v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23757v2.pdf](http://arxiv.org/pdf/2503.23757v2.pdf)
*   **Summary:** This study introduces the Topological Information Supervised (TIS) Prediction framework, which combines neural networks and Conditional Generative Adversarial Networks (CGANs) to address challenges in integrating Topological Data Analysis (TDA) with time-series prediction. The proposed TIS models, TIS-BiGRU and TIS-Informer, outperform conventional predictors by capturing short-term and long-term temporal dependencies while preserving the distribution of synthetic topological features, showcasing the potential of leveraging topological information

--------------------

## 107. Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable   Scales

*   **Category:** `cs.CV`
*   **Authors:** Shuokai Pan, Gerti Tuzi, Sudarshan Sreeram, Dibakar Gope
*   **Published:** 2024-12-27 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2412.19867v2](http://arxiv.org/abs/2412.19867v2)
*   **PDF Link:** [http://arxiv.org/pdf/2412.19867v2.pdf](http://arxiv.org/pdf/2412.19867v2.pdf)
*   **Summary:** The abstract discusses the challenges of high computational costs in large-scale text-to-image diffusion models and proposes a method for quantization using finer-grained group-wise quantization to improve inference time without significant quality loss. The proposed method, which involves finetuning only the scale parameters of Winograd transform matrices, demonstrates near-lossless quality for text-to-image generation and outperforms existing methods in image classification tasks.

--------------------

## 108. DC-SGD: Differentially Private SGD with Dynamic Clipping through   Gradient Norm Distribution Estimation

*   **Category:** `cs.LG`
*   **Authors:** Chengkun Wei, Weixian Li, Chen Gong, Wenzhi Chen
*   **Published:** 2025-03-29 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.22988v2](http://arxiv.org/abs/2503.22988v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.22988v2.pdf](http://arxiv.org/pdf/2503.22988v2.pdf)
*   **Summary:** The paper introduces Dynamic Clipping DP-SGD (DC-SGD), a framework that uses differentially private histograms to adjust the clipping threshold dynamically in Differentially Private Stochastic Gradient Descent (DP-SGD) to reduce hyperparameter tuning burden. Experimental results demonstrate up to 9 times faster hyperparameter tuning and a 10.62% accuracy improvement on CIFAR10 compared to DP-SGD under the same privacy budget, with theoretical privacy and convergence analyses supporting the integration with the Adam optimizer.

--------------------

## 109. Visual Acoustic Fields

*   **Category:** `cs.CV`
*   **Authors:** Yuelei Li, Hyunjin Kim, Fangneng Zhan, Ri-Zhao Qiu, Mazeyu Ji, Xiaojun Shan, Xueyan Zou, Paul Liang, Hanspeter Pfister, Xiaolong Wang
*   **Published:** 2025-03-31 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.24270v2](http://arxiv.org/abs/2503.24270v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.24270v2.pdf](http://arxiv.org/pdf/2503.24270v2.pdf)
*   **Summary:** The paper introduces Visual Acoustic Fields, a framework that connects hitting sounds and visual signals in a 3D space using 3D Gaussian Splatting (3DGS). It includes modules for sound generation and localization, leveraging a conditional diffusion model and a novel pipeline for collecting scene-level visual-sound sample pairs to generate realistic impact sounds and accurately localize impact sources.

--------------------

## 110. Astrea: A MOE-based Visual Understanding Model with Progressive   Alignment

*   **Category:** `cs.CV`
*   **Authors:** Xiaoda Yang, JunYu Lu, Hongshun Qiu, Sijing Li, Hao Li, Shengpeng Ji, Xudong Tang, Jiayang Xu, Jiaqi Duan, Ziyue Jiang, Cong Lin, Sihang Cai, Zejian Xie, Zhuoyang Song, Songxin Zhang
*   **Published:** 2025-03-12 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.09445v2](http://arxiv.org/abs/2503.09445v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.09445v2.pdf](http://arxiv.org/pdf/2503.09445v2.pdf)
*   **Summary:** The paper introduces Astrea, a novel vision-language model architecture that addresses task heterogeneity and expert load imbalance by utilizing a heterogeneous expert coordination mechanism, dynamic knowledge fusion strategy, and enhanced optimization framework. Astrea outperforms existing models across various benchmark tasks, showcasing a +4.7% average performance gain and demonstrating the effectiveness of progressive pre-alignment strategies in improving multimodal agent capabilities.

--------------------

## 111. 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large   Language Models

*   **Category:** `cs.CV`
*   **Authors:** Wanhua Li, Renping Zhou, Jiawei Zhou, Yingwei Song, Johannes Herter, Minghan Qin, Gao Huang, Hanspeter Pfister
*   **Published:** 2025-03-13 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.10437v2](http://arxiv.org/abs/2503.10437v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.10437v2.pdf](http://arxiv.org/pdf/2503.10437v2.pdf)
*   **Summary:** The paper introduces 4D LangSplat, a method that learns 4D language fields to handle time-sensitive language queries in dynamic scenes efficiently by generating detailed, temporally consistent captions for objects in videos. By leveraging Multimodal Large Language Models and a status deformable network, 4D LangSplat achieves precise and efficient results for open-vocabulary queries in dynamic environments.

--------------------

## 112. Learned Image Compression and Restoration for Digital Pathology

*   **Category:** `cs.CV`
*   **Authors:** SeonYeong Lee, EonSeung Seong, DongEon Lee, SiYeoul Lee, Yubin Cho, Chunsu Park, Seonho Kim, MinKyung Seo, YoungSin Ko, MinWoo Kim
*   **Published:** 2025-03-31 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23862v2](http://arxiv.org/abs/2503.23862v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23862v2.pdf](http://arxiv.org/pdf/2503.23862v2.pdf)
*   **Summary:** CLERIC is a deep learning-based image compression framework tailored for whole slide images in digital pathology, utilizing a lifting scheme and advanced convolutional techniques to efficiently compress images while preserving crucial pathological details. Experimental results show that CLERIC outperforms state-of-the-art learned image compression models in terms of rate-distortion performance, reducing storage requirements without compromising diagnostic image quality.

--------------------

## 113. TextCrafter: Accurately Rendering Multiple Texts in Complex Visual   Scenes

*   **Category:** `cs.CV`
*   **Authors:** Nikai Du, Zhennan Chen, Zhizhou Chen, Shan Gao, Xi Chen, Zhengkai Jiang, Jian Yang, Ying Tai
*   **Published:** 2025-03-30 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23461v2](http://arxiv.org/abs/2503.23461v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23461v2.pdf](http://arxiv.org/pdf/2503.23461v2.pdf)
*   **Summary:** This paper introduces TextCrafter, a novel method for Complex Visual Text Generation (CVTG) that decomposes complex visual text into distinct components and enhances the prominence of visual text during generation to address challenges like text confusion and blurriness. TextCrafter outperforms existing approaches on the new benchmark dataset, CVTG-2K, designed to evaluate generative models for CVTG tasks.

--------------------

## 114. EventMamba: Enhancing Spatio-Temporal Locality with State Space Models   for Event-Based Video Reconstruction

*   **Category:** `cs.CV`
*   **Authors:** Chengjie Ge, Xueyang Fu, Peng He, Kunyu Wang, Chengzhi Cao, Zheng-Jun Zha
*   **Published:** 2025-03-25 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.19721v3](http://arxiv.org/abs/2503.19721v3)
*   **PDF Link:** [http://arxiv.org/pdf/2503.19721v3.pdf](http://arxiv.org/pdf/2503.19721v3.pdf)
*   **Summary:** EventMamba is a specialized model for event-based video reconstruction that addresses the limitations of conventional Mamba algorithms by introducing random window offset in the spatial domain and a consistent traversal serialization approach in the spatio-temporal domain. It significantly improves computation speed and visual quality compared to Transformer-based methods, retaining Mamba's robust modeling capabilities while preserving spatio-temporal locality of event data.

--------------------

## 115. Where am I? Cross-View Geo-localization with Natural Language   Descriptions

*   **Category:** `cs.CV`
*   **Authors:** Junyan Ye, Honglin Lin, Leyan Ou, Dairong Chen, Zihao Wang, Qi Zhu, Conghui He, Weijia Li
*   **Published:** 2024-12-22 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2412.17007v2](http://arxiv.org/abs/2412.17007v2)
*   **PDF Link:** [http://arxiv.org/pdf/2412.17007v2.pdf](http://arxiv.org/pdf/2412.17007v2.pdf)
*   **Summary:** This study introduces a novel task for cross-view geo-localization with natural language descriptions to retrieve corresponding satellite images or OSM database based on scene text descriptions. The proposed method, CrossText2Loc, improves recall by 10% and enhances long-text retrieval capabilities while providing explanation for retrieval results. More details can be found at https://yejy53.github.io/CVG-Text/.

--------------------

## 116. Alleviating Hallucinations in Large Vision-Language Models through   Hallucination-Induced Optimization

*   **Category:** `cs.CV`
*   **Authors:** Xinyu Lyu, Beitao Chen, Lianli Gao, Jingkuan Song, Heng Tao Shen
*   **Published:** 2024-05-24 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2405.15356v3](http://arxiv.org/abs/2405.15356v3)
*   **PDF Link:** [http://arxiv.org/pdf/2405.15356v3.pdf](http://arxiv.org/pdf/2405.15356v3.pdf)
*   **Summary:** The paper addresses the issue of hallucinations in Large Visual Language Models (LVLMs) by introducing a novel optimization strategy called Hallucination-Induced Optimization (HIO). Through theoretical analysis and leveraging a Contrary Bradley-Terry Model, HIO effectively reduces hallucinations in LVLMs, outperforming existing methods in experimental evaluations on different benchmarks.

--------------------

## 117. On-device Sora: Enabling Training-Free Diffusion-based Text-to-Video   Generation for Mobile Devices

*   **Category:** `cs.CV`
*   **Authors:** Bosung Kim, Kyuhwan Lee, Isu Jeong, Jungmin Cheon, Yeojin Lee, Seulki Lee
*   **Published:** 2025-03-31 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23796v2](http://arxiv.org/abs/2503.23796v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23796v2.pdf](http://arxiv.org/pdf/2503.23796v2.pdf)
*   **Summary:** On-device Sora is introduced as a model training-free solution for diffusion-based text-to-video generation on mobile devices, using novel techniques like Linear Proportional Leap, Temporal Dimension Token Merging, and Concurrent Inference with Dynamic Loading. Experimental results on an iPhone 15 Pro demonstrate its ability to generate high-quality videos comparable to those from high-end GPUs, showcasing its potential for efficient video generation on resource-constrained mobile devices without the need for resource-intensive re-training. The code implementation is available on

--------------------

## 118. Conditional Variable Flow Matching: Transforming Conditional Densities   with Amortized Conditional Optimal Transport

*   **Category:** `cs.LG`
*   **Authors:** Adam P. Generale, Andreas E. Robertson, Surya R. Kalidindi
*   **Published:** 2024-11-13 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2411.08314v4](http://arxiv.org/abs/2411.08314v4)
*   **PDF Link:** [http://arxiv.org/pdf/2411.08314v4.pdf](http://arxiv.org/pdf/2411.08314v4.pdf)
*   **Summary:** The paper introduces Conditional Variable Flow Matching (CVFM), a framework that enables learning flows transforming conditional distributions across continuous conditioning variables, enhancing predictions across the conditional density manifold. By employing simultaneous sample conditioned flows and a conditional Wasserstein distance with loss reweighting kernel, CVFM demonstrates improved performance and convergence characteristics on various challenging problems compared to existing conditional methods.

--------------------

## 119. Lean Formalization of Generalization Error Bound by Rademacher   Complexity

*   **Category:** `cs.LG`
*   **Authors:** Sho Sonoda, Kazumi Kasaura, Yuma Mizuno, Kei Tsukamoto, Naoto Onda
*   **Published:** 2025-03-25 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.19605v2](http://arxiv.org/abs/2503.19605v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.19605v2.pdf](http://arxiv.org/pdf/2503.19605v2.pdf)
*   **Summary:** This paper introduces a formalization of generalization error bounds using Rademacher complexity in the Lean 4 theorem prover. It discusses how Rademacher complexity can estimate the generalization error by considering the complexity of learning machines, applicable to various machine learning scenarios including deep learning and kernel methods, and provides formal proofs for key concepts and theorems such as empirical and population Rademacher complexities.

--------------------

## 120. WaveFormer: A 3D Transformer with Wavelet-Driven Feature Representation   for Efficient Medical Image Segmentation

*   **Category:** `cs.CV`
*   **Authors:** Md Mahfuz Al Hasan, Mahdi Zaman, Abdul Jawad, Alberto Santamaria-Pang, Ho Hin Lee, Ivan Tarapov, Kyle See, Md Shah Imran, Antika Roy, Yaser Pourmohammadi Fallah, Navid Asadizanjani, Reza Forghani
*   **Published:** 2025-03-31 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23764v2](http://arxiv.org/abs/2503.23764v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23764v2.pdf](http://arxiv.org/pdf/2503.23764v2.pdf)
*   **Summary:** WaveFormer is introduced as a 3D-transformer that addresses memory overhead and local feature capture challenges in medical image analysis by leveraging frequency-domain properties and a biologically inspired top-down mechanism. By utilizing discrete wavelet transformations, WaveFormer efficiently preserves global context and high-frequency details, reducing parameters for improved real-world deployment, with evaluations showing competitive performance and lower computational complexity compared to existing methods on various datasets.

--------------------

## 121. LVAgent: Long Video Understanding by Multi-Round Dynamical Collaboration   of MLLM Agents

*   **Category:** `cs.CV`
*   **Authors:** Boyu Chen, Zhengrong Yue, Siran Chen, Zikang Wang, Yang Liu, Peng Li, Yali Wang
*   **Published:** 2025-03-13 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.10200v2](http://arxiv.org/abs/2503.10200v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.10200v2.pdf](http://arxiv.org/pdf/2503.10200v2.pdf)
*   **Summary:** LVAgent is a novel framework that facilitates multi-round dynamic collaboration among Multimodal Large Language Models (MLLMs) to enhance long video understanding. By enabling agent teams to iteratively refine answers through a selection, perception, action, and reflection process, LVAgent achieves superior performance compared to existing models, with an accuracy of 80% on various long video tasks and up to a 13.3% improvement over the current state-of-the-art on the LongVideoBench dataset.

--------------------

## 122. PSF-4D: A Progressive Sampling Framework for View Consistent 4D Editing

*   **Category:** `cs.CV`
*   **Authors:** Hasan Iqbal, Nazmul Karim, Umar Khalid, Azib Farooq, Zichun Zhong, Chen Chen, Jing Hua
*   **Published:** 2025-03-14 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.11044v3](http://arxiv.org/abs/2503.11044v3)
*   **PDF Link:** [http://arxiv.org/pdf/2503.11044v3.pdf](http://arxiv.org/pdf/2503.11044v3.pdf)
*   **Summary:** The paper introduces a progressive sampling framework for 4D editing (PSF-4D) that ensures temporal and multi-view consistency by controlling noise initialization during forward diffusion. By incorporating correlated Gaussian noise structures for temporal coherence and cross-view noise models for spatial consistency, PSF-4D achieves high-quality 4D editing without external models, outperforming state-of-the-art methods in various editing tasks according to extensive evaluations on multiple benchmarks.

--------------------

## 123. Rerouting Connection: Hybrid Computer Vision Analysis Reveals Visual   Similarity Between Indus and Tibetan-Yi Corridor Writing Systems

*   **Category:** `cs.CV`
*   **Authors:** Ooha Lakkadi Reddy
*   **Published:** 2025-03-27 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.21074v2](http://arxiv.org/abs/2503.21074v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.21074v2.pdf](http://arxiv.org/pdf/2503.21074v2.pdf)
*   **Summary:** This thesis uses a hybrid CNN-Transformer architecture and an anthropological framework to explore historical connections between the Indus Valley script and Tibetan-Yi Corridor pictographic systems. Results show higher visual similarity between Tibetan-Yi Corridor scripts and the Indus script compared to Bronze Age Proto-Cuneiform and Proto-Elamite systems, challenging traditional views of script development and suggesting intricate cultural transmission networks between South and East Asia.

--------------------

## 124. Devils in Middle Layers of Large Vision-Language Models: Interpreting,   Detecting and Mitigating Object Hallucinations via Attention Lens

*   **Category:** `cs.CV`
*   **Authors:** Zhangqi Jiang, Junkai Chen, Beier Zhu, Tingjin Luo, Yankun Shen, Xu Yang
*   **Published:** 2024-11-23 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2411.16724v3](http://arxiv.org/abs/2411.16724v3)
*   **PDF Link:** [http://arxiv.org/pdf/2411.16724v3.pdf](http://arxiv.org/pdf/2411.16724v3.pdf)
*   **Summary:** This paper investigates how Large Vision-Language Models (LVLMs) process visual information, identifying middle layers as crucial for causing hallucinations. By analyzing attention patterns, the study proposes an inference-time method that adjusts visual attention to reduce hallucinations in LVLMs effectively.

--------------------

## 125. UniFlow: A Foundation Model for Unified Urban Spatio-Temporal Flow   Prediction

*   **Category:** `cs.LG`
*   **Authors:** Yuan Yuan, Jingtao Ding, Chonghua Han, Zhi Sheng, Depeng Jin, Yong Li
*   **Published:** 2024-11-20 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2411.12972v3](http://arxiv.org/abs/2411.12972v3)
*   **PDF Link:** [http://arxiv.org/pdf/2411.12972v3.pdf](http://arxiv.org/pdf/2411.12972v3.pdf)
*   **Summary:** This paper introduces UniFlow, a model for urban flow prediction that combines grid-based and graph-based data using a multi-view spatio-temporal patching mechanism and a spatio-temporal transformer architecture. The proposed SpatioTemporal Memory Retrieval Augmentation (ST-MRA) enhances predictions by leveraging shared spatio-temporal patterns stored in structured memory modules. UniFlow outperforms existing models in both grid-based and graph-based flow prediction, especially in scenarios with limited data availability, demonstrating its superior performance

--------------------

## 126. Towards shutdownable agents via stochastic choice

*   **Category:** `cs.AI`
*   **Authors:** Elliott Thornley, Alexander Roman, Christos Ziakas, Leyton Ho, Louis Thomson
*   **Published:** 2024-06-30 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2407.00805v5](http://arxiv.org/abs/2407.00805v5)
*   **PDF Link:** [http://arxiv.org/pdf/2407.00805v5.pdf](http://arxiv.org/pdf/2407.00805v5.pdf)
*   **Summary:** The paper introduces the Incomplete Preferences Proposal (IPP) to ensure artificial agents don't resist shutdown by using the DReST reward function to train agents to effectively pursue goals and be neutral about trajectory lengths. The study demonstrates that agents trained with the DReST reward function in gridworlds learn to be both useful and neutral, suggesting potential for training advanced agents to be compliant and effective.

--------------------

## 127. Provably-Safe Neural Network Training Using Hybrid Zonotope Reachability   Analysis

*   **Category:** `cs.LG`
*   **Authors:** Long Kiu Chung, Shreyas Kousik
*   **Published:** 2025-01-22 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2501.13023v2](http://arxiv.org/abs/2501.13023v2)
*   **PDF Link:** [http://arxiv.org/pdf/2501.13023v2.pdf](http://arxiv.org/pdf/2501.13023v2.pdf)
*   **Summary:** This paper introduces a neural network training method that aims to steer a network away from unsafe regions by encouraging exact images of non-convex input sets using reachability analysis with scaled hybrid zonotopes. The proposed method is effective and computationally efficient for networks with up to 240 neurons, demonstrating practical applications in training controllers and generating safe plans for dynamical systems.

--------------------

## 128. View-Invariant Pixelwise Anomaly Detection in Multi-object Scenes with   Adaptive View Synthesis

*   **Category:** `cs.CV`
*   **Authors:** Subin Varghese, Vedhus Hoskere
*   **Published:** 2024-06-26 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2406.18012v2](http://arxiv.org/abs/2406.18012v2)
*   **PDF Link:** [http://arxiv.org/pdf/2406.18012v2.pdf](http://arxiv.org/pdf/2406.18012v2.pdf)
*   **Summary:** The paper introduces Scene Anomaly Detection (Scene AD) for detecting anomalies in images with varying camera poses in the built environment. They propose a novel network, OmniAD, which improves pixel-level anomaly detection by 40% using a reverse distillation anomaly detection method and new data augmentation strategies. The method is evaluated on a new dataset, ToyCity, and an established dataset, MAD, showing significant enhancements for robust anomaly detection in real-world scenes.

--------------------

## 129. Robust Bayesian Optimization via Localized Online Conformal Prediction

*   **Category:** `cs.LG`
*   **Authors:** Dongwon Kim, Matteo Zecchin, Sangwoo Park, Joonhyuk Kang, Osvaldo Simeone
*   **Published:** 2024-11-26 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2411.17387v2](http://arxiv.org/abs/2411.17387v2)
*   **PDF Link:** [http://arxiv.org/pdf/2411.17387v2.pdf](http://arxiv.org/pdf/2411.17387v2.pdf)
*   **Summary:** The paper introduces LOCBO, a Bayesian optimization algorithm that uses localized online conformal prediction to calibrate Gaussian process models for improved performance in the presence of model misspecification. The method provides theoretical performance guarantees and outperforms existing BO algorithms in experiments on synthetic and real-world tasks.

--------------------

## 130. Convolutional Neural Networks Can (Meta-)Learn the Same-Different   Relation

*   **Category:** `cs.CV`
*   **Authors:** Max Gupta, Sunayana Rane, R. Thomas McCoy, Thomas L. Griffiths
*   **Published:** 2025-03-29 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.23212v2](http://arxiv.org/abs/2503.23212v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.23212v2.pdf](http://arxiv.org/pdf/2503.23212v2.pdf)
*   **Summary:** This study highlights that while CNNs struggle to generalize the same-different relation in visual tasks through conventional training, they can succeed when trained using meta-learning, which promotes abstraction and generalization across tasks.

--------------------

## 131. Enhancing Domain Adaptation through Prompt Gradient Alignment

*   **Category:** `cs.LG`
*   **Authors:** Hoang Phan, Lam Tran, Quyen Tran, Trung Le
*   **Published:** 2024-06-13 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2406.09353v3](http://arxiv.org/abs/2406.09353v3)
*   **PDF Link:** [http://arxiv.org/pdf/2406.09353v3.pdf](http://arxiv.org/pdf/2406.09353v3.pdf)
*   **Summary:** The paper introduces a new method for Unsupervised Domain Adaptation (UDA) that aligns per-objective gradients to enhance consensus between domain losses, with a focus on preventing overfitting during fine-tuning of deep learning models. The proposed method consistently outperforms existing vision-language model adaptation techniques in both single-source and multi-source UDA scenarios.

--------------------

## 132. Independent and Decentralized Learning in Markov Potential Games

*   **Category:** `cs.LG`
*   **Authors:** Chinmay Maheshwari, Manxi Wu, Druv Pai, Shankar Sastry
*   **Published:** 2022-05-29 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2205.14590v8](http://arxiv.org/abs/2205.14590v8)
*   **PDF Link:** [http://arxiv.org/pdf/2205.14590v8.pdf](http://arxiv.org/pdf/2205.14590v8.pdf)
*   **Summary:** This study examines multi-agent reinforcement learning dynamics in infinite-horizon discounted Markov potential games, focusing on independent and decentralized settings where players lack game parameters and communication. Players update Q-function estimates asynchronously and adjust policies based on estimated Q-functions, with a key feature being faster Q-function updates compared to policy updates. The convergence of these learning dynamics is characterized using tools from two-timescale asynchronous stochastic approximation theory.

--------------------

## 133. Process or Result? Manipulated Ending Tokens Can Mislead Reasoning LLMs   to Ignore the Correct Reasoning Steps

*   **Category:** `cs.AI`
*   **Authors:** Yu Cui, Bryan Hooi, Yujun Cai, Yiwei Wang
*   **Published:** 2025-03-25 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2503.19326v2](http://arxiv.org/abs/2503.19326v2)
*   **PDF Link:** [http://arxiv.org/pdf/2503.19326v2.pdf](http://arxiv.org/pdf/2503.19326v2.pdf)
*   **Summary:** The study explores the vulnerability of reasoning large language models (LLMs) to subtle errors in input reasoning chains, introducing "Compromising Thought" (CPT) where manipulated calculation results can lead models to adopt incorrect reasoning steps. The research reveals that models struggle to identify and correct these manipulations, with local ending token manipulations having a greater impact on reasoning outcomes than structural changes, and identifies a security vulnerability in DeepSeek-R1 where tampered reasoning tokens can cause reasoning cessation.

--------------------

## 134. Disentangling Safe and Unsafe Corruptions via Anisotropy and Locality

*   **Category:** `cs.CV`
*   **Authors:** Ramchandran Muthukumar, Ambar Pal, Jeremias Sulam, Rene Vidal
*   **Published:** 2025-01-30 (Updated: 2025-04-01)
*   **Link:** [http://arxiv.org/abs/2501.18098v3](http://arxiv.org/abs/2501.18098v3)
*   **PDF Link:** [http://arxiv.org/pdf/2501.18098v3.pdf](http://arxiv.org/pdf/2501.18098v3.pdf)
*   **Summary:** This paper introduces a novel threat model called Projected Displacement (PD) to assess robustness in machine learning systems beyond existing isotropic and global threat models. The PD model measures threat based on alignment with unsafe directions in the input space, showing anisotropy and locality. Experiments on Imagenet-1k data demonstrate that the PD model can identify safe perturbations like noise, blur, and compression while excluding unsafe perturbations that alter the true label, without requiring pre-training

--------------------

