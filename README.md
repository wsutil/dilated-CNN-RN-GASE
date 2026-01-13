# dilated-CNN-RN-GASE
Diffusion Tractography Biomarker for Epilepsy Severity in Children With Drug-Resistant Epilepsy

## Overview
![Fig1](https://github.com/user-attachments/assets/3bc81d09-97dc-4298-8521-87f2dcd43d72)

Objective: To develop a novel deep-learning model of clinical DWI tractography that can accurately predict the general assessment of epilepsy severity (GASE) in pediatric drug-resistant epilepsy (DRE) and test if it can screen diverse neurocognitive impairments identified through neuropsychological assessments.
Methods: DRE children and age-sex-matched healthy controls were enrolled to construct an epilepsy severity network (ESN), whose edges were significantly correlated with GASE scores of DRE children. An ESN-based biomarker called the predicted GASE score was obtained using dilated deep convolutional neural network with a relational network (dilated DCNN+RN) and used to quantify the risk of neurocognitive impairments using global/verbal/non-verbal neuropsychological assessments of 36/37/32 children performed on average 3.2 ± 2.7 months prior to the MRI scan. To warrant the generalizability, the proposed biomarker was trained and evaluated using separate development and independent test sets, with the random score learning experiment included to assess potential overfitting.
Results: The dilated DCNN+RN outperformed other state-of-the art methods to create the predicted GASE scores with significant correlation (r = 0.92 and 0.83 for development and test sets with clinical GASE scores) and minimal overfitting (r = −0.25 and 0.00 for development and test sets with random GASE scores). Both univariate and multivariate models demonstrated that compared with the clinical GASE scores, the predicted GASE scores provide better model fit and discriminatory ability, suggesting more adjusted and accurate estimate of epilepsy severity contributing to the overall risk. 
Interpretation: The proposed biomarker shows strong potential for early identification of DRE children at risk of neurocognitive impairments, enabling timely, personalized interventions to prevent long-term effects.

## Reference

If you use this work, please cite our papers. For the base model and prior work, please also cite the following.

```bibtex
@article{Jeong2025,
  title={Diffusion Tractography Biomarker for Epilepsy Severity in Children With Drug-Resistant Epilepsy},
  author={Jeong, Jeong-Won and Lee, Min-Hee and Uda, Hiroshi and Hwang, Yoon Ho and Behen, Michael and Luat, Aimee and Juhász, Csaba and Asano, Eishi},
  journal={Annals of Clinical and Translational Neurology},
  year={2025},
  doi={10.1002/acn3.70217}
}

@article{banerjee2021,
  title={Deep relational reasoning for the prediction of language impairment and postoperative seizure outcome using preoperative DWI connectome data of children with focal epilepsy},
  author={Banerjee, Soumyanil and Dong, Ming and Lee, Min-Hee and O’Hara, Nolan and Juhász, Csaba and Asano, Eishi and Jeong, Jeong-Won},
  journal={IEEE Transactions on Medical Imaging},
  volume={40},
  number={3},
  pages={793--804},
  year={2021},
  doi={10.1109/TMI.2020.3036933}
}
