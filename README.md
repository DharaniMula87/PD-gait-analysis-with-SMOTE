# PD-gait-analysis-with-SMOTE
Parkinson’s Disease stage classification using gait analysis with SMOTE-based class balancing. Machine learning models are trained and tested on gait features extracted from VGRF signals. Results show that SMOTE improves accuracy and prediction performance compared to the original imbalanced dataset.

## Research Paper Reference

*Title*:Parkinson’s Disease Stage Classification with Gait Analysis using Machine Learning Techniques and SMOTE-based Approach for Class Imbalance Problem  
*Authors*: Aishwarya Balakrishnan, Jeevan Medikonda, Pramod K Namboothiri, Manikandan Natarajan  
*Conference*: 2022 International Conference on Distributed Computing, VLSI, Electrical Circuits and Robotics (DISCOVER)  
*DOI*: [10.1109/DISCOVER55800.2022.9974754](https://doi.org/10.1109/DISCOVER55800.2022.9974754)

---

## Dataset

- *Source*: PhysioNet Gait in Parkinson’s Disease Database  
- *Subset Used*: Si subset (includes VGRF signals and demographics)  
- *Link*: [https://physionet.org/content/gaitpdb/1.0.0/](https://physionet.org/content/gaitpdb/1.0.0/)

---

## Workflow Overview

1. *Data Collection*  
   - Downloaded Si subset from PhysioNet (VGRF signals + demographics)

2. *Preprocessing*  
   - Cleaned and merged VGRF signals  
   - Filtered demographics based on research criteria  
   - Applied Savitzky-Golay filter to smooth noisy signals

3. *Feature Engineering*  
   - Extracted gait features: cadence, stride length, regularity, symmetry, step count  
   - Combined gait features with demographic data

4. *Class Balancing*  
   - Applied SMOTE to balance Healthy, Stage 2, and Stage 2.5 samples

5. *Model Training*
   - Trained multiple classifiers using 5-fold cross-validation  
   - *Random Forest* achieved the best accuracy

---

## Key Insight

> The Random Forest classifier achieved the best performance in this study. Compared to the original research paper, which reported an accuracy of *86.2%*, this implementation with Random Forest surpassed that benchmark — demonstrating improved generalizability and robustness in multi-class classification of PD stages.

---

##  License

This project is licensed under the MIT License.

---

## Contact Me

If you have any related queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/public-profile/settings?trk=d_flagship3_profile_self_view_public_profile).

---
