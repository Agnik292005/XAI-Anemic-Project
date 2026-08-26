# Anaemia Severity Dataset - README

## Project

XAI-Driven Anaemia Severity Stratification from Conjunctival Images
B.Tech CSE (AI & ML), BCSE497J - Project 1
Vellore Institute of Technology

## Team

- Aditya Rajeev Nair (23BAI0036)
- Agnik Patra (23BAI0001)
- Anasmita Bhattacharya (23BAI0028)

Guide: Prof. Prabin SM

## Google Drive Link

The full image dataset (folders and CSV) is available here:
https://drive.google.com/drive/folders/1XBTL86t9YB3qb5gVKlOrCmIuEHYQHUl5?usp=sharing

## What This Dataset Is

This is a unified dataset built by combining two public conjunctival image datasets, CP-AnemiC and Eyes-Defy-Anemia, into a single labeled collection for anaemia severity classification. Both source datasets contain images of the palpebral conjunctiva (the inside of the lower eyelid), which is used clinically as a visual indicator of anaemia.

The combined dataset has 909 usable images across four severity classes: Non-Anemic, Mild, Moderate, and Severe.

## Folder Structure

```
unified_dataset/
    Non-Anemic/
    Mild/
    Moderate/
    Severe/
    unified_dataset.csv
```

Each folder contains the conjunctival images belonging to that severity class. Images are named using the pattern `<source_dataset>_<image_id>.<extension>` so the original source is traceable from the filename alone.

The file `unified_dataset.csv` contains one row per image with all metadata (source dataset, subject ID, haemoglobin value, severity class, age, sex, population, mask paths where available).

## Source Datasets

### 1. CP-AnemiC

- A Conjunctival Pallor dataset collected in Ghana.
- Population: children aged 6 to 59 months.
- 710 images used in this unified dataset, split as Anemic and Non-anemic subfolders in the original release.
- Comes with an accompanying Excel sheet (Anemia_Data_Collection_Sheet.xlsx) containing IMAGE_ID, HB_LEVEL, Severity, Age (months), Gender, Remark, Hospital, City/Town, District, Region, and Country for each image.
- Severity labels (Non-Anemic, Mild, Moderate, Severe) are provided directly in the original dataset and were used as is, since they already follow WHO child severity thresholds.
- Images are already cropped to the conjunctival region.

### 2. Eyes-Defy-Anemia

- Collected from adult patients in two locations: Bari, Italy, and Chennai, India.
- 123 subjects from Italy and 95 subjects from India in the original release.
- Each subject folder contains the original eye photo plus hand-segmented crops: palpebral.png, forniceal.png, and forniceal_palpebral.png. A small number of subjects (folders 1, 35, 54, 58, 75, 109 in the Italy set) do not have a forniceal crop because the forniceal conjunctiva was not visible.
- Each location has its own Excel sheet (Italy.xlsx, India.xlsx) with Number, Hgb, Sex, Age, and a Notes column.
- 19 subjects in the Italy sheet had no recorded haemoglobin value and were excluded from this unified dataset. All 95 India subjects had valid haemoglobin values.
- Total usable after exclusion: 199 images (104 from Italy, 95 from India).
- The palpebral.png crop was used directly as the training image for each subject, since it is already a pre-segmented crop of the palpebral conjunctiva, matching the crop style of CP-AnemiC.
- This dataset does not come with severity labels, only continuous haemoglobin values. Severity labels were derived using WHO adult haemoglobin thresholds, applied separately for males and females, since the cutoffs differ by sex.

## How Severity Labels Were Assigned

CP-AnemiC already provides severity labels for each image (Non-Anemic, Mild, Moderate, Severe), based on WHO thresholds for children aged 6 to 59 months. These were used without modification.

Eyes-Defy-Anemia only provides a haemoglobin value in g/dL, along with the sex of each subject. Severity was computed using the WHO adult thresholds below.

Women (non-pregnant, age 15 and above):
- Non-Anemic: Hb 12.0 g/dL or above
- Mild: Hb 11.0 to 11.9 g/dL
- Moderate: Hb 8.0 to 10.9 g/dL
- Severe: Hb below 8.0 g/dL

Men (age 15 and above):
- Non-Anemic: Hb 13.0 g/dL or above
- Mild: Hb 11.0 to 12.9 g/dL
- Moderate: Hb 8.0 to 10.9 g/dL
- Severe: Hb below 8.0 g/dL

Since CP-AnemiC subjects are children (6 to 59 months) and Eyes-Defy-Anemia subjects are adults, two different WHO threshold tables were applied depending on the source dataset and, for adults, the recorded sex. This is an important detail because a "Moderate" label does not correspond to the exact same haemoglobin range across the two datasets, since the age groups use different clinical cutoffs.

## Final Class Distribution

| Severity | CP-AnemiC (children) | Eyes-Defy-Anemia (adults) | Total |
|----------|----------------------|----------------------------|-------|
| Non-Anemic | 286 | 109 | 395 |
| Mild | 144 | 36 | 180 |
| Moderate | 232 | 51 | 283 |
| Severe | 48 | 3 | 51 |
| Total | 710 | 199 | 909 |

## Known Limitations

1. Population mismatch: CP-AnemiC is entirely children under 5 years old from Ghana. Eyes-Defy-Anemia is entirely adults from Italy and India. A model trained on this combined dataset is learning across two different age groups and populations, which is a limitation to keep in mind when interpreting results or generalizing to new populations.

2. Class imbalance: the Severe class has only 51 images out of 909 total (about 5.6 percent), and 48 of those 51 come from CP-AnemiC alone. This means the Severe class is heavily dominated by one source dataset. Any model evaluation should report Severe-class performance separately by source dataset to check whether the model is learning actual pallor signals or just picking up on dataset-specific characteristics.

3. Different capture conditions: CP-AnemiC images were captured on regular phone cameras across different hospitals in Ghana. Eyes-Defy-Anemia images were captured using a Samsung S6 smartphone fitted with a device that magnifies the image and standardizes lighting. This difference in capture setup could introduce visual differences between the two sources that are unrelated to anaemia severity.

4. Missing data: 19 subjects from the Eyes-Defy-Anemia Italy set were excluded because they had no recorded haemoglobin value. 6 subjects in the same set do not have a forniceal crop available.

## Citations

If this dataset is used or referenced, please cite the two original sources:

CP-AnemiC:
P. Appiahene, E. T. Donkoh, G. Dimauro, and R. Maglietta, "CP-AnemiC: A conjunctival pallor dataset and benchmark for anemia detection in children," Data in Brief, vol. 49, 2023.

Eyes-Defy-Anemia:
R. Maglietta, M. E. Griseta, F. Clemente, A. Guarini, G. Dimauro, "Advances in the Automated Diagnosis of Anemia: a shared Eyes-defy-anemia dataset and a novel non-invasive system with machine learning."

G. Dimauro, D. Caivano, P. Di Pilato, A. Dipalma, M. G. Camporeale, "A Systematic Mapping Study on Research in Anemia Assessment with Non-Invasive Devices," Appl. Sci. 2020, 10, 4804.

G. Dimauro and L. Simone, "Novel Biased Normalized Cuts Approach for the Automatic Segmentation of the Conjunctiva," Electronics 2020, 9, 997.

## How the Data Was Prepared

The two source datasets were combined using a Python script that:

1. Reads CP-AnemiC's Excel sheet directly and keeps its existing severity labels.
2. Reads Eyes-Defy-Anemia's Italy and India Excel sheets, excludes subjects with missing haemoglobin values, and computes severity using the WHO adult thresholds shown above.
3. Copies each image into a folder named after its severity class, renaming it to include the source dataset name.
4. Writes out unified_dataset.csv with one row per image containing all metadata.

The full script used to build this dataset is available in the project repository.
