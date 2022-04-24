# AnemiaFeatureSelect
An implementation of feature selection for anemia detection. Raj Reddy Center for Technology and Society (RCTS) 
## Inputs: 
1. **50 x 10 resized image** for best results (showed best results as demonstrated in C.C. Fan, Application of artificial classification techniques to assess the anemia conditions via 
palpebral conjunctiva color component, Master Thesis, National Ilan University, Taiwan, 
(2013).)
2. **Quick or Robust Algorithm**
3. **Threshold** for HHR based on the image selected

## Outputs:
1. **Quick algorithm**: HHR value and PVM for RGB channel
2. **Robust algorithm**: Grayscale histogram equalization to extract features like arteries through the contrast.

Feature selection for anemia detection and classification based on conjunctiva pallor.
Current features being identified are based on those highlighted in  Yi-Ming Chen , Shaou-Gang Miaou , Hongyu Bian , “Examining palpebral conjunctiva for anemia assessment with image processing methods,” Journal of Computer Methods And Programs In Biomedicine -125–135, Elsevier,2016

