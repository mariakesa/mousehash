Across DANDI electrophysiology datasets:
- Which brain regions are most stimulus-decodable?
- Which modalities have the most predictable neural activity?
- Are behavioral variables more decodable than stimulus variables?
- Which encoding model family wins most often: linear, Poisson GLM, negative binomial, temporal model?
- Do spike-count models systematically outperform Gaussian models?
- Are some datasets “highly decodable but weakly encodable”?
- Which experiment types produce the most reproducible decoder confusion patterns?

MetaEncodingAtlas
Input:
  per-experiment CV scores
  model class
  target variable
  brain area
  species
  modality
  sampling/binning parameters

Output:
  cross-dataset effect-size table
  model ranking
  region/task/modality map
  confidence intervals

- How many dimensions does neural activity usually occupy in different brain areas?
- Are hippocampal recordings higher-dimensional than visual cortical recordings?
- Do task datasets have lower-dimensional structure than spontaneous datasets?
- Which experiments show smooth latent trajectories versus scattered state clouds?
- Are there recurring latent motifs across unrelated electrophysiology datasets?
- Which datasets have strong low-rank structure and are therefore good candidates for state-space modeling?
- Do Neuropixels recordings and intracellular recordings produce different dimensionality profiles?