## RSAAnalysis

This directory holds repo-level prototype scripts for standardized RSA paths.

The first path implemented here targets the easiest family currently available
in the indexed DANDI manifests:

- spike-based datasets
- trial structure present
- categorical trial labels available

The prototype uses the existing MouseHash manifest and readiness library to:

1. load an `EvidenceBackedRoleManifest`
2. verify the dataset is compatible with a spike-trial RSA path
3. extract trial-level spike counts from a fixed response window
4. build a categorical target RDM from a trial label column
5. compute RSA with a permutation test

Files:

- `spike_trial_rsa.py`: run the standardized prototype on one manifest/NWB pair
- `batch_spike_trial_rsa.py`: apply the prototype across many ready manifests

Typical usage:

```bash
/home/maria/mousehash/.venv/bin/python -m RSAAnalysis.spike_trial_rsa \
  --manifest-path /media/maria/notsudata/MousehashManifests/000006_0.220126.1855.manifest.json \
  --output-dir /tmp/rsa_single

/home/maria/mousehash/.venv/bin/python -m RSAAnalysis.batch_spike_trial_rsa \
  --ready-json /media/maria/notsudata/MousehashManifests/rsa_ready_experiments.json \
  --output-dir /tmp/rsa_batch \
  --max-experiments 25
```
