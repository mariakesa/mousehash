mousehash_roles:
  neural_data:
    - spikes
    - lfp
    - eeg
    - calcium
    - photometry
    - images
    - current clamp
    - voltage clamp
    - patch clamp

  stimuli:
    sensory:
      - visual
      - auditory
      - tactile
      - odor
    interventions:
      - optogenetic
      - electrical
      - pharmacological
      - anesthesia

  behavior:
    - choices
    - reaction_times
    - pose
    - locomotion
    - pupil
    - kinematics
    - behavioral_states
    - running

  conditions:
    - task_labels
    - trial_labels
    - experimental_groups
    - brain_states
    - session_phases
    - perturbation_labels
    - session_phases (sample, delay, response)

  time_organization:
    - continuous_time
    - trials
    - epochs
    - events
    - frames
    - alignment_rules

  metadata:
    - subject
    - species
    - genotype
    - session
    - brain_area
    - probe/electrode/imaging_plane
    - acquisition_device
    - preprocessing_info