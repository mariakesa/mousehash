from mousehash.schema.representations import RepresentationSpec, AnimateInanimateRule
from mousehash.schema.decompositions import DecompositionSpec

RepresentationSpec.insert1(
    dict(
        representation_spec_id="vit_b16_imagenet_cpu",
        model_name="google/vit-base-patch16-224",
        model_family="vit",
        feature_space="imagenet_classifier_output",
        preprocessing="imagenet_default",
        batch_size=16,
        device="cpu",
    ),
    skip_duplicates=True,
)

AnimateInanimateRule.insert1(
    dict(
        rule_id="imagenet_top1_leq_397",
        rule_name="Animate if top1 <= 397",
        threshold_max_class_idx=397,
        description="1 = animate, 0 = inanimate",
    ),
    skip_duplicates=True,
)

DecompositionSpec.insert1(
    dict(
        decomposition_spec_id="pca_logits_10_exploratory",
        method="pca",
        input_kind="logits",
        n_components=10,
        normalize_input=True,
        mode="exploratory",
    ),
    skip_duplicates=True,
)

DecompositionSpec.insert1(
    dict(
        decomposition_spec_id="nmf_probs_10_exploratory",
        method="nmf",
        input_kind="probabilities",
        n_components=10,
        normalize_input=False,
        mode="exploratory",
        nmf_temperature=1.0,
    ),
    skip_duplicates=True,
)

print("Seeded lookup tables.")