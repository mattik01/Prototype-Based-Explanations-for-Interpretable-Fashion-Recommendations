"""Prototype naming: derive human-readable names for prototypes from the features of
their representative items.

Architecture (see plan): a *source* produces a ProtoFeatureProfile, the *namer* turns it
into a name + transparent stats. The derived (top-k) source lives here; a future intrinsic
source can reuse everything from ProtoFeatureProfile onward.
"""
from utilities.explanations.naming.config import (
    NAMING_DEFAULTS,
    FeatureSpec,
    NamingConfig,
    attr_naming_config,
    get_naming_config,
    intrinsic_naming_config,
)
from utilities.explanations.naming.describers import FeatureValueStat
from utilities.explanations.naming.namer import (
    NamingResult,
    PrototypeName,
    name_from_profile,
    name_prototypes_from_score_matrix,
    name_prototypes_from_weights,
    name_prototypes_intrinsic,
)
from utilities.explanations.naming.profile import ProtoFeatureProfile

__all__ = [
    "NamingConfig",
    "FeatureSpec",
    "NAMING_DEFAULTS",
    "get_naming_config",
    "intrinsic_naming_config",
    "attr_naming_config",
    "FeatureValueStat",
    "ProtoFeatureProfile",
    "PrototypeName",
    "NamingResult",
    "name_from_profile",
    "name_prototypes_from_weights",
    "name_prototypes_from_score_matrix",
    "name_prototypes_intrinsic",
]
