# Label axes: the per-sample attributes of a dataset that say what a sample is (targets, domains, super_classes, ...).
# A dataset has one value per sample on each axis it carries, and the names of the values where it has them
# (classes for targets, domain_names for domains). This module imports nothing, so that what only reads labels
# (the samplers) does not need what the datasets need.

# The value of an axis for a sample that has no label on it (e.g. the domain of a test sample).
UNLABELED = -1
