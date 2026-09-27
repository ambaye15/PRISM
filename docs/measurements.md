# Measurement contract

This document states the definitions that code, manuscript, and validation must share.

## Intensity

1. Estimate consecutive-frame DIS optical flow.
2. Remove a RANSAC background similarity transform on real videos, or the known simulator transform in synthetic validation.
3. Retain source pixels whose flow endpoints remain inside the next tracked mask and pass forward/backward consistency.
4. Estimate bulk body translation as the median vector in the central elliptical torso proxy.
5. Subtract this vector from all retained subject-flow vectors.
6. Divide vector magnitudes by smoothed mask height and frame interval.
7. Take the spatial 75th percentile per frame pair and temporal median per bout.

## Frequency

The primary estimator forms the body-area-normalized Euclidean distance matrix
between masks. Its lag profile is

$$
s[L] = mean_i D(i, i + L).
$$

## Extent

The primary extent is the RMS radius of the unnormalized sliding-window mask orbit. Given its Euclidean distance matrix `D`,

$$
radius = sqrt(mean(D**2) / 2).
$$

## Periodicity

The sliding-window cloud is centered over time within each window and normalized to the unit sphere. Let the two longest finite H1 classes have births and deaths `(b1, d1)` and `(b2, d2)`, ordered by persistence `mi = di - bi`. PRISM uses

$$
PS = exp(-b1 / 2) * m1 / sqrt(3) - exp(-b2 / 2) * m2 / sqrt(3).
$$

The second term is zero when a second finite class is absent. The first term rewards a dominant persistent loop; the second penalizes a competing loop of comparable persistence.
