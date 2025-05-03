#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <vector>
#include <string>

class FeatureExtractor
{
public:
    FeatureExtractor();

    // Extract HOG (Histogram of Oriented Gradients) features
    std::vector<float> extractHOGFeatures(const std::vector<float> &image, int width, int height) const;

    // Extract simple edge features using Sobel operator
    std::vector<float> extractEdgeFeatures(const std::vector<float> &image, int width, int height) const;

    // Extract pixel intensity histograms
    std::vector<float> extractHistogramFeatures(const std::vector<float> &image, int numBins = 16) const;

    // Combine all features
    std::vector<float> extractFeatures(const std::vector<float> &image, int width, int height) const;
};

#endif // FEATURE_EXTRACTOR_H