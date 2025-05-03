#include "feature_extractor.h"
#include <cmath>
#include <algorithm>
#include <iostream>

FeatureExtractor::FeatureExtractor()
{
}

std::vector<float> FeatureExtractor::extractHOGFeatures(const std::vector<float> &image, int width, int height) const
{
    // Parameters
    const int cellSize = 8;  // Size of cell (in pixels)
    const int blockSize = 2; // Size of block (in cells)
    const int numBins = 9;   // Number of orientation bins

    // Calculate the number of cells in each dimension
    int numCellsX = width / cellSize;
    int numCellsY = height / cellSize;

    // Allocate memory for gradients and histograms
    std::vector<float> gradientMagnitudes((width - 2) * (height - 2));
    std::vector<float> gradientOrientations((width - 2) * (height - 2));
    std::vector<float> cellHistograms(numCellsX * numCellsY * numBins, 0.0f);

    // Step 1: Calculate gradients
    // For each pixel (except borders)
    int g = 0;
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            // Calculate dx using horizontal Sobel operator [-1, 0, 1]
            float dx = image[y * width + x + 1] - image[y * width + x - 1];

            // Calculate dy using vertical Sobel operator [-1, 0, 1]^T
            float dy = image[(y + 1) * width + x] - image[(y - 1) * width + x];

            // Calculate magnitude and orientation
            float magnitude = std::sqrt(dx * dx + dy * dy);
            float orientation = std::atan2(dy, dx);

            // Convert orientation to degrees (0-180)
            // Unsigned orientation (negate the sign)
            orientation = orientation * 180.0f / M_PI;
            if (orientation < 0)
            {
                orientation += 180.0f;
            }

            // Store gradient information
            gradientMagnitudes[g] = magnitude;
            gradientOrientations[g] = orientation;
            g++;
        }
    }

    // Step 2: Calculate cell histograms
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            // Skip boundary pixels
            if (x < cellSize / 2 || y < cellSize / 2 ||
                x >= width - cellSize / 2 || y >= height - cellSize / 2)
            {
                continue;
            }

            // Get the gradient
            int gradientIndex = (y - 1) * (width - 2) + (x - 1);
            float magnitude = gradientMagnitudes[gradientIndex];
            float orientation = gradientOrientations[gradientIndex];

            // Determine the cell this pixel belongs to
            int cellX = (x - cellSize / 2) / cellSize;
            int cellY = (y - cellSize / 2) / cellSize;

            if (cellX < 0 || cellX >= numCellsX || cellY < 0 || cellY >= numCellsY)
            {
                continue;
            }

            // Determine the bin
            float binSize = 180.0f / numBins;
            int bin = static_cast<int>(orientation / binSize);
            if (bin >= numBins)
                bin = numBins - 1;

            // Add to the histogram
            cellHistograms[(cellY * numCellsX + cellX) * numBins + bin] += magnitude;
        }
    }

    // Step 3: Normalize histograms in blocks
    std::vector<float> hogFeatures;
    hogFeatures.reserve(numCellsX * numCellsY * numBins);

    for (int y = 0; y <= numCellsY - blockSize; y++)
    {
        for (int x = 0; x <= numCellsX - blockSize; x++)
        {
            // Extract block
            std::vector<float> blockHistogram;
            blockHistogram.reserve(blockSize * blockSize * numBins);

            for (int cy = 0; cy < blockSize; cy++)
            {
                for (int cx = 0; cx < blockSize; cx++)
                {
                    int cellIndex = ((y + cy) * numCellsX + (x + cx)) * numBins;
                    blockHistogram.insert(blockHistogram.end(),
                                          cellHistograms.begin() + cellIndex,
                                          cellHistograms.begin() + cellIndex + numBins);
                }
            }

            // Normalize block (L2-norm)
            float sum = 0.0f;
            for (float val : blockHistogram)
            {
                sum += val * val;
            }

            float normFactor = 1.0f / (std::sqrt(sum) + 1e-5f);

            for (float &val : blockHistogram)
            {
                val *= normFactor;
            }

            // Add to feature vector
            hogFeatures.insert(hogFeatures.end(), blockHistogram.begin(), blockHistogram.end());
        }
    }

    return hogFeatures;
}

std::vector<float> FeatureExtractor::extractEdgeFeatures(const std::vector<float> &image, int width, int height) const
{
    // A simplified edge detector using Sobel operators

    // Parameters
    const int numBins = 8; // Number of edge orientation bins

    // Output feature vector (edge orientation histogram)
    std::vector<float> edgeHistogram(numBins, 0.0f);

    // For each pixel (except borders)
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            // Calculate dx using horizontal Sobel operator [-1, 0, 1]
            float dx = image[y * width + x + 1] - image[y * width + x - 1];

            // Calculate dy using vertical Sobel operator [-1, 0, 1]^T
            float dy = image[(y + 1) * width + x] - image[(y - 1) * width + x];

            // Calculate magnitude and orientation
            float magnitude = std::sqrt(dx * dx + dy * dy);

            // Skip weak edges
            if (magnitude < 0.1f)
            {
                continue;
            }

            float orientation = std::atan2(dy, dx);

            // Convert orientation to degrees (0-180)
            orientation = orientation * 180.0f / M_PI;
            if (orientation < 0)
            {
                orientation += 180.0f;
            }

            // Determine the bin
            float binSize = 180.0f / numBins;
            int bin = static_cast<int>(orientation / binSize);
            if (bin >= numBins)
                bin = numBins - 1;

            // Add to the histogram (weighted by magnitude)
            edgeHistogram[bin] += magnitude;
        }
    }

    // Normalize histogram
    float sum = 0.0f;
    for (float val : edgeHistogram)
    {
        sum += val;
    }

    if (sum > 0.0f)
    {
        for (float &val : edgeHistogram)
        {
            val /= sum;
        }
    }

    return edgeHistogram;
}

std::vector<float> FeatureExtractor::extractHistogramFeatures(const std::vector<float> &image, int numBins) const
{
    // Create histogram of pixel intensities
    std::vector<float> histogram(numBins, 0.0f);

    for (float pixelValue : image)
    {
        int bin = static_cast<int>(pixelValue * numBins);
        if (bin >= numBins)
            bin = numBins - 1;

        histogram[bin] += 1.0f;
    }

    // Normalize histogram
    float sum = 0.0f;
    for (float val : histogram)
    {
        sum += val;
    }

    if (sum > 0.0f)
    {
        for (float &val : histogram)
        {
            val /= sum;
        }
    }

    return histogram;
}

std::vector<float> FeatureExtractor::extractFeatures(const std::vector<float> &image, int width, int height) const
{
    // Extract HOG features
    std::vector<float> hogFeatures = extractHOGFeatures(image, width, height);

    // Extract edge features
    std::vector<float> edgeFeatures = extractEdgeFeatures(image, width, height);

    // Extract histogram features
    std::vector<float> histFeatures = extractHistogramFeatures(image, 16);

    // Combine all features
    std::vector<float> allFeatures;
    allFeatures.reserve(hogFeatures.size() + edgeFeatures.size() + histFeatures.size());

    allFeatures.insert(allFeatures.end(), hogFeatures.begin(), hogFeatures.end());
    allFeatures.insert(allFeatures.end(), edgeFeatures.begin(), edgeFeatures.end());
    allFeatures.insert(allFeatures.end(), histFeatures.begin(), histFeatures.end());

    return allFeatures;
}