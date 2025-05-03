#include "classifier.h"
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>

Classifier::Classifier(float learningRate, int maxIterations)
    : learningRate(learningRate), maxIterations(maxIterations), bias(0.0f)
{
}

bool Classifier::train(const std::vector<std::vector<float>> &features,
                       const std::vector<Label> &labels)
{
    if (features.empty() || features.size() != labels.size())
    {
        std::cerr << "Invalid training data" << std::endl;
        return false;
    }

    // Initialize weights
    int featureSize = features[0].size();
    weights = std::vector<float>(featureSize, 0.0f);

    // Initialize with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0f, 0.01f);

    for (float &w : weights)
    {
        w = distribution(gen);
    }

    // Train using stochastic gradient descent
    for (int iteration = 0; iteration < maxIterations; iteration++)
    {
        float totalLoss = 0.0f;

        // Shuffle indices
        std::vector<int> indices(features.size());
        for (size_t i = 0; i < indices.size(); i++)
        {
            indices[i] = static_cast<int>(i);
        }
        std::shuffle(indices.begin(), indices.end(), gen);

        // Update weights for each sample
        for (int idx : indices)
        {
            const std::vector<float> &x = features[idx];
            float y = static_cast<int>(labels[idx]);

            // Compute prediction
            float z = bias;
            for (size_t i = 0; i < weights.size(); i++)
            {
                z += weights[i] * x[i];
            }

            // Apply sigmoid function
            float pred = 1.0f / (1.0f + std::exp(-z));

            // Compute error
            float error = y - pred;
            totalLoss += -y * std::log(pred) - (1 - y) * std::log(1 - pred);

            // Update bias
            bias += learningRate * error;

            // Update weights
            for (size_t i = 0; i < weights.size(); i++)
            {
                weights[i] += learningRate * error * x[i];
            }
        }

        // Calculate average loss
        float avgLoss = totalLoss / features.size();

        // Print progress every 100 iterations
        if ((iteration + 1) % 100 == 0 || iteration == 0)
        {
            std::cout << "Iteration " << iteration + 1 << ", Loss: " << avgLoss << std::endl;
        }

        // Early stopping if loss is small enough
        if (avgLoss < 0.01f)
        {
            std::cout << "Converged after " << iteration + 1 << " iterations" << std::endl;
            break;
        }
    }

    return true;
}

Label Classifier::predict(const std::vector<float> &features) const
{
    if (features.size() != weights.size())
    {
        std::cerr << "Feature size mismatch" << std::endl;
        return Label::CAT; // Default prediction
    }

    // Compute linear combination
    float z = bias;
    for (size_t i = 0; i < weights.size(); i++)
    {
        z += weights[i] * features[i];
    }

    // Apply sigmoid function
    float probability = 1.0f / (1.0f + std::exp(-z));

    // Classify based on threshold
    return (probability >= 0.5f) ? Label::DOG : Label::CAT;
}

float Classifier::evaluate(const std::vector<std::vector<float>> &features,
                           const std::vector<Label> &labels) const
{
    if (features.empty() || features.size() != labels.size())
    {
        return 0.0f;
    }

    int correct = 0;
    for (size_t i = 0; i < features.size(); i++)
    {
        Label prediction = predict(features[i]);
        if (prediction == labels[i])
        {
            correct++;
        }
    }

    return static_cast<float>(correct) / features.size();
}

bool Classifier::saveModel(const std::string &filename) const
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return false;
    }

    // Save bias
    file.write(reinterpret_cast<const char *>(&bias), sizeof(bias));

    // Save weights size
    size_t size = weights.size();
    file.write(reinterpret_cast<const char *>(&size), sizeof(size));

    // Save weights
    file.write(reinterpret_cast<const char *>(weights.data()), sizeof(float) * weights.size());

    return !file.fail();
}

bool Classifier::loadModel(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return false;
    }

    // Load bias
    file.read(reinterpret_cast<char *>(&bias), sizeof(bias));

    // Load weights size
    size_t size;
    file.read(reinterpret_cast<char *>(&size), sizeof(size));

    // Load weights
    weights.resize(size);
    file.read(reinterpret_cast<char *>(weights.data()), sizeof(float) * size);

    return !file.fail();
}