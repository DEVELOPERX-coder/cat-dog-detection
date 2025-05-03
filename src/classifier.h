#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include <string>

enum class Label
{
    CAT = 0,
    DOG = 1
};

class Classifier
{
private:
    std::vector<float> weights;
    float bias;

    // Learning parameters
    float learningRate;
    int maxIterations;

public:
    Classifier(float learningRate = 0.01f, int maxIterations = 1000);

    // Train the classifier with labeled data
    bool train(const std::vector<std::vector<float>> &features,
               const std::vector<Label> &labels);

    // Make a prediction
    Label predict(const std::vector<float> &features) const;

    // Calculate accuracy on test data
    float evaluate(const std::vector<std::vector<float>> &features,
                   const std::vector<Label> &labels) const;

    // Save/load model
    bool saveModel(const std::string &filename) const;
    bool loadModel(const std::string &filename);
};

#endif // CLASSIFIER_H