#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <random>
#include "image_processor.h"
#include "feature_extractor.h"
#include "classifier.h"
#include "utils.h"

// Function to extract features from an image
std::vector<float> processImage(const std::string &imagePath, int targetWidth, int targetHeight)
{
    ImageProcessor processor;
    if (!processor.loadImage(imagePath))
    {
        return {};
    }

    if (!processor.resize(targetWidth, targetHeight))
    {
        return {};
    }

    std::vector<float> imageData = processor.getLinearizedData();

    FeatureExtractor extractor;
    return extractor.extractFeatures(imageData, targetWidth, targetHeight);
}

// Function to train the model
bool trainModel(const std::string &trainingDir, const std::string &modelPath)
{
    const int IMAGE_WIDTH = 64;
    const int IMAGE_HEIGHT = 64;

    std::cout << "Loading training images..." << std::endl;

    // Load image paths and labels
    std::vector<std::string> imagePaths;
    std::vector<Label> labels;

    if (!loadImagesFromDirectory(trainingDir, imagePaths, labels))
    {
        std::cerr << "Failed to load training images" << std::endl;
        return false;
    }

    if (imagePaths.empty())
    {
        std::cerr << "No images found in " << trainingDir << std::endl;
        return false;
    }

    std::cout << "Found " << imagePaths.size() << " images for training" << std::endl;

    // Shuffle data
    shuffleData(imagePaths, labels);

    // Extract features
    std::cout << "Extracting features from training images..." << std::endl;
    std::vector<std::vector<float>> features;

    for (size_t i = 0; i < imagePaths.size(); i++)
    {
        std::cout << "Processing image " << i + 1 << "/" << imagePaths.size()
                  << ": " << imagePaths[i] << std::endl;

        std::vector<float> imageFeatures = processImage(imagePaths[i], IMAGE_WIDTH, IMAGE_HEIGHT);

        if (!imageFeatures.empty())
        {
            features.push_back(imageFeatures);
        }
        else
        {
            // Remove the corresponding label if feature extraction failed
            labels.erase(labels.begin() + i);
            imagePaths.erase(imagePaths.begin() + i);
            i--;
        }
    }

    std::cout << "Successfully extracted features from " << features.size() << " images" << std::endl;

    if (features.empty())
    {
        std::cerr << "No valid features extracted" << std::endl;
        return false;
    }

    // Train the classifier
    std::cout << "Training the classifier..." << std::endl;
    Classifier classifier(0.01f, 2000);

    auto start = std::chrono::high_resolution_clock::now();

    if (!classifier.train(features, labels))
    {
        std::cerr << "Failed to train the classifier" << std::endl;
        return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Training completed in " << elapsed.count() << " seconds" << std::endl;

    // Calculate training accuracy
    float accuracy = classifier.evaluate(features, labels);
    std::cout << "Training accuracy: " << accuracy * 100.0f << "%" << std::endl;

    // Save the model
    std::cout << "Saving model to " << modelPath << std::endl;
    if (!classifier.saveModel(modelPath))
    {
        std::cerr << "Failed to save the model" << std::endl;
        return false;
    }

    return true;
}

// Function to test the model
bool testModel(const std::string &testingDir, const std::string &modelPath)
{
    const int IMAGE_WIDTH = 64;
    const int IMAGE_HEIGHT = 64;

    std::cout << "Loading model from " << modelPath << std::endl;

    // Load the classifier
    Classifier classifier;
    if (!classifier.loadModel(modelPath))
    {
        std::cerr << "Failed to load the model" << std::endl;
        return false;
    }

    // Load test images
    std::cout << "Loading test images..." << std::endl;
    std::vector<std::string> imagePaths;
    std::vector<Label> labels;

    if (!loadImagesFromDirectory(testingDir, imagePaths, labels))
    {
        std::cerr << "Failed to load test images" << std::endl;
        return false;
    }

    if (imagePaths.empty())
    {
        std::cerr << "No images found in " << testingDir << std::endl;
        return false;
    }

    std::cout << "Found " << imagePaths.size() << " images for testing" << std::endl;

    // Extract features and evaluate
    std::cout << "Testing the model..." << std::endl;
    int correct = 0;

    for (size_t i = 0; i < imagePaths.size(); i++)
    {
        std::cout << "Processing image " << i + 1 << "/" << imagePaths.size()
                  << ": " << imagePaths[i] << std::endl;

        std::vector<float> features = processImage(imagePaths[i], IMAGE_WIDTH, IMAGE_HEIGHT);

        if (features.empty())
        {
            std::cerr << "Failed to extract features from " << imagePaths[i] << std::endl;
            continue;
        }

        Label prediction = classifier.predict(features);
        bool isCorrect = (prediction == labels[i]);

        std::cout << "Prediction: " << (prediction == Label::CAT ? "CAT" : "DOG")
                  << ", Actual: " << (labels[i] == Label::CAT ? "CAT" : "DOG")
                  << ", " << (isCorrect ? "CORRECT" : "WRONG") << std::endl;

        if (isCorrect)
        {
            correct++;
        }
    }

    // Calculate and display accuracy
    float accuracy = static_cast<float>(correct) / imagePaths.size();
    std::cout << "Test accuracy: " << accuracy * 100.0f << "% ("
              << correct << "/" << imagePaths.size() << ")" << std::endl;

    return true;
}

// Function to classify a single image
bool classifySingleImage(const std::string &imagePath, const std::string &modelPath)
{
    const int IMAGE_WIDTH = 64;
    const int IMAGE_HEIGHT = 64;

    // Load the classifier
    Classifier classifier;
    if (!classifier.loadModel(modelPath))
    {
        std::cerr << "Failed to load the model" << std::endl;
        return false;
    }

    // Process the image
    std::vector<float> features = processImage(imagePath, IMAGE_WIDTH, IMAGE_HEIGHT);

    if (features.empty())
    {
        std::cerr << "Failed to extract features from " << imagePath << std::endl;
        return false;
    }

    // Make prediction
    Label prediction = classifier.predict(features);

    std::cout << "Image: " << imagePath << std::endl;
    std::cout << "Prediction: " << (prediction == Label::CAT ? "CAT" : "DOG") << std::endl;

    return true;
}

// Main function
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "  " << argv[0] << " train <training_dir> <model_path>" << std::endl;
        std::cout << "  " << argv[0] << " test <testing_dir> <model_path>" << std::endl;
        std::cout << "  " << argv[0] << " classify /path/to/image.jpg model.bin" << std::endl;
        return 1;
    }

    std::string command = argv[1];

    if (command == "train")
    {
        if (argc < 4)
        {
            std::cerr << "Missing arguments for train command" << std::endl;
            return 1;
        }

        std::string trainingDir = argv[2];
        std::string modelPath = argv[3];

        if (!trainModel(trainingDir, modelPath))
        {
            return 1;
        }
    }
    else if (command == "test")
    {
        if (argc < 4)
        {
            std::cerr << "Missing arguments for test command" << std::endl;
            return 1;
        }

        std::string testingDir = argv[2];
        std::string modelPath = argv[3];

        if (!testModel(testingDir, modelPath))
        {
            return 1;
        }
    }
    else if (command == "classify")
    {
        if (argc < 4)
        {
            std::cerr << "Missing arguments for classify command" << std::endl;
            return 1;
        }

        std::string imagePath = argv[2];
        std::string modelPath = argv[3];

        if (!classifySingleImage(imagePath, modelPath))
        {
            return 1;
        }
    }
    else
    {
        std::cerr << "Unknown command: " << command << std::endl;
        return 1;
    }

    return 0;
}