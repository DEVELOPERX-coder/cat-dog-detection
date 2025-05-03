#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <random>
#include <algorithm>
#include "classifier.h"

namespace fs = std::filesystem;

// Function to load all images from a directory (support for JPG files)
bool loadImagesFromDirectory(const std::string &directory,
                             std::vector<std::string> &imagePaths,
                             std::vector<Label> &labels)
{
    try
    {
        for (const auto &entry : fs::directory_iterator(directory + "/cats"))
        {
            // Look for JPG and JPEG files
            if (entry.path().extension() == ".jpg" ||
                entry.path().extension() == ".jpeg")
            {
                imagePaths.push_back(entry.path().string());
                labels.push_back(Label::CAT);
            }
        }

        for (const auto &entry : fs::directory_iterator(directory + "/dogs"))
        {
            // Look for JPG and JPEG files
            if (entry.path().extension() == ".jpg" ||
                entry.path().extension() == ".jpeg")
            {
                imagePaths.push_back(entry.path().string());
                labels.push_back(Label::DOG);
            }
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading images: " << e.what() << std::endl;
        return false;
    }
}

// Function to shuffle data (unchanged)
void shuffleData(std::vector<std::string> &imagePaths, std::vector<Label> &labels)
{
    // Create indices
    std::vector<size_t> indices(imagePaths.size());
    for (size_t i = 0; i < indices.size(); i++)
    {
        indices[i] = i;
    }

    // Shuffle indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Reorder data
    std::vector<std::string> shuffledPaths(imagePaths.size());
    std::vector<Label> shuffledLabels(labels.size());

    for (size_t i = 0; i < indices.size(); i++)
    {
        shuffledPaths[i] = imagePaths[indices[i]];
        shuffledLabels[i] = labels[indices[i]];
    }

    imagePaths = shuffledPaths;
    labels = shuffledLabels;
}

#endif // UTILS_H