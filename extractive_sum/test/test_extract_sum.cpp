#include <Python.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

/*
Compile and run C++ code with:
g++ -o summarize test_extract_sum.cpp -I/usr/include/python3.10 -lpython3.10
./summarize
*/

// Function to initialize the Python interpreter and import the Cython module
void initialize_python() {
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('.')");
}

// Function to finalize the Python interpreter
void finalize_python() {
    Py_Finalize();
}

// Function to call the mass_extract_summaries function from the Cython module
std::vector<std::string> mass_extract_summaries(const std::vector<std::string>& inputs) {
    std::vector<std::string> summaries;

    // Import the Cython module
    PyObject* pModule = PyImport_ImportModule("extract_sum_mp");
    if (!pModule) {
        PyErr_Print();
        throw std::runtime_error("Failed to load extract_sum_mp module");
    }

    // Get the mass_extract_summaries function
    PyObject* pFunc = PyObject_GetAttrString(pModule, "mass_extract_summaries");
    if (!pFunc || !PyCallable_Check(pFunc)) {
        PyErr_Print();
        throw std::runtime_error("Failed to load mass_extract_summaries function");
    }

    // Convert the inputs to a Python list
    PyObject* pInputs = PyList_New(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        PyObject* pValue = PyUnicode_FromString(inputs[i].c_str());
        PyList_SetItem(pInputs, i, pValue);
    }

    // Call the mass_extract_summaries function
    PyObject* pResult = PyObject_CallObject(pFunc, PyTuple_Pack(1, pInputs));
    if (!pResult) {
        PyErr_Print();
        throw std::runtime_error("Failed to call mass_extract_summaries function");
    }

    // Convert the result to a C++ vector
    for (Py_ssize_t i = 0; i < PyList_Size(pResult); ++i) {
        PyObject* pItem = PyList_GetItem(pResult, i);
        summaries.push_back(PyUnicode_AsUTF8(pItem));
    }

    // Clean up
    Py_DECREF(pInputs);
    Py_DECREF(pResult);
    Py_DECREF(pFunc);
    Py_DECREF(pModule);

    return summaries;
}

void process_csv() {
    initialize_python();

    std::string input_file = "../data/biology.csv";
    std::string output_file = "./summary_output.txt";
    std::string time_file = "./time_log.txt";

    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "Failed to open input file: " << input_file << std::endl;
        return;
    }

    std::vector<std::string> user_inputs;
    std::string line;
    std::getline(infile, line); // Skip header line
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string empty_col, id, question, answer, keywords;
        std::getline(ss, empty_col, ','); // Ignore the first column
        std::getline(ss, id, ',');
        std::getline(ss, question, ',');
        std::getline(ss, answer, ',');
        std::getline(ss, keywords, ',');

        user_inputs.push_back(answer);
    }
    infile.close();

    std::ofstream timefile(time_file);
    if (!timefile.is_open()) {
        std::cerr << "Failed to open time file: " << time_file << std::endl;
        return;
    }

    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }

    std::cout << "Processing all rows" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < user_inputs.size(); i += 10000) {
        size_t end = std::min(i + 10000, user_inputs.size());
        std::vector<std::string> batch(user_inputs.begin() + i, user_inputs.begin() + end);
        std::vector<std::string> summaries = mass_extract_summaries(batch);

        // Write summaries to the output file
        for (const auto& summary : summaries) {
            outfile << summary << std::endl;
        }

        auto elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed_time).count();
        timefile << "Processed " << end << " rows in " << elapsed_seconds << " seconds" << std::endl;
        std::cout << "Processed " << end << " rows in " << elapsed_seconds << " seconds" << std::endl;
    }
    auto total_elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
    double total_elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(total_elapsed_time).count();
    timefile << "Total processing time: " << total_elapsed_seconds << " seconds" << std::endl;
    std::cout << "Total processing time: " << total_elapsed_seconds << " seconds" << std::endl;

    timefile.close();
    outfile.close();
    finalize_python();
    std::cout << "All done!" << std::endl;
}

int main() {
    try {
        process_csv();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        finalize_python();
        return 1;
    }

    return 0;
}