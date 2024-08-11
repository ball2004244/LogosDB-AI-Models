#include <Python.h>
#include <iostream>
#include <vector>
#include <string>

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

int main() {
    try {
        initialize_python();

        std::vector<std::string> inputs = {
            "Gravitational waves, ripples in the fabric of spacetime, were first predicted by Einstein's theory of general relativity but remained undetected for a century. In 2015, the Laser Interferometer Gravitational-Wave Observatory (LIGO) made history by directly observing gravitational waves from a binary black hole merger. This breakthrough opened a new era of gravitational wave astronomy, allowing scientists to \"hear\" cosmic events like colliding black holes and neutron stars. These waves carry unique information about their sources and offer insights into the nature of gravity, the properties of matter at extreme densities, and the evolution of the universe. Ongoing improvements in detector sensitivity promise to reveal an ever-richer cosmos of gravitational wave sources.",
            "The concept of entropy in physics describes the degree of disorder or randomness in a system, playing a crucial role in thermodynamics and our understanding of the universe's evolution. Originating from the study of heat engines in the 19th century, entropy has far-reaching implications, from explaining why time seems to flow in one direction to predicting the ultimate fate of the universe. The Second Law of Thermodynamics states that the total entropy of an isolated system always increases over time, leading to the idea of the \"heat death\" of the universe - a state of maximum entropy where no useful energy remains. This principle underlies many natural processes and has applications in fields ranging from chemistry to information theory.",
            "String theory proposes that the fundamental constituents of the universe are tiny, vibrating strings of energy. This elegant concept attempts to unify quantum mechanics and general relativity, potentially resolving long-standing conflicts between these theories. String theory suggests that our universe has additional spatial dimensions beyond the three we experience, curled up at microscopic scales. These extra dimensions could explain the apparent weakness of gravity compared to other fundamental forces. Despite its mathematical beauty, string theory remains controversial due to the lack of experimental evidence. Critics argue that its predictions may be untestable, while proponents believe it offers the best path towards a \"theory of everything.\"",
            "Quantum tunneling is a fascinating phenomenon where particles can pass through barriers that classical physics deems impassable. This effect arises from the wave-like nature of matter at the quantum scale, where particles behave probabilistically rather than deterministically. Quantum tunneling plays a crucial role in various natural processes, from nuclear fusion in stars to the operation of scanning tunneling microscopes. It also enables technologies like flash memory in computers and quantum tunneling diodes. Interestingly, quantum tunneling contributes to radioactive decay and even suggests the possibility, albeit extremely unlikely, of macroscopic objects tunneling through solid barriers, challenging our intuitive understanding of reality.",
            "The Casimir effect, predicted by Dutch physicist Hendrik Casimir in 1948, demonstrates that empty space is not truly empty but filled with fluctuating quantum fields. This effect manifests as a tiny attractive force between two parallel conducting plates in a vacuum, caused by the exclusion of certain wavelengths of virtual particles between the plates. The Casimir effect has been experimentally verified and has implications for nanotechnology, where it can cause microscopic components to stick together. It also plays a role in some theories of the cosmological constant and vacuum energy, potentially contributing to our understanding of dark energy and the expansion of the universe."
        };

        std::vector<std::string> summaries = mass_extract_summaries(inputs);

        for (const auto& summary : summaries) {
            std::cout << "Summary: " << summary << std::endl;
        }

        finalize_python();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        finalize_python();
        return 1;
    }

    return 0;
}