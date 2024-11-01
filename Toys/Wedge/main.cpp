#include <iostream>
#include <Python.h>
#include <vector>
#include <string>

struct Point3D {
    double x, y, z;
};

std::vector<Point3D> extract_points(const std::string& file_path, int n_points) {
    std::vector<Point3D> points;

    // Initialize the Python interpreter
    Py_Initialize();

    // Ensure the Python interpreter can find your module
    PyObject* sysPath = PySys_GetObject("path");
    PyObject* currentDir = PyUnicode_FromString(".");  // Add current directory to Python path
    PyList_Append(sysPath, currentDir);
    Py_DECREF(currentDir);

    // Debugging: Print sys.path to ensure current directory is added
    PyRun_SimpleString("import sys; print('sys.path:', sys.path)");

    // Import the compiled module
    PyObject* pName = PyUnicode_FromString("process");  // Name of the .so file (without the extension)
    PyObject* pModule = PyImport_Import(pName);
    if (!pModule) {
        PyErr_Print();
        std::cerr << "Failed to load module 'process'" << std::endl;
    }

    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Get the function from the module
        PyObject* pFunc = PyObject_GetAttrString(pModule, "read_vtk_and_extract_points");
        if (pFunc && PyCallable_Check(pFunc)) {
            // Prepare arguments for the Python function
            PyObject* pArgs = PyTuple_Pack(2, PyUnicode_FromString(file_path.c_str()), PyLong_FromLong(n_points));
            // Call the function
            PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            // Check if function call was successful
            if (pValue != nullptr) {
                int SizeOfList = PyObject_Size(pValue);
                std::cout << "Logging: Size of list is " << SizeOfList << std::endl;
                for (Py_ssize_t i = 0; i < PyObject_Size(pValue); i++) {
                    PyObject* pItem = PyList_GetItem(pValue, i);
                    double x = PyFloat_AsDouble(PyList_GetItem(pItem, 0));
                    double y = PyFloat_AsDouble(PyList_GetItem(pItem, 1));
                    double z = PyFloat_AsDouble(PyList_GetItem(pItem, 2));
                    points.emplace_back(Point3D{x, y, z});
                }
                Py_DECREF(pValue);
            } else {
                PyErr_Print();
                std::cerr << "Failed to call function." << std::endl;
            }
            Py_DECREF(pFunc);
        } else {
            PyErr_Print();
            std::cerr << "Function not found or not callable." << std::endl;
        }
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        std::cerr << "Failed to load module." << std::endl;
    }

    // Finalize the Python interpreter
    Py_Finalize();

    return points;
}

int main() {
    const std::string file_path = "./wedge.vtk";  // Provide the actual path
    int n_points = 16;
    std::vector<Point3D> points = extract_points(file_path, n_points);

    for (size_t i = 0; i < points.size(); ++i) {
        std::cout << "Coordinate " << i << ": (" << points[i].x << ", " << points[i].y << ", " << points[i].z << ")\n";
    }

    return 0;
}
