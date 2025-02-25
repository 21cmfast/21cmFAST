#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "raytracing.cuh"
#include "memory.cuh"

// ===========================================================================
// ASORA Python C-extension module
// Mostly boilerplate code, this file contains the wrappers for python
// to access the C++ functions of the ASORA library. Care has to be taken
// mostly with the numpy array arguments, since the underlying raw C pointer
// is passed directly to the C++ functions without additional type checking.
// ===========================================================================

extern "C"
{   
    // ========================================================================
    // Raytrace all sources and compute photoionization rates
    // ========================================================================
    static PyObject *
    asora_do_all_sources(PyObject *self, PyObject *args)
    {
        double R;
        PyArrayObject * coldensh_out;
        double sig;
        double dr;
        PyArrayObject * ndens;
        PyArrayObject * xh_av;
        PyArrayObject * phi_ion;
        int NumSrc;
        int m1;
        double minlogtau;
        double dlogtau;
        int NumTau;

        if (!PyArg_ParseTuple(args,"dOddOOOiiddi",
        &R,
        &coldensh_out,
        &sig,
        &dr,
        &ndens,
        &xh_av,
        &phi_ion,
        &NumSrc,
        &m1,
        &minlogtau,
        &dlogtau,
        &NumTau))
            return NULL;
        
        // Error checking
        if (!PyArray_Check(coldensh_out) || PyArray_TYPE(coldensh_out) != NPY_DOUBLE)
        {
            PyErr_SetString(PyExc_TypeError,"coldensh_out must be Array of type double");
            return NULL;
        }

        // Get Array data
        double * coldensh_out_data = (double*)PyArray_DATA(coldensh_out);
        double * ndens_data = (double*)PyArray_DATA(ndens);
        double * phi_ion_data = (double*)PyArray_DATA(phi_ion);
        double * xh_av_data = (double*)PyArray_DATA(xh_av);

        do_all_sources_gpu(R,coldensh_out_data,sig,dr,ndens_data,xh_av_data,phi_ion_data,NumSrc,m1,minlogtau,dlogtau,NumTau);

        return Py_None;
    }

    // ========================================================================
    // Allocate GPU memory for grid data
    // ========================================================================
    static PyObject *
    asora_device_init(PyObject *self, PyObject *args)
    {
        int N;
        int num_src_par;
        if (!PyArg_ParseTuple(args,"ii",&N,&num_src_par))
            return NULL;
        device_init(N,num_src_par);
        return Py_None;
    }

    // ========================================================================
    // Deallocate GPU memory
    // ========================================================================
    static PyObject *
    asora_device_close(PyObject *self, PyObject *args)
    {
        device_close();
        return Py_None;
    }

    // ========================================================================
    // Copy density grid to GPU
    // ========================================================================
    static PyObject *
    asora_density_to_device(PyObject *self, PyObject *args)
    {
        int N;
        PyArrayObject * ndens;
        if (!PyArg_ParseTuple(args,"Oi",&ndens,&N))
            return NULL;

        double * ndens_data = (double*)PyArray_DATA(ndens);
        density_to_device(ndens_data,N);

        return Py_None;
    }

    // ========================================================================
    // Copy radiation table to GPU
    // ========================================================================
    static PyObject *
    asora_photo_table_to_device(PyObject *self, PyObject *args)
    {
        int NumTau;
        PyArrayObject * thin_table;
        PyArrayObject * thick_table;
        if (!PyArg_ParseTuple(args,"OOi",&thin_table,&thick_table,&NumTau))
            return NULL;

        double * thin_table_data = (double*)PyArray_DATA(thin_table);
        double * thick_table_data = (double*)PyArray_DATA(thick_table);
        photo_table_to_device(thin_table_data,thick_table_data,NumTau);

        return Py_None;
    }

    // ========================================================================
    // Copy source data to GPU
    // ========================================================================
    static PyObject *
    asora_source_data_to_device(PyObject *self, PyObject *args)
    {
        int NumSrc;
        PyArrayObject * pos;
        PyArrayObject * flux;
        if (!PyArg_ParseTuple(args,"OOi",&pos,&flux,&NumSrc))
            return NULL;

        int * pos_data = (int*)PyArray_DATA(pos);
        double * flux_data = (double*)PyArray_DATA(flux);

        source_data_to_device(pos_data,flux_data,NumSrc);

        return Py_None;
    }

    // ========================================================================
    // Define module functions and initialization function
    // ========================================================================
    static PyMethodDef asoraMethods[] = {
        {"do_all_sources",  asora_do_all_sources, METH_VARARGS,"Do OCTA raytracing (GPU)"},
        {"device_init",  asora_device_init, METH_VARARGS,"Free GPU memory"},
        {"device_close",  asora_device_close, METH_VARARGS,"Free GPU memory"},
        {"density_to_device",  asora_density_to_device, METH_VARARGS,"Copy density field to GPU"},
        {"photo_table_to_device",  asora_photo_table_to_device, METH_VARARGS,"Copy radiation table to GPU"},
        {"source_data_to_device",  asora_source_data_to_device, METH_VARARGS,"Copy radiation table to GPU"},
        {NULL, NULL, 0, NULL}        /* Sentinel */
    };

    static struct PyModuleDef asoramodule = {
        PyModuleDef_HEAD_INIT,
        "libasora",   /* name of module */
        "CUDA C++ implementation of the short-characteristics RT", /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
        asoraMethods
    };

    PyMODINIT_FUNC
    PyInit_libasora(void)
    {   
        PyObject* module = PyModule_Create(&asoramodule);
        import_array();
        return module;
    }
}