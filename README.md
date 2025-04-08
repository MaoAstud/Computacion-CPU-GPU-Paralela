
# 🧬 Procesamiento Paralelo de Genomas con CPU y GPU

Este proyecto permite contar la frecuencia de bases genéticas (A, T, C, G) en un archivo `.fna` utilizando técnicas de computación paralela con **CPU (Numba)** y **GPU (OpenCL)**. Se puede ejecutar en tres modos: CPU, GPU o ambos en paralelo.

---

## 📦 Requisitos

### 🔧 Instalación de librerías

Instala las dependencias con:

```bash
pip install numpy numba pyopencl
```

### ⚠️ Requisitos del sistema

- **Drivers actualizados** de la GPU.
- **OpenCL Runtime**:
  - [Intel OpenCL Runtime](https://www.intel.com/content/www/us/en/developer/tools/opencl/opencl-runtime.html)
  - [NVIDIA OpenCL](https://developer.nvidia.com/opencl)
  - [AMD OpenCL](https://gpuopen.com/compute-product/opencl-sdk/)

---

## 🚀 Ejecución

1. Coloca el archivo `.fna` del genoma en el mismo directorio del script.
2. Ejecuta el script principal:

```bash
python cpuGpuParallelProcessing.py
```

3. Selecciona el modo de procesamiento:
   - `1`: Solo CPU
   - `2`: Solo GPU
   - `3`: CPU + GPU

4. Ingresa el número de núcleos CPU o el tamaño de los work-items (GPU) según el modo elegido.

---

## 🛠 Estructura del proyecto

- `cpuGpuParallelProcessing.py`: Script principal con todo el procesamiento.
- `Informe_CPU_GPU_Paralela.docx`: Documento formal explicativo del proyecto.
- `README.md`: Este archivo.

---

## 📊 Tecnologías utilizadas

- **Numba**: Paralelismo en CPU con compilación JIT.
- **PyOpenCL**: Ejecución de kernels en GPU con OpenCL.
- **Threading**: Para ejecutar CPU y GPU simultáneamente.
- **Multiprocessing**: Para detectar núcleos disponibles.

---

## 📄 Autores

- Mao Astudillo
- Mateo Bonilla
- Gabriela Corella
- Arman Zargaran

Fecha: 04/08/2025
