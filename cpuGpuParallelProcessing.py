import os
import time
import numpy as np
import pyopencl as cl
from numba import jit, prange
from threading import Thread
import multiprocessing

# ========== Cargar archivo .fna ==========

def cargar_genoma(path):
    secuencia = []
    with open(path, 'r') as f:
        for linea in f:
            if linea.startswith('>'):
                continue
            secuencia.append(linea.strip().upper())
    return ''.join(secuencia)

# ========== CPU con Numba (paralelo) ==========

@jit(nopython=True, parallel=True)
def contar_bases_cpu(data):
    counts = np.zeros(4, dtype=np.int64)
    for i in prange(len(data)):
        c = data[i]
        if c == 'A':
            counts[0] += 1
        elif c == 'T':
            counts[1] += 1
        elif c == 'C':
            counts[2] += 1
        elif c == 'G':
            counts[3] += 1
    return counts

# ========== OpenCL kernel (GPU) ==========

kernel_code = """
__kernel void contar_bases(__global const uchar* data, __global int* resultado) {
    int gid = get_global_id(0);
    uchar c = data[gid];
    if (c == 'A') atomic_add(&resultado[0], 1);
    else if (c == 'T') atomic_add(&resultado[1], 1);
    else if (c == 'C') atomic_add(&resultado[2], 1);
    else if (c == 'G') atomic_add(&resultado[3], 1);
}
"""

# ========== GPU por chunks ==========

def contar_bases_gpu_opencl(data_chunk, chunk_size=1000000):
    total_resultado = np.zeros(4, dtype=np.int32)
    arr = np.frombuffer(data_chunk.encode('ascii'), dtype=np.uint8)

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    program = cl.Program(ctx, kernel_code).build()

    for start in range(0, len(arr), chunk_size):
        end = min(start + chunk_size, len(arr))
        chunk = arr[start:end]
        resultado = np.zeros(4, dtype=np.int32)

        data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chunk)
        result_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=resultado)

        global_size = (len(chunk),)
        program.contar_bases(queue, global_size, None, data_buf, result_buf)
        cl.enqueue_copy(queue, resultado, result_buf)
        queue.finish()

        total_resultado += resultado

    return total_resultado

# ========== Función principal ==========

def main():
    ruta = "GCA_000001405.29_GRCh38.p14_genomic.fna"
    if not os.path.exists(ruta):
        print("Archivo no encontrado.")
        return

    # Mostrar núcleos disponibles
    cpu_count = multiprocessing.cpu_count()
    try:
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        max_gpu_work_items = device.max_work_group_size
        gpu_name = device.name
    except Exception as e:
        max_gpu_work_items = "Desconocido"
        gpu_name = "No detectada"

    print(f"\n🧠 Núcleos CPU disponibles: {cpu_count}")
    print(f"🖥️  GPU detectada: {gpu_name}")
    print(f"⚙️  Máx. work-items recomendados por batch (GPU): {max_gpu_work_items}\n")

    print("Elige el modo de ejecución:")
    print("1. Solo CPU")
    print("2. Solo GPU (OpenCL)")
    print("3. CPU + GPU en paralelo")
    modo = input("Opción [1/2/3]: ").strip()

    print("Cargando genoma...")
    genoma = cargar_genoma(ruta)
    print(f"Genoma cargado ({len(genoma)} bases)")

    resultado_cpu = np.zeros(4, dtype=np.int64)
    resultado_gpu = np.zeros(4, dtype=np.int32)

    # ===== Mover el cronómetro aquí, justo después de elegir parámetros =====
    if modo == '1':
        nucleos = int(input(f"Número de núcleos a usar (máximo {cpu_count}): "))
        os.environ["NUMBA_NUM_THREADS"] = str(nucleos)
        os.environ["OMP_NUM_THREADS"] = str(nucleos)
        print(f"Procesando solo en CPU con {nucleos} núcleo(s)...")
        start = time.time()
        resultado_cpu = contar_bases_cpu(genoma)

    elif modo == '2':
        chunk_size = int(input(f"Tamaño del chunk GPU (work-items por batch, ej. 100000): "))
        if chunk_size > len(genoma):
            print(f"⚠️  El chunk es más grande que el genoma. Se ajustará automáticamente a {len(genoma)}")
            chunk_size = len(genoma)
        print("Procesando solo en GPU con OpenCL...")
        start = time.time()
        resultado_gpu = contar_bases_gpu_opencl(genoma, chunk_size)

    elif modo == '3':
        nucleos = int(input(f"Número de núcleos CPU a usar (máximo {cpu_count}): "))
        chunk_size = int(input(f"Tamaño del chunk GPU (work-items por batch, ej. 100000): "))
        os.environ["NUMBA_NUM_THREADS"] = str(nucleos)
        os.environ["OMP_NUM_THREADS"] = str(nucleos)
        print("Procesando en paralelo: CPU + GPU...")

        mitad = len(genoma) // 2
        cpu_chunk = genoma[:mitad]
        gpu_chunk = genoma[mitad:]

        if chunk_size > len(gpu_chunk):
            print(f"⚠️  El chunk es más grande que la parte del genoma asignada a la GPU. Se ajustará a {len(gpu_chunk)}")
            chunk_size = len(gpu_chunk)

        start = time.time()

        def tarea_cpu():
            nonlocal resultado_cpu
            resultado_cpu = contar_bases_cpu(cpu_chunk)

        def tarea_gpu():
            nonlocal resultado_gpu
            resultado_gpu = contar_bases_gpu_opencl(gpu_chunk, chunk_size)

        hilo_cpu = Thread(target=tarea_cpu)
        hilo_gpu = Thread(target=tarea_gpu)

        hilo_cpu.start()
        hilo_gpu.start()

        hilo_cpu.join()
        hilo_gpu.join()

    else:
        print("Opción no válida.")
        return

    total = resultado_cpu + resultado_gpu
    elapsed = time.time() - start

    print("\nFrecuencia de bases totales:")
    print(f"A: {total[0]}")
    print(f"T: {total[1]}")
    print(f"C: {total[2]}")
    print(f"G: {total[3]}")
    print(f"\n⏱️ Tiempo total de ejecución: {elapsed:.2f} segundos")



if __name__ == "__main__":
    main()
