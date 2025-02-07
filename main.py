from benchs import *
import multiprocessing
import psutil


def set_cpu_affinity(process_id, cpu_core):
    process = psutil.Process(process_id)
    process.cpu_affinity(cpu_core)


if __name__ == "__main__":
    num_processes = 4  # You can adjust the number of processes
    chunk_size = 700 // num_processes
    total_images_per_second_face = 0

    processes = []
    result_queue_face = multiprocessing.Queue()
    result_queue_obj = multiprocessing.Queue()

    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = start_index + chunk_size if i < num_processes - 1 else 700
        ret_value_face = multiprocessing.Value("d", 0.0, lock=False)
        p_face = multiprocessing.Process(target=BenchFaceDet, args=(start_index, end_index, result_queue_face))
        processes.append(p_face)
        p_face.start()

    for p in processes:
        p.join()

        individual_rate_face = result_queue_face.get()
        total_images_per_second_face += individual_rate_face

    # ----------------------------------------------------------------------------------

    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = start_index + chunk_size if i < num_processes - 1 else 700
        chunk = [(filename,
                  (random.randint(0, 400), random.randint(0, 300), random.randint(50, 150), random.randint(50, 150)))
                 for filename in os.listdir('more_imgs')]
        p_obj = multiprocessing.Process(target=BenchObjRem, args=(chunk, result_queue_obj))
        processes.append(p_obj)
        p_obj.start()

    for p in processes:
        p.join()

    # total_images_per_second_obj = 0
    total_megapixels_obj = 0
    images_per_second_obj, avg_megapixels_per_second_obj = result_queue_obj.get()
    # total_images_per_second_obj += images_per_second_obj
    total_megapixels_obj += avg_megapixels_per_second_obj

    print(f'Overall Face Detection Rate: {total_images_per_second_face:.2f} Images/s')
    print(f'Overall Object Removal Rate: {total_images_per_second_obj:.2f} Images/s')
    print(f'Overall Object Removal Megapixels Rate: {total_megapixels_obj:.2f} MPixels/s')
