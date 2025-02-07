import multiprocessing
from Crypto.Cipher import AES
import random
import cv2
import numpy as np
import os
import time
import gzip
import shutil

result_queue_face = multiprocessing.Queue()
result_queue_obj = multiprocessing.Queue()

num_processes = 4
chunk_size = 700 // num_processes

for i in range(num_processes):
    start_index = i * chunk_size
    end_index = start_index + chunk_size if i < num_processes - 1 else 700
    chunk = [(filename,
              (random.randint(0, 400), random.randint(0, 300), random.randint(50, 150), random.randint(50, 150)))
             for filename in os.listdir('more_imgs')]


def BenchFaceDet(num_b, num_a, res_queue):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image_directory = 'more_imgs/'
    image_files = os.listdir(image_directory)
    output_directory = 'output_images/'
    os.makedirs(output_directory, exist_ok=True)
    processed_images = 0

    def detect_faces(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return image

    start_time = time.time()

    for filename in image_files[num_b:num_a]:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_directory, filename)
            img = cv2.imread(image_path)
            detect_faces(img)
            processed_images += 1

            if processed_images % 20 == 0:
                print(f"Processed {processed_images} images.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    folder_path = "output_images/"

    shutil.rmtree(folder_path)

    images_per_second = len(image_files) / elapsed_time
    print(f'Bench Face Detection: {images_per_second:.2f} Images/s')
    res_queue.put(images_per_second)
    return images_per_second


def BenchObjRem(chunk, res_queue):
    global elapsed_time
    total_elapsed_time = 0
    total_megapixels = 0

    for filename, roi in chunk:
        input_image_path = os.path.join('more_imgs', filename)
        image = cv2.imread(input_image_path)
        output_image = image.copy()
        mask = np.zeros(image.shape[:2], dtype="uint8")

        start_time = time.time()

        cv2.inpaint(output_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        end_time = time.time()
        elapsed_time = end_time - start_time

        height, width, _ = image.shape
        total_megapixels += (height * width) / 1e6
        total_elapsed_time += elapsed_time

    avg_megapixels_per_second = total_megapixels / total_elapsed_time
    images_per_second = len(chunk) / total_elapsed_time

    print(f"Bench Object Removal: {avg_megapixels_per_second:.2f} MPixels/s")
    res_queue.put((images_per_second, avg_megapixels_per_second))


def BenchFileCompnDecomp():
    # set_cpu_affinity(os.getpid(), [7])
    file_size_bytes = 100 * 1024 * 1024
    file_size_mb = 100
    file_path = os.path.join("temp_file.txt")
    with open(file_path, 'wb') as file:
        file.write(os.urandom(file_size_bytes))

    start_time = time.time()

    with open(file_path, 'rb') as file:
        input_data = file.read()
    with gzip.open(file_path + '.gz', 'wb') as gzipped_file:
        gzipped_file.write(input_data)

    compression_time = time.time() - start_time
    start_time = time.time()

    with gzip.open(file_path + '.gz', 'rb') as gzipped_file:
        gzipped_file.read()
    decompression_time = time.time() - start_time

    os.remove(file_path + '.gz')
    os.remove(file_path)

    compression_speed_MBps = file_size_mb / compression_time
    decompression_speed_MBps = file_size_mb / decompression_time
    print(f"Compression Speed: {compression_speed_MBps:.2f} MB/s")
    print(f"Decompression Speed: {decompression_speed_MBps:.2f} MB/s")


def encrypt_data(data, key):
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce, ciphertext, tag


def benchmark_aes_encryption(data, key):
    iterations = 100000
    total_time = 0.0

    for _ in range(iterations):
        start_time = time.time()
        cipher = AES.new(key, AES.MODE_EAX)
        var = cipher.nonce
        cipher.encrypt_and_digest(data)
        end_time = time.time()
        total_time += end_time - start_time

    average_time = total_time / iterations
    bytes_encrypted = len(data) * iterations

    encryption_speed = bytes_encrypted / average_time
    encryption_speed /= 1000000

    return encryption_speed


if __name__ == "__main__":
    data = os.urandom(1024)
    key = os.urandom(32)
    encryption_speed = benchmark_aes_encryption(data, key)
    print(f"AES Encryption Speed: {encryption_speed:.2f} MBytes/second")
    BenchFileCompnDecomp()