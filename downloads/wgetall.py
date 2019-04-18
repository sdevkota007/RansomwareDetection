import subprocess
import os

malware_folder = "./malware_data"
normal_folder = "./normal_data"

with open('malware_dataset.txt', 'r') as f:
    malware_dataset = f.readlines()

with open('normal_dataset.txt', 'r') as f:
    normal_dataset = f.readlines()

if not os.path.isdir(malware_folder):
        os.mkdir(malware_folder)

if not os.path.isdir(normal_folder):
        os.mkdir(normal_folder)

for url in malware_dataset:
    if url.startswith("http"):
        url = url.strip('\n')
        subprocess.call(["wget","-P", malware_folder, "-nc", "-l 1", url])


for url in normal_dataset:
    if url.startswith("http"):
        url = url.strip('\n')
        subprocess.call(["wget","-P", normal_folder, "-nc", "-l 1", url])