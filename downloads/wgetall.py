import subprocess


with open('malware_dataset.txt', 'r') as f:
    malware_dataset = f.readlines()

with open('normal_dataset.txt', 'r') as f:
    normal_dataset = f.readlines()


for url in malware_dataset:
    if url.startswith("http"):
        url = url.strip('\n')
        subprocess.call(["wget", "-r", "-nc", "-l 1", url])


for url in normal_dataset:
    if url.startswith("http"):
        url = url.strip('\n')
        subprocess.call(["wget", "-r", "-nc", "-l 1", url])