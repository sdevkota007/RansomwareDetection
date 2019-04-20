# Ransomware Detection using Deep Learning

Download the malware and normal dataset using
```bash
cd downloads
python wgetall.py
```

unzip all files inside "./downloads/malware_data" directory with password 'infected'
```
unzip -P infected \*.zip
```

Now, we can remove all of the unnecessary zip files inside "./downloads/malware_data" directory
```bash
rm *.zip
```

Move all of the malware pcap files into  directory "./malware_pcap/train_source/" 

and all of the normal pcap files into directory "./normal_pcap/train/"


Now, to parse http header from pcap files, run:
```bash
./extract_http.sh
```
If you want to parse tcp header instead of http, run this instead: 
```bash
./extract_tcp.sh
```

Use PCA to reduce dimensions of initial payloads.
```bash
python ./visual/pca.py ./dataset/train
```

It will produced a pickle file

Finally to run the deep neural network model
```bash
python simple_dnn.py
```