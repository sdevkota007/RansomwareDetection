# RansomwareDetection
Ransomware Detection using Deep Learning


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

Move around 90% of the downloaded malware pcap files into ./malware_pcap/train_source/ and remaining 10% to 
./malware_pcap/test_source/

Simarly, move around 90% of the downloaded normal pcap files into ./normal_pcap/train/ and remaining 10% to 
./normal_pcap/test/


Now, to parse http header from pcap files, run:
```
python pcap_Parser.py > "parser.log" 
python load_shift.py
```
If you want to parse tcp header instead of http, run this instead: 
```
python pcap_Parser.py -p > "parser.log" 
python load_shift.py
```
