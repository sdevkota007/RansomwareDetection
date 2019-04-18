# RansomwareDetection
Ransomware Detection using Deep Learning


Download the malware and normal dataset using
```bash
cd downloads
python wgetall.py
```

Copy around 90% of the downloaded malware dataset into ./malware_pcap/train_source/ and remaining 10% to 
./malware_pcap/test_source/

Simarly, Copy around 90% of the downloaded normal dataset into ./normal_pcap/train_source/ and remaining 10% to 
./normal_pcap/test_source/

unzip all files in a directory with password
```
unzip -P <password> \*.zip
```


python pcap_Parser.py -p > "parser.log"
python load_shift.py