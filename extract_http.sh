echo "Start to process the data......"
#use this to parse http header files
time python pcap_Parser.py > "parser.log"
time python load_shift.py
