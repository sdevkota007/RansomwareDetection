echo "Start to process the data......"
#use this to parse tcp header files
time python pcap_Parser.py -p > "parser.log"
time python load_shift.py
