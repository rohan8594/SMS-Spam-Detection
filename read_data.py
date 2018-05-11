# File:        read_data.py
#
# Author:      Rohan Patel
#
# Date:        05/09/2018
#
# Description: This script simply reads the sms data from the SMSSpamCollection file, prints the total number of
#              sms messages in the dataset, and then individually prints the first 100 lines from the SMSSpamCollection
#              file. The purpose of this script is to simply give an initial idea of how the sms data is organized in 
#              then dataset.

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]

print('\nTotal number of messages:' + str(len(messages)))
print('\n')

for messno, msg in enumerate(messages[:100]):
    print(messno, msg)
    print('\n')