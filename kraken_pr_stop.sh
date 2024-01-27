#!/bin/bash

UIID=$(pgrep -af python3 | grep kraken | grep -oP '^\d+')
sudo kill -64 $UIID
# Holy shit dash is sticky.  There has to be a cleaner way to do this
UIID=$(pgrep -af python3 | grep kraken | grep -oP '^\d+')
sudo kill -9 $UIID
/home/krakenrf/heimdall_daq_fw/Firmware/daq_stop.sh