#!/bin/bash
condor_qedit $1 RequestDisk 2000M
condor_qedit $1 RequestMemory 2048M
condor_release $1



