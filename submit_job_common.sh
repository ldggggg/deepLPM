#!/bin/bash

oarsub -p "gpucapability >= '5.0' and gpu='YES' and (host='nefgpu19.inria.fr' or host='nefgpu18.inria.fr' or host='nefgpu20.inria.fr' or host='nefgpu42.inria.fr' or host='nefgpu43.inria.fr' or host='nefgpu44.inria.fr' or host='nefgpu45.inria.fr' or host='nefgpu31.inria.fr' or host='nefgpu30.inria.fr' or host='nefgpu32.inria.fr')" -l /gpunum=2,walltime=50:00:00 -S "./run_network.sh "

# oarsub -p "gpu='NO' and mem > 160000" -l /core=6,walltime=10:00:00 -S "./run_network.sh"