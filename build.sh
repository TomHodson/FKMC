clear
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

##note this bug https://bugs.python.org/issue1011113 which may mean that python setup.py install ignores --build-base

######################## CMTH specific stuff ########################################
if [ "$HOME" = "/home/tch14" ]; then

echo -e "${GREEN}Machine recognised as CMTH${NC}"
. /workspace/tch14/miniconda3/etc/profile.d/conda.sh
echo -e "sourced /workspace/tch14/miniconda3/etc/profile.d/conda.sh${NC}"
conda activate cmth_intelpython3_2
echo -e "activated cmth_intelpython3 conda environment${NC}"
DIR="./cmth"

###################### CX1 specific stuff ########################################
elif [ "$HOME" = "/rds/general/user/tch14/home" ]; then

echo -e "${GREEN}Machine recognised as CX1${NC}"

module load intel-suite anaconda3/personal
echo "Loaded intel-suite and anaconda3/personal"
. /apps/jupyterhub/2018-02-19/anaconda/etc/profile.d/conda.sh
conda activate /home/tch14/anaconda3/envs/idp
echo "activated idp conda environment"
source $MKL_HOME/bin/mklvars.sh intel64
echo "sourced mklvars.sh"

#doing this fixes a bug with the fact that on cx1 /tmp is on a differnt filesystem to ~
TMPDIR="./cx1/tmp" 
DIR="./cx1"

else
echo "Machine not recognised as cx1 or cmth"
fi

###################### general stuff ########################################

echo -e "${GREEN}Numpy include dirs are:${NC}"
echo $(python -c 'import numpy; print(numpy.get_include())')

echo -e "${GREEN}Starting Build${NC}"
mkdir -p ${DIR}_build
mkdir -p ${DIR}_dist

#python setup.py build --build-base=$BUILD_DIR bdist_wheel
#python setup.py build install

python setup.py build --build-base=${DIR}_build bdist_wheel --dist-dir=${DIR}_dist

echo 'Wheel file path is ' ${DIR}_dist/$(ls -t ${DIR}_dist | head -n 1)
pip uninstall -y FKMC
pip install ${DIR}_dist/$(ls -t ${DIR}_dist | head -n 1)

#mv /workspace/tch14/conda-envs/cmth_intelpython3/lib/python3.6/site-packages/FKMC-0-py3.6-linux-x86_64.egg/MC /workspace/tch14/conda-envs/cmth_intelpython3/lib/python3.6/site-packages/FKMC-0-py3.6-linux-x86_64.egg/FKMC 

mv FKMC/*.c ${DIR}_build
mv FKMC/*.html ${DIR}_build