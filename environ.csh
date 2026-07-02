set VENVNAME=hgtcrn

module load /projects/dspsoftware/members/modulefiles/python/3.9.23


# deactivate current virtualenv
if (! ($?VIRTUAL_ENV)) then
    echo "no virtualenv active"
else
    echo "deactivating $VIRTUAL_ENV"
    deactivate
endif

# create virtualenv
if (! -d .venv/$VENVNAME ) then
    echo "creating new virtualenv"
    echo "present working dir"
    pwd
    mkdir -p .venv/$VENVNAME
    python3.9 -m venv .venv/$VENVNAME
endif

# activate virtualenv and install requirements
source .venv/$VENVNAME/bin/activate.csh
echo "activated $VIRTUAL_ENV"

pip install --upgrade pip

pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install einops
pip install soundfile



# deactivate and reactivate virtualenv to ensure that scripts such as pytest get added to the path
deactivate
source .venv/$VENVNAME/bin/activate.csh
