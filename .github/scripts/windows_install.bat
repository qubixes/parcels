SET PATH=%PYTHON%;%PYTHON%\\Scripts;%PATH%
conda env create -f %ENV_NAME% -n parcels
conda activate parcels
python setup.py install
