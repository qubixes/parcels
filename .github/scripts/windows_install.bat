SET PATH=%PYTHON%;%PYTHON%\\Scripts;%PATH%
conda env create -f %ENV_NAME% -n parcels
activate parcels
python setup.py install
