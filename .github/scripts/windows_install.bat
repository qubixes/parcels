SET PATH=%PYTHON%;%PYTHON%\\Scripts;%PATH%
conda env create -f %ENV_NAME% parcels
activate parcels
python setup.py install
