SET PATH=%PYTHON%;%PYTHON%\Scripts;%PATH%
conda env create -f environment_py%PY_VERSION%_%OS_NAME%.yml -n parcels
call activate parcels
python setup.py install
