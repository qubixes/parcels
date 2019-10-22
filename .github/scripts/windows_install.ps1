$env:Path += ";$PYTHON;$PYTHON\Scripts"
conda env create -f $ENV_NAME -n parcels
activate parcels
python setup.py install
